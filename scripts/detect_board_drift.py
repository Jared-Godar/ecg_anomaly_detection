#!/usr/bin/env python3
"""Detect open issues/PRs missing Project #5 membership or label-derivable fields.

The scheduled verification backstop behind the creation-time board automation
(issue #233): `project-item-autofill.yml` populates the board when items are
opened or labeled, but an automation outage, a race, or an item created
through a surface that never fired the events would otherwise drift silently.
This script runs from `repository-hygiene.yml` on the weekly schedule (and
manually) and flags, for every OPEN issue and pull request:

- missing Project #5 membership entirely;
- an unset Status field (every tracked item must occupy a lane);
- any field that is unset even though the item carries a label the shared
  mapping table (`scripts/github/project_label_mapping.py`) derives it from;
  and
- a milestone <-> Target Release incoherence (issue #240, the first tier-3
  migration under the verification graduation ladder): a milestone/Target
  Release pair outside the enumerated coherence table
  (`scripts/github/milestone_release_mapping.py`), a milestoned item whose
  Target Release is unset, an unmilestoned item carrying a release-vehicle
  Target Release, or a milestone the table has no row for.

It deliberately does NOT flag a populated field whose value differs from the
label-derived one: curated values win (house rule from AGENTS.md), so a
divergence is legitimate maintainer judgment, not drift. Workstream (the
other label-less field) stays unchecked, and Target Release is only
cross-checked against the milestone, never required outright -- full
nine-field completeness is the PR-time metadata gate's job
(docs/governance/github-metadata-automation.md); this backstop watches
exactly the surface the creation-time automation owns plus the enumerated
coherence invariant above. Governed-bot (Dependabot) items are excluded,
mirroring the automation's own skip.

This is read-only: it never adds items or mutates fields. Remediation is a
manual `populate_project_item.py` run (or `gh project item-add`/`item-edit`)
-- see docs/governance/github-metadata-automation.md's creation-time
population section for the fallback commands.

Quota stewardship (issue #173): the run preflights the shared GraphQL pool
and prints a before/after/consumed report. Unlike the sibling
`detect_label_drift.py` (observe-only default 0, tiny listings), this script
pays for one full board snapshot (~203 points measured live 2026-07-12), so
its preflight default matches the validator's 250: starting a snapshot-sized
read on a pool that cannot serve it would fail mid-run anyway.

Exit codes: 0 no drift, 1 drift detected, 2 a genuine failure
(authentication, gh CLI, malformed data), 3 a GraphQL quota condition --
transient shared-pool infrastructure, never board drift itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Repository root, resolved from this script's own location rather than the current
# working directory, so the script behaves identically regardless of where it's invoked from.
ROOT = Path(__file__).resolve().parents[1]

# The shared GitHub access layer and the label->field mapping table live in
# scripts/github/ (operational tooling, not an installed package), one
# directory below this script, so they are imported by putting that directory
# on sys.path first -- the same file-system-adjacency convention
# detect_label_drift.py uses. The guard keeps the insertion idempotent when
# several of these scripts are loaded into one process (e.g. by the test suite).
_GITHUB_SCRIPTS_DIR = str(ROOT / "scripts" / "github")
# Only insert when absent, so repeated loads never stack duplicate entries.
if _GITHUB_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _GITHUB_SCRIPTS_DIR)

import github_api  # noqa: E402  (needs the sys.path insertion above)
import milestone_release_mapping  # noqa: E402  (needs the sys.path insertion above)
import project_label_mapping  # noqa: E402  (needs the sys.path insertion above)

# This script's own preflight threshold: the board snapshot it takes measured
# 203 GraphQL points live (2026-07-12, recorded in
# docs/governance/github-metadata-automation.md), so 250 covers it with
# margin -- the same sizing rationale as validate_project_metadata.py, and
# deliberately NOT the observe-only 0 its listing-only sibling
# detect_label_drift.py uses, because starting a snapshot on a pool that
# cannot serve it would fail mid-run anyway.
_MIN_GRAPHQL_QUOTA_DEFAULT: int = 250

# gh CLI normalizes each Project field's JSON key in item-list output to a
# lowercase-first-word version of its display name (observed directly by
# validate_project_metadata.py: "Issue Type" -> "issue Type", "Repository
# Area" -> "repository Area"); this helper mirrors that observed convention.
_STATUS_FIELD_NAME: str = "Status"

# gh's author objects for bot accounts report is_bot=true with the bare app
# slug as the login (observed: Dependabot lists as login "app/dependabot" in
# some surfaces and "dependabot" in list JSON, while REST reports
# "dependabot[bot]"); all three spellings are accepted, each still requiring
# the is_bot flag, so a human account with a lookalike login is never skipped.
_GOVERNED_BOT_LOGINS: frozenset[str] = frozenset(
    {"dependabot", "app/dependabot", "dependabot[bot]"}
)


class BoardDriftError(github_api.GitHubApiError):
    """Raised when required GitHub data for the drift check cannot be read.

    Subclasses the shared GitHubApiError so main()'s single except clause
    catches script-level defects and access-layer failures uniformly.
    """


@dataclass(frozen=True, slots=True)
class DriftFinding:
    """One open issue/PR with a board-membership or derivable-field gap."""

    number: int
    kind: str
    title: str
    problems: tuple[str, ...]


def _field_json_key(field_name: str) -> str:
    """Return gh item-list's JSON key for a Project field's display name.

    Args:
        field_name: The field's display name, e.g. "Issue Type".

    Returns:
        The gh JSON key, e.g. "issue Type" (lowercase first character only --
        the convention validate_project_metadata.py observed directly in gh
        output).
    """

    return field_name[0].lower() + field_name[1:]


def is_governed_bot_author(author: dict[str, Any] | None) -> bool:
    """Return whether a gh list author object identifies a governed bot.

    Args:
        author: The `author` object from `gh issue list`/`gh pr list` JSON
            output, or None when the account was deleted.

    Returns:
        True when the author is a governed bot (is_bot flag AND an accepted
        Dependabot login spelling); False otherwise.
    """

    # A deleted author is not a bot; the item still deserves board hygiene.
    if not author:
        return False
    # Both signals are required: is_bot is server-derived, and the login
    # match narrows the skip to exactly the governed Dependabot identity.
    return bool(author.get("is_bot")) and author.get("login") in _GOVERNED_BOT_LOGINS


def find_board_drift(
    open_items: Sequence[dict[str, Any]],
    project_items: Sequence[dict[str, Any]],
) -> tuple[DriftFinding, ...]:
    """Cross-check every open issue/PR against the board snapshot for gaps.

    Args:
        open_items: Open issues/PRs as dicts with `kind` ("issue" or
            "pull request"), `number`, `title`, `labels` (name strings),
            `author` (gh's author object or None), and `milestone` (the
            milestone title string, or None when unmilestoned).
        project_items: The Project's full item-list snapshot, in gh's
            item-list JSON shape.

    Returns:
        One DriftFinding per open item with at least one gap, in the input
        order, each carrying every problem found for that item.
    """

    # Index the board snapshot by (content type, number) once, so the check
    # over N open items costs one pass over the snapshot rather than N scans.
    board_index: dict[tuple[str, int], dict[str, Any]] = {}
    # One pass over the snapshot builds the whole index.
    for item in project_items:
        content = item.get("content") or {}
        content_type = content.get("type")
        content_number = content.get("number")
        # Draft items have no repository content to match; anything without
        # both a type and a number cannot correspond to an open issue/PR.
        if isinstance(content_type, str) and isinstance(content_number, int):
            board_index[(content_type, content_number)] = item

    findings: list[DriftFinding] = []
    # Check every open item independently, accumulating one finding per item
    # that has at least one gap.
    for open_item in open_items:
        # Governed-bot items are the Dependabot autofill path's territory;
        # flagging them here would duplicate that path's own governance.
        if is_governed_bot_author(open_item.get("author")):
            continue
        # gh's item-list content types are "Issue" and "PullRequest"; map the
        # listing's human kind onto that vocabulary for the index lookup.
        content_type = "Issue" if open_item["kind"] == "issue" else "PullRequest"
        board_item = board_index.get((content_type, open_item["number"]))
        # Missing membership makes every field check moot; report it alone.
        if board_item is None:
            findings.append(
                DriftFinding(
                    open_item["number"],
                    open_item["kind"],
                    open_item["title"],
                    ("not a member of the tracked Project",),
                )
            )
            continue
        problems: list[str] = []
        # Every tracked item must occupy a Status lane; the automation's
        # Backlog default (and every manual transition) should have set one.
        if not board_item.get(_field_json_key(_STATUS_FIELD_NAME)):
            problems.append("Status is unset")
        # Only the label-derivable fields are checked, and only for being
        # unset: a populated field that differs from the derivation is
        # curated maintainer judgment, deliberately not flagged.
        derived, _conflicts = project_label_mapping.derive_field_options(open_item["labels"])
        # Walk each cleanly derived field and compare against the snapshot.
        for field_name, option_name in derived.items():
            # Empty/missing means unset; any populated value is curated and fine.
            if not board_item.get(_field_json_key(field_name)):
                problems.append(f"{field_name} is unset despite a label deriving {option_name!r}")
        # Milestone <-> Target Release coherence (issue #240): both sides are
        # normalized to None when absent (an item created without a milestone
        # carries no milestone key at all in some callers; an unset board
        # field is missing from the snapshot row), then the enumerated
        # coherence table judges the pair. The check itself lives in the
        # shared mapping module so its table stays pinned by completeness
        # tests, exactly like the label table this loop consumes.
        target_release = (
            board_item.get(_field_json_key(milestone_release_mapping.TARGET_RELEASE_FIELD)) or None
        )
        problems.extend(
            milestone_release_mapping.coherence_problems(open_item.get("milestone"), target_release)
        )
        # Zero problems means this item is fully converged; report nothing.
        if problems:
            findings.append(
                DriftFinding(
                    open_item["number"],
                    open_item["kind"],
                    open_item["title"],
                    tuple(problems),
                )
            )
    return tuple(findings)


def fetch_open_items(repo: str | None) -> list[dict[str, Any]]:
    """Fetch open issues and pull requests with labels and author via the gh CLI.

    The subprocess plumbing (retry classification, error translation) lives in
    the shared access layer's run_gh; failures surface as
    github_api.GitHubApiError for main() to map.

    Args:
        repo: The "OWNER/REPO" slug, or None to let gh infer it from the
            current directory's Git remote.

    Returns:
        One dict per open item with `kind`, `number`, `title`, `labels`
        (name strings), `author` (gh's author object), and `milestone` (the
        milestone title, or None when unmilestoned).
    """

    items: list[dict[str, Any]] = []
    # gh uses separate subcommands for issues and pull requests; query both so
    # the drift check covers the whole repository, not just one kind of item.
    for kind, subcommand in (("issue", "issue"), ("pull request", "pr")):
        args = [
            subcommand,
            "list",
            "--state",
            "open",
            "--json",
            "number,title,labels,author,milestone",
            "--limit",
            "500",
        ]
        # repo is optional; omitting --repo lets gh infer it from the current
        # directory's Git remote, matching gh's own default behavior.
        if repo:
            args.extend(["--repo", repo])
        payload = json.loads(github_api.run_gh(args))
        # Flatten gh's nested label objects into plain name strings, flatten
        # the milestone object to its title (gh reports null for an
        # unmilestoned item), and tag each row with its kind, before
        # appending to the combined items list.
        for row in payload:
            items.append(
                {
                    "kind": kind,
                    "number": row["number"],
                    "title": row["title"],
                    "labels": [label["name"] for label in row.get("labels", [])],
                    "author": row.get("author"),
                    "milestone": (row.get("milestone") or {}).get("title"),
                }
            )
    return items


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the board-drift detection entry point.

    Args:
        argv: Optional command-line arguments; defaults to the process arguments.

    Returns:
        Parsed arguments: optional target repo, project owner, project
        number, and the minimum-remaining GraphQL quota threshold.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", help="GitHub OWNER/REPO; defaults to the current repository")
    parser.add_argument("--owner", default="Jared-Godar", help="Project owner login")
    parser.add_argument("--project-number", type=int, default=5)
    parser.add_argument(
        "--min-graphql-quota",
        type=int,
        default=_MIN_GRAPHQL_QUOTA_DEFAULT,
        help=(
            "Minimum remaining GraphQL points required by the preflight check before "
            "the board snapshot is fetched; 0 or below disables the stop (the "
            "before/after report is still printed). Default: "
            f"{_MIN_GRAPHQL_QUOTA_DEFAULT} (sized to the ~203-point snapshot, "
            "like the metadata validator)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line entry point and return its process exit status.

    Keeping orchestration here makes terminal behavior and error translation straightforward
    to audit.

    Args:
        argv: Optional command-line arguments; defaults to the process arguments.

    Returns:
        The value produced by the documented operation.
    """

    args = parse_args(argv)

    # One monitor per run: preflight before the GraphQL-backed listings and
    # snapshot, report after -- both reads are REST and free, so accounting
    # can never worsen the quota.
    monitor = github_api.QuotaMonitor(minimum_remaining=args.min_graphql_quota)
    # One client per run = one logical phase: the board snapshot behind
    # client.items() is fetched at most once and cached on the client.
    client = github_api.ProjectClient(args.owner, args.project_number)

    # Every failure mode this script can hit (gh CLI, quota) is part of the
    # GitHubApiError hierarchy -- BoardDriftError subclasses it -- so it is
    # caught once here for uniform error reporting, with the two quota
    # conditions mapped to their own exit code so hygiene output can never
    # conflate a drained shared pool with genuine board drift.
    try:
        monitor.preflight()
        open_items = fetch_open_items(args.repo)
        # The one full board snapshot this run is allowed (issue #173's
        # one-snapshot-per-phase rule); every membership and field check
        # below reads from this cached result.
        project_items = client.items()
    except (
        github_api.GraphQLQuotaInsufficientError,
        github_api.PrimaryRateLimitError,
    ) as error:
        # Quota conditions get their own exit code (3) and wording: transient
        # shared-pool infrastructure, resumable by rerunning after the reset,
        # never evidence about the board.
        print(f"quota: {error}", file=sys.stderr)
        return 3
    except github_api.GitHubApiError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    finally:
        # The consumption report prints on success and failure alike -- a
        # failed run's consumption is exactly the evidence needed when
        # diagnosing a drained pool. A run that failed before preflight ever
        # recorded a baseline has nothing to report; otherwise reporting is
        # best-effort, because a report failure must never mask the run's
        # real outcome (including a drift finding).
        if monitor.preflighted:
            # Best-effort only: see the comment above for why a report failure
            # is downgraded to a warning instead of changing the exit code.
            try:
                print(monitor.report(), file=sys.stderr)
            except github_api.GitHubApiError as report_error:
                print(
                    f"warning: could not report quota consumption: {report_error}",
                    file=sys.stderr,
                )

    findings = find_board_drift(open_items, project_items)
    # A non-empty findings list means at least one open item has a board gap;
    # report every one explicitly (never auto-fix) and exit non-zero so the
    # scheduled hygiene run surfaces it.
    if findings:
        print(f"Board drift detected on {len(findings)} item(s):", file=sys.stderr)
        # List every drifted item's own problems, so a reviewer can act on
        # the complete picture without re-running with different flags.
        for finding in findings:
            print(
                f"  - #{finding.number} ({finding.kind}) {finding.title!r}: "
                + "; ".join(finding.problems),
                file=sys.stderr,
            )
        print(
            "\nRemediation is manual and read-back-verified: run "
            "`uv run python scripts/github/populate_project_item.py` for the "
            "flagged item, or use the fallback commands in "
            "docs/governance/github-metadata-automation.md. For a milestone "
            "<-> Target Release incoherence, the maintainer decides which "
            "side is wrong (the milestone, the field, or a missing table row "
            "in scripts/github/milestone_release_mapping.py) -- this check "
            "never mutates the board itself.",
            file=sys.stderr,
        )
        return 1

    print("No board drift detected.")
    return 0


# Standard script entry-point guard: only run main() when executed directly, not when
# imported (e.g. by this script's own test module).
if __name__ == "__main__":
    raise SystemExit(main())
