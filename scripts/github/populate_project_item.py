#!/usr/bin/env python3
"""Add an issue or pull request to Project #5 and fill its label-derivable fields.

The creation-time board automation decided in issue #233: Project #5
membership and field population were entirely manual at item-creation time,
which repeatedly produced items with zero board presence (issue #210 at
filing; the ten #218-audit findings issues #223-#232, whose manual back-fill
took ~100 GraphQL operations and was interrupted mid-batch). This script runs
from the `project-item-autofill` workflow on `issues`/`pull_request`
`opened` and `labeled` events and converges the item toward board
completeness:

- adds the item to Project #5 when absent (idempotent -- verified by a
  targeted re-lookup, never by item-add's exit status);
- sets Status to the documented new-item default, Backlog, ONLY when Status
  is unset -- the automation never regresses a lane a human or a built-in
  workflow already chose;
- fills every field derivable from the item's current labels via the shared
  mapping table (`project_label_mapping.py`: `type:` -> Issue Type,
  `priority:` -> Priority, `risk:` -> Risk, `size:` -> Size, `area:` ->
  Repository Area, `portfolio:` -> Portfolio Signal), again ONLY where the
  field is currently unset -- curated values always win (house rule from
  AGENTS.md); and
- reports (without failing) any label combination whose derived options
  conflict, leaving those fields for maintainer review.

Workstream and Target Release have no label source and are an explicit
non-goal (issue #233): they stay human-set, and the PR-time metadata gate
remains the enforcement point for full nine-field completeness.

Governed-bot (Dependabot) items are skipped -- verified server-side via REST
exactly like `sync_dependabot_pr_metadata.py`'s inverse guard -- because the
Dependabot autofill path owns their board stamping with different,
bot-specific defaults.

Option IDs are always resolved by name at runtime (one cached field-list read
per run) and never hardcoded: the 2026-07-14 board-wide option-ID
regeneration (docs/governance/github-project.md) proved stored IDs go stale
wholesale.

Quota stewardship (issue #173): one GraphQL quota preflight before any work,
targeted content -> project-item lookups (never a full board snapshot), a
fresh targeted read-back after every actual mutation, and a
before/after/consumed report printed on success and failure alike. A full
run costs roughly 10-25 points (lookup, optional add with verifying
re-lookup, one field-list schema read, and at most seven edit-plus-read-back
pairs), so the preflight default of 50 gives a 2x margin.

Exit codes follow the house convention: 0 success (including the governed-bot
skip), 1 policy/validation failure (a mutation whose read-back never
confirmed), 2 required data unreadable (gh/API failures, a content-type
mismatch, an item-add the verifying re-lookup could not find), 3 a GraphQL
quota condition (transient shared-pool infrastructure, never a defect in the
item being populated).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# scripts/github/ is operational tooling, not an installed package, so the
# shared helpers that live alongside this script are imported by putting this
# script's own directory on sys.path first. That is already true when the
# script runs directly (sys.path[0] is the script's directory) but not when
# the test suite loads this file from its path, so the insertion is explicit
# and idempotent.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
# Guard against duplicate insertion when several governance scripts are loaded
# into one process (e.g. the test suite imports each of them).
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import github_api  # noqa: E402  (needs the sys.path insertion above)
import project_label_mapping  # noqa: E402  (needs the sys.path insertion above)

# This script's own preflight threshold: a full run costs roughly 10-25
# GraphQL points (one targeted content-item lookup, an optional item-add with
# its verifying re-lookup, one field-list schema read, and up to seven
# edit-plus-read-back pairs at ~1 point per read), so 50 gives a 2x margin.
# Deliberately far below the shared github_api.DEFAULT_MINIMUM_GRAPHQL_QUOTA
# (sized for a ~203-point board snapshot this script never takes): requiring
# that much headroom would block a cheap populate run that a moderately
# drained pool could easily serve.
_MIN_GRAPHQL_QUOTA_DEFAULT: int = 50

# The board's documented default lane for newly created items
# (docs/governance/github-project.md#status-lifecycle: "New project items
# begin in Backlog"). Set only when Status is unset, so the automation can
# never regress a lane a human, an agent, or a built-in workflow already
# advanced.
_DEFAULT_STATUS_OPTION: str = "Backlog"

# The governed bot authors whose items this script must NOT stamp, because
# the Dependabot autofill path (sync_dependabot_pr_metadata.py) owns their
# board metadata with bot-specific defaults. Mirrors
# validate_project_metadata.py's GOVERNED_BOT_AUTHORS; membership requires
# BOTH the exact login AND GitHub's server-assigned Bot account type, so a
# user account registered with a lookalike login cannot trigger the skip.
_GOVERNED_BOT_LOGINS: tuple[str, ...] = ("dependabot[bot]",)
# The required account type for a governed-bot skip. GitHub assigns `Bot`
# server-side and it cannot be self-selected.
_GOVERNED_BOT_ACCOUNT_TYPE: str = "Bot"


@dataclass(frozen=True, slots=True)
class ContentInfo:
    """One issue/PR's population-relevant metadata from the REST content read."""

    labels: tuple[str, ...]
    author_login: str | None
    author_type: str | None


class PopulatePolicyError(github_api.GitHubApiError):
    """Raised for policy/validation failures that must map to exit code 1.

    Currently one condition per the house exit-code convention: a field
    mutation whose targeted read-back never confirmed the requested value
    within the bounded retry. Subclasses the shared GitHubApiError so main()'s
    handlers stay a simple most-specific-first chain.
    """


def resolve_repository(repo: str | None) -> str:
    """Return the OWNER/REPO slug, asking gh to resolve it when not supplied.

    The targeted project-item lookup, the REST content read, and the item-add
    URL all need a concrete slug, but local operators inside the checkout
    should not have to spell it out -- gh already knows it from the
    surrounding Git checkout.

    Args:
        repo: The caller-supplied "OWNER/REPO" slug, or None to resolve it.

    Returns:
        The concrete "OWNER/REPO" slug.
    """

    # An explicit slug (the workflow always passes one) skips the extra call.
    if repo is not None:
        return repo
    # gh resolves the repository from the current checkout's Git remotes; the
    # JSON format keeps parsing exact rather than scraping human output.
    payload = json.loads(github_api.run_gh(["repo", "view", "--json", "nameWithOwner"]))
    return payload["nameWithOwner"]


def fetch_content(repo: str, number: int, content_kind: str) -> ContentInfo:
    """Read the item's labels and author over REST, cross-checking its content kind.

    Uses the REST issues endpoint, which serves both issues and pull requests
    (a PR row carries a `pull_request` key), keeping this native-metadata read
    off the shared GraphQL pool entirely (issue #173's REST-for-native-reads
    rule). The declared `--content-type` is cross-checked against what GitHub
    actually returned, so a mis-wired workflow invocation fails loudly instead
    of stamping the wrong kind of item.

    Args:
        repo: The content's "OWNER/REPO" slug.
        number: The issue or pull-request number in that repository.
        content_kind: The caller-declared kind, "issue" or "pull-request".

    Returns:
        The item's labels and author identity as a typed ContentInfo.

    Raises:
        github_api.GitHubApiError: The declared kind does not match what
            GitHub returned for this number (exit code 2 -- an invocation
            defect, not a transient condition).
    """

    # REST, not GraphQL: native metadata reads must not draw from the shared
    # GraphQL pool this script's preflight budgets for.
    payload = json.loads(github_api.run_gh(["api", f"repos/{repo}/issues/{number}"]))
    # The REST issues endpoint marks pull requests with a `pull_request` key;
    # its absence means the number is a plain issue.
    actual_kind = "pull-request" if "pull_request" in payload else "issue"
    # A kind mismatch means the invocation is mis-wired (e.g. an issues event
    # passed a PR number); stamping anyway could put the wrong defaults on
    # the wrong item, so this is a hard stop.
    if actual_kind != content_kind:
        raise github_api.GitHubApiError(
            f"#{number} in {repo} is a {actual_kind}, but the invocation declared "
            f"{content_kind!r}; refusing to populate a mis-declared item"
        )
    # A deleted author account leaves `user` null; treated as an ordinary
    # (non-bot) author, so the item still gets populated.
    author = payload.get("user") or {}
    return ContentInfo(
        labels=tuple(label["name"] for label in payload.get("labels", [])),
        author_login=author.get("login"),
        author_type=author.get("type"),
    )


def is_governed_bot_author(author_login: str | None, author_type: str | None) -> bool:
    """Return whether an item's author is a governed bot this script must skip.

    Args:
        author_login: The author's login as reported by GitHub, if any.
        author_type: The author's account type as reported by GitHub, if any.

    Returns:
        True when the author is a governed bot (exact login match AND the
        server-assigned Bot account type); False otherwise.
    """

    # Both checks are required: the `Bot` account type cannot be
    # self-assigned, so it defends against a user account registered with a
    # bot-lookalike login.
    return author_login in _GOVERNED_BOT_LOGINS and author_type == _GOVERNED_BOT_ACCOUNT_TYPE


def populate_item(
    client: github_api.ProjectClient,
    item: github_api.ProjectItemRef,
    labels: Sequence[str],
    project_number: int,
) -> tuple[int, int]:
    """Fill the item's unset default-Status and label-derivable fields, verified.

    Every write goes through the shared curated-values-win primitive
    (`ProjectClient.set_single_select_if_unset`): a field holding ANY value is
    preserved untouched, a blank field's write is verified by a fresh
    targeted read-back with one bounded retry. Conflicting label derivations
    are reported and left unfilled -- ambiguity is for maintainer review, not
    automation (docs/governance/github-project.md: leave uncertain metadata
    blank).

    Args:
        client: The Project access layer bound to the target project.
        item: The project item to populate, with its owning project's node id.
        labels: The item's current repository label names.
        project_number: The project's user-visible number, for log messages.

    Returns:
        A (fields_set, fields_preserved) tally for the closing summary line.

    Raises:
        PopulatePolicyError: A mutation's read-back never confirmed the
            requested value within the bounded retry (exit code 1).
    """

    derived, conflicts = project_label_mapping.derive_field_options(labels)
    # Conflicts are surfaced loudly but do not fail the run: the remaining
    # unambiguous fields still deserve population, and the conflicted field
    # is deliberately left blank for the maintainer.
    for conflict in conflicts:
        print(f"warning: {conflict}; leaving it for maintainer review", file=sys.stderr)

    # Status first (the documented new-item default), then the derivable
    # fields in their canonical order, so logs and tests see one
    # deterministic sequence.
    planned: list[tuple[str, str]] = [("Status", _DEFAULT_STATUS_OPTION)]
    planned.extend(
        (field_name, derived[field_name])
        for field_name in project_label_mapping.DERIVABLE_FIELDS
        if field_name in derived
    )

    fields_set = 0
    fields_preserved = 0
    # Apply the plan sequentially; each field is independently preserved-or-set
    # with its own verification read (never a back-to-back unverified batch).
    for field_name, option_name in planned:
        # The shared primitive raises its own unverified-mutation error class;
        # re-raise it as this script's policy error so main() maps it to the
        # house policy exit code (1) rather than the unreadable-data code (2).
        try:
            mutated = client.set_single_select_if_unset(item, field_name, option_name)
        except github_api.FieldMutationUnverifiedError as error:
            raise PopulatePolicyError(str(error)) from error
        # Tally each outcome for the closing summary line; the primitive has
        # already printed the per-field set/preserved detail.
        if mutated:
            fields_set += 1
        else:
            fields_preserved += 1
    # The project_number parameter exists purely for this closing context
    # line, keeping per-run logs self-describing when read in CI output.
    print(
        f"Project #{project_number}: {fields_set} field(s) set, "
        f"{fields_preserved} preserved, {len(conflicts)} conflict(s) left unfilled."
    )
    return fields_set, fields_preserved


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the board-population entry point.

    Args:
        argv: Optional command-line arguments; defaults to the process arguments.

    Returns:
        Parsed arguments: content type, item number, optional repo slug,
        project owner, project number, and the minimum-remaining GraphQL
        quota threshold.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--content-type",
        required=True,
        choices=("issue", "pull-request"),
        help="Whether the item is an issue or a pull request",
    )
    parser.add_argument("--number", type=int, required=True, help="Issue or PR number")
    parser.add_argument(
        "--repo",
        default=None,
        help=(
            "GitHub OWNER/REPO of the item; omitted, gh resolves it from the "
            "surrounding Git checkout"
        ),
    )
    parser.add_argument("--owner", default="Jared-Godar", help="Project owner login")
    parser.add_argument("--project-number", type=int, default=5)
    parser.add_argument(
        "--min-graphql-quota",
        type=int,
        default=_MIN_GRAPHQL_QUOTA_DEFAULT,
        help=(
            "Minimum remaining GraphQL points required by the preflight check before "
            "any mutation is attempted; 0 or below disables the stop (the "
            "before/after report is still printed). Default: "
            f"{_MIN_GRAPHQL_QUOTA_DEFAULT}."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line entry point and return its process exit status.

    Keeping orchestration here makes terminal behavior and error translation
    straightforward to audit: preflight, REST content read with the
    governed-bot skip, board membership, then the Status default and the
    label-derived field fills, with quota accounting reported either way.

    Args:
        argv: Optional command-line arguments; defaults to the process arguments.

    Returns:
        The house-convention exit code: 0 success (including the governed-bot
        skip), 1 policy/validation failure, 2 required data unreadable, 3 a
        GraphQL quota condition.
    """

    args = parse_args(argv)

    # One monitor per run: preflight before any GraphQL work, report after --
    # both reads are REST and free, so accounting can never worsen the quota.
    monitor = github_api.QuotaMonitor(minimum_remaining=args.min_graphql_quota)
    # One client per run = one logical phase: schema lookups are cached on it,
    # and no code path in this script ever requests a full board snapshot.
    client = github_api.ProjectClient(args.owner, args.project_number)

    # Every failure mode is collapsed into the GitHubApiError hierarchy by
    # the access layer; the handlers below map it onto the house exit-code
    # convention, most specific first, so shared-pool exhaustion, a policy
    # refusal, and unreadable data can never be conflated in CI output.
    try:
        # Stop before any mutation when the shared pool is already too low --
        # a half-populated item on a drained pool is exactly the interrupted
        # back-fill failure mode issue #233 exists to end; the run is
        # idempotent, so a stopped run is simply rerun after the reset.
        monitor.preflight()
        repo = resolve_repository(args.repo)
        content = fetch_content(repo, args.number, args.content_type)
        # Governed-bot items are the Dependabot autofill path's job; skipping
        # cleanly (exit 0) keeps the workflow green on bot events that slip
        # past the workflow-level condition (e.g. a manual invocation).
        if is_governed_bot_author(content.author_login, content.author_type):
            print(
                f"{args.content_type} #{args.number} is authored by governed bot "
                f"{content.author_login!r}; the Dependabot autofill path owns its "
                "board metadata -- nothing to do"
            )
            return 0
        item = client.ensure_item(repo, args.number, content_kind=args.content_type)
        populate_item(client, item, content.labels, args.project_number)
    except (
        github_api.GraphQLQuotaInsufficientError,
        github_api.PrimaryRateLimitError,
    ) as error:
        # Quota conditions get their own exit code (3) and wording: transient
        # shared-pool infrastructure, resumable by rerunning after the reset,
        # never a defect in the item being populated.
        print(f"quota: {error}", file=sys.stderr)
        return 3
    except PopulatePolicyError as error:
        # Policy/validation failures (an unconfirmed mutation) are exit 1 per
        # the house convention.
        print(f"error: {error}", file=sys.stderr)
        return 1
    except github_api.GitHubApiError as error:
        # Everything else in the hierarchy is required data being unreadable
        # (gh/API failures, a content-kind mismatch, an unverifiable
        # item-add): exit 2.
        print(f"error: {error}", file=sys.stderr)
        return 2
    finally:
        # The consumption report prints on success and failure alike -- a
        # failed run's consumption is exactly the evidence needed when
        # diagnosing a drained pool. A run that failed before preflight ever
        # recorded a baseline has nothing to report; otherwise reporting is
        # best-effort, because a report failure must never mask the run's
        # real outcome.
        if monitor.preflighted:
            # Best-effort only: see the comment above for why a report
            # failure is downgraded to a warning instead of changing the
            # exit code.
            try:
                print(monitor.report(), file=sys.stderr)
            except github_api.GitHubApiError as report_error:
                print(
                    f"warning: could not report quota consumption: {report_error}",
                    file=sys.stderr,
                )

    print(
        f"Populated Project #{args.project_number} metadata for {args.content_type} #{args.number}."
    )
    return 0


# Standard script entry-point guard: only run main() when executed directly, not when
# imported (e.g. by this script's own test module).
if __name__ == "__main__":
    raise SystemExit(main())
