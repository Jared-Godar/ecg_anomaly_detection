#!/usr/bin/env python3
"""Detect labels applied to issues/PRs that are not part of the canonical
.github/labels.json manifest (for example, retired or pre-taxonomy label names).

This is read-only: it never modifies labels, issues, or pull requests, and never
guesses a corrected label. Remediation is a separate, human-directed decision --
see docs/governance/label-taxonomy.md.

Quota stewardship (issue #175, extending #173): the `gh issue list` / `gh pr
list` reads this script performs are GraphQL-backed, so every run draws on the
shared 5000-points/hour pool. The run therefore preflights that pool and prints
a before/after/consumed report, both via the shared access layer
(`scripts/github/github_api.py`). The preflight threshold defaults to 0 --
observe-only -- because this is low-frequency manual/scheduled hygiene whose
spend is small; a manual run must never be blocked by a merely busy pool
(issue #175's explicit non-goal).

Exit codes: 0 no drift, 1 drift detected, 2 a genuine failure (manifest,
authentication, gh CLI), 3 a GraphQL quota condition -- transient shared-pool
infrastructure, never label drift itself.
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
# The same committed label manifest sync_github_labels.py syncs to GitHub; this
# script cross-checks live issue/PR labels against exactly that source of truth.
DEFAULT_MANIFEST = ROOT / ".github" / "labels.json"

# The shared GitHub access layer lives in scripts/github/ (operational tooling,
# not an installed package), one directory below this script, so it is imported
# by putting that directory on sys.path first -- the same file-system-adjacency
# convention the scripts/github/ governance scripts themselves use. The guard
# keeps the insertion idempotent when several of these scripts are loaded into
# one process (e.g. by the test suite).
_GITHUB_SCRIPTS_DIR = str(ROOT / "scripts" / "github")
# Only insert when absent, so repeated loads never stack duplicate entries.
if _GITHUB_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _GITHUB_SCRIPTS_DIR)

import github_api  # noqa: E402  (needs the sys.path insertion above)

# This script's own preflight threshold: observe-only by default (0 disables
# the stop but keeps the before/after report). Deliberately not the shared
# github_api.DEFAULT_MINIMUM_GRAPHQL_QUOTA: this is low-frequency manual or
# scheduled hygiene whose two bounded listings cost a handful of points, and
# issue #175 explicitly rules out defaults that would block a manual hygiene
# run on a merely busy pool.
_MIN_GRAPHQL_QUOTA_DEFAULT: int = 0


class LabelDriftError(github_api.GitHubApiError):
    """Raised when the label manifest or GitHub data cannot be read.

    Subclasses the shared GitHubApiError so main()'s single except clause
    catches script-level defects and access-layer failures uniformly.
    """


@dataclass(frozen=True, slots=True)
class DriftedItem:
    """One issue or pull request carrying at least one non-canonical label."""

    number: int
    kind: str
    title: str
    drifted_labels: tuple[str, ...]


def load_canonical_label_names(manifest_path: Path) -> frozenset[str]:
    """Load the set of canonical label names from the repository's label manifest."""
    data: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    # schema_version pins this loader's understanding of the manifest's shape.
    if data.get("schema_version") != 1 or not isinstance(data.get("labels"), list):
        raise LabelDriftError("manifest must contain schema_version 1 and a labels array")
    names: set[str] = set()
    # Collect every declared label name; only the name matters here (unlike
    # sync_github_labels.py, this script doesn't need color/description).
    for item in data["labels"]:
        name = item.get("name") if isinstance(item, dict) else None
        # A missing/empty name means the manifest itself is malformed.
        if not isinstance(name, str) or not name:
            raise LabelDriftError("each manifest label requires a non-empty name")
        names.add(name)
    return frozenset(names)


def find_drifted_labels(labels: Sequence[str], canonical: frozenset[str]) -> tuple[str, ...]:
    """Return the labels in `labels` that are not present in the canonical set, in order."""
    return tuple(label for label in labels if label not in canonical)


def find_drifted_items(
    items: Sequence[dict[str, Any]], canonical: frozenset[str]
) -> tuple[DriftedItem, ...]:
    """Return a DriftedItem for every item that carries at least one non-canonical label."""
    drifted: list[DriftedItem] = []
    # Check every fetched issue/PR independently, keeping only the ones that actually
    # carry drift, so the caller sees a focused list rather than every item.
    for item in items:
        labels = find_drifted_labels(item["labels"], canonical)
        # An empty drift list means this item's labels are all canonical.
        if labels:
            drifted.append(DriftedItem(item["number"], item["kind"], item["title"], labels))
    return tuple(drifted)


def fetch_items(repo: str | None, *, include_closed: bool) -> list[dict[str, Any]]:
    """Fetch issues and pull requests with their labels via the gh CLI.

    The subprocess plumbing (retry classification, error translation) lives in
    the shared access layer's run_gh (issue #175 removed this script's private
    copy); failures surface as github_api.GitHubApiError for main() to map.
    """
    state = "all" if include_closed else "open"
    items: list[dict[str, Any]] = []
    # gh uses separate subcommands for issues and pull requests; query both so drift
    # detection covers the whole repository, not just one kind of item.
    for kind, subcommand in (("issue", "issue"), ("pull request", "pr")):
        args = [
            subcommand,
            "list",
            "--state",
            state,
            "--json",
            "number,title,labels",
            "--limit",
            "500",
        ]
        # repo is optional; omitting --repo lets gh infer it from the current
        # directory's Git remote, matching gh's own default behavior.
        if repo:
            args.extend(["--repo", repo])
        payload = json.loads(github_api.run_gh(args))
        # Flatten gh's nested label objects into plain name strings, and tag each row
        # with its kind, before appending to the combined items list.
        for row in payload:
            items.append(
                {
                    "kind": kind,
                    "number": row["number"],
                    "title": row["title"],
                    "labels": [label["name"] for label in row.get("labels", [])],
                }
            )
    return items


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the label-drift detection entry point.

    Args:
        argv: Optional command-line arguments; defaults to the process arguments.

    Returns:
        Parsed arguments: manifest path, optional target repo, include-closed
        flag, and the minimum-remaining GraphQL quota threshold.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repo", help="GitHub OWNER/REPO; defaults to the current repository")
    parser.add_argument(
        "--include-closed",
        action="store_true",
        help="Also check closed issues and merged or closed pull requests",
    )
    parser.add_argument(
        "--min-graphql-quota",
        type=int,
        default=_MIN_GRAPHQL_QUOTA_DEFAULT,
        help=(
            "Minimum remaining GraphQL points required by the preflight check before "
            "any listing is fetched; 0 or below disables the stop (the "
            "before/after report is still printed). Default: "
            f"{_MIN_GRAPHQL_QUOTA_DEFAULT} (observe-only, so manual hygiene "
            "runs are never blocked)."
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

    # One monitor per run: preflight before the GraphQL-backed listings, report
    # after -- both reads are REST and free, so accounting can never worsen the
    # quota. The default threshold of 0 observes without ever blocking.
    monitor = github_api.QuotaMonitor(minimum_remaining=args.min_graphql_quota)

    # Every failure mode this script can hit (manifest load, gh CLI, quota) is
    # part of the GitHubApiError hierarchy -- LabelDriftError subclasses it --
    # so it is caught once here for uniform error reporting, with the two quota
    # conditions mapped to their own exit code so hygiene output can never
    # conflate a drained shared pool with label drift or a broken manifest.
    try:
        # The manifest is local and free; load it before spending any gh call
        # so a malformed manifest never costs even the free quota reads.
        canonical = load_canonical_label_names(args.manifest)
        monitor.preflight()
        items = fetch_items(args.repo, include_closed=args.include_closed)
    except (
        github_api.GraphQLQuotaInsufficientError,
        github_api.PrimaryRateLimitError,
    ) as error:
        # Quota conditions get their own exit code (3) and wording: transient
        # shared-pool infrastructure, resumable by rerunning after the reset,
        # never evidence about the repository's labels.
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

    drifted = find_drifted_items(items, canonical)
    # A non-empty drift list means at least one item carries a non-canonical label;
    # report every one explicitly (never auto-fix) and exit non-zero so this can gate CI.
    if drifted:
        print(f"Label drift detected on {len(drifted)} item(s):", file=sys.stderr)
        # List every drifted item's own drifted labels, so a reviewer can act on the
        # complete picture without re-running with different flags.
        for item in drifted:
            print(
                f"  - #{item.number} ({item.kind}) {item.title!r}: "
                + ", ".join(item.drifted_labels),
                file=sys.stderr,
            )
        print(
            "\nSee docs/governance/label-taxonomy.md. Remediation is a separate, "
            "human-directed decision -- this check does not relabel anything.",
            file=sys.stderr,
        )
        return 1

    print("No label drift detected.")
    return 0


# Standard script entry-point guard: only run main() when executed directly, not when
# imported (e.g. by this script's own test module).
if __name__ == "__main__":
    raise SystemExit(main())
