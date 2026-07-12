#!/usr/bin/env python3
"""Create or update the repository labels declared in .github/labels.json.

Quota stewardship (issue #175, extending #173): the `gh label create --force`
mutations this script performs are REST, so a sync spends no GraphQL points of
its own -- but the run still preflights the shared 5000-points/hour GraphQL
pool and prints a before/after/consumed report via the shared access layer
(`scripts/github/github_api.py`), so every gh-calling script in the repository
is accountable in its own logs under one convention. The preflight threshold
defaults to 0 -- observe-only -- because this is low-frequency manual hygiene;
a manual run must never be blocked by a merely busy pool (issue #175's
explicit non-goal). `--dry-run` performs no gh calls at all, including the
free quota reads.

Exit codes: 0 success, 2 a gh CLI/API failure (a malformed manifest still
raises ValueError directly, unchanged from before the migration), 3 a GraphQL
quota condition -- transient shared-pool infrastructure, never a label defect.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Repository root, resolved from this script's own location rather than the current
# working directory, so the script behaves identically regardless of where it's invoked from.
ROOT = Path(__file__).resolve().parents[1]
# The committed, version-controlled source of truth for repository labels; every
# label GitHub should have is declared here, not created ad hoc via the GitHub UI.
DEFAULT_MANIFEST = ROOT / ".github" / "labels.json"
# GitHub label colors are exactly six hex digits (no leading '#'); this pattern
# validates the manifest's color field matches that format before it's sent to `gh`.
COLOR_PATTERN = re.compile(r"[0-9a-fA-F]{6}")

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
# github_api.DEFAULT_MINIMUM_GRAPHQL_QUOTA: the sync's mutations are REST and
# spend no GraphQL points, and issue #175 explicitly rules out defaults that
# would block a manual hygiene run on a merely busy pool.
_MIN_GRAPHQL_QUOTA_DEFAULT: int = 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the label-sync entry point.

    Args:
        argv: Optional command-line arguments; defaults to the process arguments.

    Returns:
        Parsed arguments: manifest path, optional target repo, dry-run flag,
        and the minimum-remaining GraphQL quota threshold.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repo", help="GitHub OWNER/REPO; defaults to the current repository")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without changing labels"
    )
    parser.add_argument(
        "--min-graphql-quota",
        type=int,
        default=_MIN_GRAPHQL_QUOTA_DEFAULT,
        help=(
            "Minimum remaining GraphQL points required by the preflight check before "
            "any label mutation is attempted; 0 or below disables the stop (the "
            "before/after report is still printed). Default: "
            f"{_MIN_GRAPHQL_QUOTA_DEFAULT} (observe-only, so manual hygiene "
            "runs are never blocked)."
        ),
    )
    return parser.parse_args(argv)


def load_labels(path: Path) -> list[dict[str, str]]:
    """Load and validate the label manifest, normalizing colors to lowercase.

    Args:
        path: Path to the labels.json manifest file.

    Returns:
        Validated label entries, each with name, color, and description.
    """

    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    # schema_version pins this loader's understanding of the manifest's shape.
    if data.get("schema_version") != 1 or not isinstance(data.get("labels"), list):
        raise ValueError("manifest must contain schema_version 1 and a labels array")

    labels: list[dict[str, str]] = []
    names: set[str] = set()
    # Validate every label entry independently so a malformed entry anywhere in the
    # manifest is caught, not just the first.
    for item in data["labels"]:
        # Every field access below assumes item is a dict.
        if not isinstance(item, dict):
            raise ValueError("each label must be an object")
        name, color, description = (item.get(key) for key in ("name", "color", "description"))
        # All three fields are required; `gh label create` would otherwise fail with a
        # less specific error, or silently accept an empty string.
        if not all(isinstance(value, str) and value for value in (name, color, description)):
            raise ValueError("each label requires non-empty name, color, and description strings")
        # A duplicated name would create two manifest entries competing to define the
        # same GitHub label, with only one actually taking effect.
        if name in names:
            raise ValueError(f"duplicate label: {name}")
        # gh label create requires a bare six-digit hex color, no leading '#'.
        if COLOR_PATTERN.fullmatch(color) is None:
            raise ValueError(f"invalid six-digit color for {name}: {color}")
        names.add(name)
        labels.append({"name": name, "color": color.lower(), "description": description})
    return labels


def _label_create_args(label: dict[str, str], repo_args: list[str]) -> list[str]:
    """Build one label's `gh label create --force` argument list (without "gh").

    Args:
        label: One validated manifest entry (name, color, description).
        repo_args: Either ["--repo", OWNER/REPO] or [] to let gh infer the repo.

    Returns:
        The gh subcommand arguments for creating or updating this label.
    """

    return [
        "label",
        "create",
        label["name"],
        "--color",
        label["color"],
        "--description",
        label["description"],
        "--force",
        *repo_args,
    ]


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
    labels = load_labels(args.manifest)
    repo_args = ["--repo", args.repo] if args.repo else []

    # --dry-run prints the command that would run for every label instead of
    # executing anything, so a reviewer can confirm intended changes before
    # actually syncing labels. It returns before the quota monitor exists
    # because a dry run performs no gh calls at all -- not even the free
    # rate_limit reads -- exactly as it did before the migration.
    if args.dry_run:
        # Print one command per manifest label, in manifest order, with the
        # leading "gh" kept so each line is byte-for-byte what a reviewer
        # could paste into a shell.
        for label in labels:
            print(json.dumps(["gh", *_label_create_args(label, repo_args)]))
        return 0

    # One monitor per run: preflight before the first mutation, report after --
    # both reads are REST and free, so accounting can never worsen the quota.
    # The default threshold of 0 observes without ever blocking.
    monitor = github_api.QuotaMonitor(minimum_remaining=args.min_graphql_quota)

    # Every gh failure mode is collapsed into the GitHubApiError hierarchy by
    # the access layer; catch it once here for uniform error reporting, with
    # the two quota conditions mapped to their own exit code so hygiene output
    # can never conflate a drained shared pool with a label defect.
    try:
        monitor.preflight()
        # Create or update (via --force) every manifest label in turn; `gh label
        # create --force` is idempotent, so re-running this script is always safe.
        for label in labels:
            # cwd=ROOT pins gh's repository inference to this checkout's root
            # when --repo is omitted, preserving the pre-migration
            # subprocess.run(cwd=ROOT) behavior exactly.
            output = github_api.run_gh(_label_create_args(label, repo_args), cwd=ROOT)
            # Before the migration gh wrote directly to the terminal; run_gh
            # captures stdout instead, so any confirmation text gh produced is
            # forwarded verbatim (gh emits nothing on success when captured,
            # so this is usually silent -- matching a piped pre-migration run).
            if output:
                print(output, end="")
    except (
        github_api.GraphQLQuotaInsufficientError,
        github_api.PrimaryRateLimitError,
    ) as error:
        # Quota conditions get their own exit code (3) and wording: transient
        # shared-pool infrastructure, resumable by rerunning after the reset
        # (the sync is idempotent), never a defect in the label manifest.
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
        # real outcome.
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
    return 0


# Standard script entry-point guard: only run main() when executed directly, not when
# imported (e.g. by this script's own test module).
if __name__ == "__main__":
    raise SystemExit(main())
