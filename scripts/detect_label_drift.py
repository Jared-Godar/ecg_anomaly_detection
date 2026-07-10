#!/usr/bin/env python3
"""Detect labels applied to issues/PRs that are not part of the canonical
.github/labels.json manifest (for example, retired or pre-taxonomy label names).

This is read-only: it never modifies labels, issues, or pull requests, and never
guesses a corrected label. Remediation is a separate, human-directed decision --
see docs/governance/label-taxonomy.md.
"""

from __future__ import annotations

import argparse
import json
import subprocess
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


class LabelDriftError(RuntimeError):
    """Raised when the label manifest or GitHub data cannot be read."""


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


def _run_gh(args: list[str]) -> str:
    """Run one fixed GitHub CLI command and return its captured output.

    Args:
        args: The `gh` subcommand and its arguments (without the leading "gh" itself).

    Returns:
        The command's captured stdout.
    """

    # Collapse "gh not installed" (FileNotFoundError) and "gh exited non-zero"
    # (CalledProcessError, since check=True) into one LabelDriftError, so main()'s
    # error handling only needs to catch this module's own exception type.
    try:
        # command is a fixed literal ("gh", *args) built from this module's own
        # subcommand arguments, not runtime/user-constructed input.
        result = subprocess.run(  # noqa: S603
            ["gh", *args],  # noqa: S607
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise LabelDriftError("gh CLI is not installed or not on PATH") from error
    except subprocess.CalledProcessError as error:
        raise LabelDriftError(
            f"gh {' '.join(args)} failed: {error.stderr.strip() or error.stdout.strip()}"
        ) from error
    return result.stdout


def fetch_items(repo: str | None, *, include_closed: bool) -> list[dict[str, Any]]:
    """Fetch issues and pull requests with their labels via the gh CLI."""
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
        payload = json.loads(_run_gh(args))
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
        Parsed arguments: manifest path, optional target repo, and include-closed flag.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repo", help="GitHub OWNER/REPO; defaults to the current repository")
    parser.add_argument(
        "--include-closed",
        action="store_true",
        help="Also check closed issues and merged or closed pull requests",
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
    # Every failure mode this script can hit (manifest load, gh CLI) is collapsed
    # into LabelDriftError by the functions above; catch it once here for uniform
    # error reporting rather than duplicating handling at each call site.
    try:
        canonical = load_canonical_label_names(args.manifest)
        items = fetch_items(args.repo, include_closed=args.include_closed)
    except LabelDriftError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

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
