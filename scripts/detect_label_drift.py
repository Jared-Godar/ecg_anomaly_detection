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

# Centralize ROOT so every caller shares the same documented invariant.
ROOT = Path(__file__).resolve().parents[1]
# Centralize DEFAULT_MANIFEST so every caller shares the same documented invariant.
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
    # Evaluate `data.get('schema_version') != 1 or not isinstance(data.get('labels'), list)`
    # explicitly so invalid or alternate states follow the documented contract.
    if data.get("schema_version") != 1 or not isinstance(data.get("labels"), list):
        raise LabelDriftError("manifest must contain schema_version 1 and a labels array")
    names: set[str] = set()
    # Iterate over `data['labels']` one item at a time so ordering, validation, and failure
    # attribution remain explicit.
    for item in data["labels"]:
        name = item.get("name") if isinstance(item, dict) else None
        # Evaluate `not isinstance(name, str) or not name` explicitly so invalid or alternate states
        # follow the documented contract.
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
    # Iterate over `items` one item at a time so ordering, validation, and failure attribution
    # remain explicit.
    for item in items:
        labels = find_drifted_labels(item["labels"], canonical)
        # Evaluate `labels` explicitly so invalid or alternate states follow the documented
        # contract.
        if labels:
            drifted.append(DriftedItem(item["number"], item["kind"], item["title"], labels))
    return tuple(drifted)


def _run_gh(args: list[str]) -> str:
    """Run one fixed GitHub CLI command and return its captured output.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        args: The args value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    # Attempt this boundary operation here so FileNotFoundError, subprocess.CalledProcessError can
    # be translated or cleaned up under the repository contract.
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
    # Iterate over `(('issue', 'issue'), ('pull request', 'pr'))` one item at a time so ordering,
    # validation, and failure attribution remain explicit.
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
        # Evaluate `repo` explicitly so invalid or alternate states follow the documented contract.
        if repo:
            args.extend(["--repo", repo])
        payload = json.loads(_run_gh(args))
        # Iterate over `payload` one item at a time so ordering, validation, and failure attribution
        # remain explicit.
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
    """Parse args according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        argv: Optional command-line arguments; defaults to the process arguments.

    Returns:
        The value produced by the documented operation.
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
    # Attempt this boundary operation here so LabelDriftError can be translated or cleaned up under
    # the repository contract.
    try:
        canonical = load_canonical_label_names(args.manifest)
        items = fetch_items(args.repo, include_closed=args.include_closed)
    except LabelDriftError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    drifted = find_drifted_items(items, canonical)
    # Evaluate `drifted` explicitly so invalid or alternate states follow the documented contract.
    if drifted:
        print(f"Label drift detected on {len(drifted)} item(s):", file=sys.stderr)
        # Iterate over `drifted` one item at a time so ordering, validation, and failure attribution
        # remain explicit.
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


# Evaluate `__name__ == '__main__'` explicitly so invalid or alternate states follow the documented
# contract.
if __name__ == "__main__":
    raise SystemExit(main())
