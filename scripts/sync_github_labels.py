#!/usr/bin/env python3
"""Create or update the repository labels declared in .github/labels.json."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the label-sync entry point.

    Returns:
        Parsed arguments: manifest path, optional target repo, and dry-run flag.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repo", help="GitHub OWNER/REPO; defaults to the current repository")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without changing labels"
    )
    return parser.parse_args()


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


def main() -> int:
    """Run the command-line entry point and return its process exit status.

    Keeping orchestration here makes terminal behavior and error translation straightforward
    to audit.

    Returns:
        The value produced by the documented operation.
    """

    args = parse_args()
    labels = load_labels(args.manifest)
    repo_args = ["--repo", args.repo] if args.repo else []

    # Create or update (via --force) every manifest label in turn; `gh label create
    # --force` is idempotent, so re-running this script is always safe.
    for label in labels:
        command = [
            "gh",
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
        # --dry-run prints the command that would run instead of executing it, so a
        # reviewer can confirm intended changes before actually syncing labels.
        if args.dry_run:
            print(json.dumps(command))
        else:
            # command is a fixed literal list built above, not runtime/user-constructed input.
            subprocess.run(command, cwd=ROOT, check=True)  # noqa: S603
    return 0


# Standard script entry-point guard: only run main() when executed directly, not when
# imported (e.g. by this script's own test module).
if __name__ == "__main__":
    raise SystemExit(main())
