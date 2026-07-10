#!/usr/bin/env python3
"""Create or update the repository labels declared in .github/labels.json."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

# Centralize ROOT so every caller shares the same documented invariant.
ROOT = Path(__file__).resolve().parents[1]
# Centralize DEFAULT_MANIFEST so every caller shares the same documented invariant.
DEFAULT_MANIFEST = ROOT / ".github" / "labels.json"
# Centralize COLOR_PATTERN so every caller shares the same documented invariant.
COLOR_PATTERN = re.compile(r"[0-9a-fA-F]{6}")


def parse_args() -> argparse.Namespace:
    """Parse args according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Returns:
        The value produced by the documented operation.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repo", help="GitHub OWNER/REPO; defaults to the current repository")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without changing labels"
    )
    return parser.parse_args()


def load_labels(path: Path) -> list[dict[str, str]]:
    """Load labels according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        path: Filesystem path identifying the input or output under review.

    Returns:
        The value produced by the documented operation.
    """

    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    # Evaluate `data.get('schema_version') != 1 or not isinstance(data.get('labels'), list)`
    # explicitly so invalid or alternate states follow the documented contract.
    if data.get("schema_version") != 1 or not isinstance(data.get("labels"), list):
        raise ValueError("manifest must contain schema_version 1 and a labels array")

    labels: list[dict[str, str]] = []
    names: set[str] = set()
    # Iterate over `data['labels']` one item at a time so ordering, validation, and failure
    # attribution remain explicit.
    for item in data["labels"]:
        # Evaluate `not isinstance(item, dict)` explicitly so invalid or alternate states follow the
        # documented contract.
        if not isinstance(item, dict):
            raise ValueError("each label must be an object")
        name, color, description = (item.get(key) for key in ("name", "color", "description"))
        # Evaluate `not all((isinstance(value, str) and value for value in (name, color,
        # description)))` explicitly so invalid or alternate states follow the documented contract.
        if not all(isinstance(value, str) and value for value in (name, color, description)):
            raise ValueError("each label requires non-empty name, color, and description strings")
        # Evaluate `name in names` explicitly so invalid or alternate states follow the documented
        # contract.
        if name in names:
            raise ValueError(f"duplicate label: {name}")
        # Evaluate `COLOR_PATTERN.fullmatch(color) is None` explicitly so invalid or alternate
        # states follow the documented contract.
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

    # Iterate over `labels` one item at a time so ordering, validation, and failure attribution
    # remain explicit.
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
        # Evaluate `args.dry_run` explicitly so invalid or alternate states follow the documented
        # contract.
        if args.dry_run:
            print(json.dumps(command))
        else:
            # command is a fixed literal list built above, not runtime/user-constructed input.
            subprocess.run(command, cwd=ROOT, check=True)  # noqa: S603
    return 0


# Evaluate `__name__ == '__main__'` explicitly so invalid or alternate states follow the documented
# contract.
if __name__ == "__main__":
    raise SystemExit(main())
