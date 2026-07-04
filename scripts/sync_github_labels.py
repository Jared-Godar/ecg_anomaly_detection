#!/usr/bin/env python3
"""Create or update the repository labels declared in .github/labels.json."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / ".github" / "labels.json"
COLOR_PATTERN = re.compile(r"[0-9a-fA-F]{6}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repo", help="GitHub OWNER/REPO; defaults to the current repository")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without changing labels"
    )
    return parser.parse_args()


def load_labels(path: Path) -> list[dict[str, str]]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("labels"), list):
        raise ValueError("manifest must contain schema_version 1 and a labels array")

    labels: list[dict[str, str]] = []
    names: set[str] = set()
    for item in data["labels"]:
        if not isinstance(item, dict):
            raise ValueError("each label must be an object")
        name, color, description = (item.get(key) for key in ("name", "color", "description"))
        if not all(isinstance(value, str) and value for value in (name, color, description)):
            raise ValueError("each label requires non-empty name, color, and description strings")
        if name in names:
            raise ValueError(f"duplicate label: {name}")
        if COLOR_PATTERN.fullmatch(color) is None:
            raise ValueError(f"invalid six-digit color for {name}: {color}")
        names.add(name)
        labels.append({"name": name, "color": color.lower(), "description": description})
    return labels


def main() -> int:
    args = parse_args()
    labels = load_labels(args.manifest)
    repo_args = ["--repo", args.repo] if args.repo else []

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
        if args.dry_run:
            print(json.dumps(command))
        else:
            subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
