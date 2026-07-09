#!/usr/bin/env python3
"""Fail if a held-out/benchmark-execution workflow could trigger on routine push/PR events.

This is a governance-as-code guard, not an execution command: it never runs, dispatches, or
inspects a workflow's job contents, and it never touches the protected `test` partition. It reads
every `.github/workflows/*.yml` file and, for any workflow whose filename or `name:` matches a
held-out/benchmark-execution naming convention, requires that its triggers are limited to
`workflow_dispatch` and/or a `push` restricted to `release-*` tags -- never a bare `push` and never
`pull_request`. This protects against an unsafe trigger being added later, when a future held-out
execution workflow is actually written (see docs/benchmark-governance.md and issue #73).
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKFLOWS_DIR = ROOT / ".github" / "workflows"
HELD_OUT_NAME_PATTERN = re.compile(r"held.?out|benchmark.?execution", re.IGNORECASE)
SAFE_RELEASE_TAG_PATTERN = re.compile(r"^release-")


class HeldOutTriggerSafetyError(RuntimeError):
    """Raised when a workflow file cannot be read or parsed as a GitHub Actions workflow."""


@dataclass(frozen=True, slots=True)
class UnsafeWorkflow:
    """One held-out/benchmark-execution workflow with a trigger that is not safely restricted."""

    path: Path
    name: str
    reasons: tuple[str, ...]


def matches_held_out_naming(path: Path, workflow_name: str) -> bool:
    """Return whether a workflow's filename or declared name looks like a held-out execution."""
    return bool(
        HELD_OUT_NAME_PATTERN.search(path.stem) or HELD_OUT_NAME_PATTERN.search(workflow_name)
    )


def check_workflow_trigger_safety(document: dict[str, Any], path: Path) -> UnsafeWorkflow | None:
    """Return an UnsafeWorkflow if a held-out-named workflow's triggers are not safely restricted."""
    name = document.get("name") if isinstance(document.get("name"), str) else path.stem
    if not matches_held_out_naming(path, name):
        return None
    reasons = _unsafe_trigger_reasons(_on_value(document))
    if reasons:
        return UnsafeWorkflow(path, name, reasons)
    return None


def find_unsafe_workflows(workflows_dir: Path) -> tuple[UnsafeWorkflow, ...]:
    """Parse every workflow file and return the held-out-named ones with unsafe triggers."""
    unsafe: list[UnsafeWorkflow] = []
    for path in sorted(_workflow_paths(workflows_dir)):
        document = _load_workflow(path)
        result = check_workflow_trigger_safety(document, path)
        if result is not None:
            unsafe.append(result)
    return tuple(unsafe)


def _workflow_paths(workflows_dir: Path) -> list[Path]:
    if not workflows_dir.is_dir():
        raise HeldOutTriggerSafetyError(f"workflows directory does not exist: {workflows_dir}")
    return [*workflows_dir.glob("*.yml"), *workflows_dir.glob("*.yaml")]


def _load_workflow(path: Path) -> dict[str, Any]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise HeldOutTriggerSafetyError(f"could not parse workflow {path}: {error}") from error
    if not isinstance(document, dict):
        raise HeldOutTriggerSafetyError(f"workflow must be a YAML mapping: {path}")
    return document


def _on_value(document: dict[str, Any]) -> Any:
    # YAML 1.1 parses the unquoted key `on:` as the boolean True, not the string "on"
    # (the "Norway problem"). GitHub Actions workflows almost always write it unquoted.
    if "on" in document:
        return document["on"]
    return document.get(True)


def _unsafe_trigger_reasons(on_value: Any) -> tuple[str, ...]:
    triggers = _trigger_kinds(on_value)
    reasons: list[str] = []
    for key, config in triggers.items():
        if key == "workflow_dispatch":
            continue
        if key == "push":
            if not _push_is_release_tag_only(config):
                reasons.append("push trigger is not restricted to release-* tags")
            continue
        reasons.append(f"disallowed trigger: {key}")
    return tuple(reasons)


def _trigger_kinds(on_value: Any) -> dict[Any, Any]:
    if on_value is None:
        return {}
    if isinstance(on_value, str):
        return {on_value: None}
    if isinstance(on_value, list):
        return {item: None for item in on_value}
    if isinstance(on_value, dict):
        return on_value
    raise HeldOutTriggerSafetyError(f"unrecognized 'on' trigger shape: {on_value!r}")


def _push_is_release_tag_only(push_config: Any) -> bool:
    if not isinstance(push_config, dict):
        return False
    if push_config.get("branches") or push_config.get("branches-ignore"):
        return False
    tags = push_config.get("tags")
    if not isinstance(tags, list) or not tags:
        return False
    return all(isinstance(tag, str) and SAFE_RELEASE_TAG_PATTERN.match(tag) for tag in tags)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflows-dir", type=Path, default=DEFAULT_WORKFLOWS_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        unsafe = find_unsafe_workflows(args.workflows_dir)
    except HeldOutTriggerSafetyError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if unsafe:
        print(
            f"Unsafe held-out execution trigger(s) detected in {len(unsafe)} workflow(s):",
            file=sys.stderr,
        )
        for item in unsafe:
            print(
                f"  - {item.path.name} ({item.name!r}): " + "; ".join(item.reasons), file=sys.stderr
            )
        print(
            "\nHeld-out/benchmark-execution workflows must trigger only on workflow_dispatch "
            "and/or a push restricted to release-* tags -- never push or pull_request on "
            "routine branches. See docs/benchmark-governance.md.",
            file=sys.stderr,
        )
        return 1

    print("No unsafe held-out execution triggers detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
