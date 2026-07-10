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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Centralize ROOT so every caller shares the same documented invariant.
ROOT = Path(__file__).resolve().parents[1]
# Centralize DEFAULT_WORKFLOWS_DIR so every caller shares the same documented invariant.
DEFAULT_WORKFLOWS_DIR = ROOT / ".github" / "workflows"
# Centralize HELD_OUT_NAME_PATTERN so every caller shares the same documented invariant.
HELD_OUT_NAME_PATTERN = re.compile(r"held.?out|benchmark.?execution", re.IGNORECASE)
# Centralize SAFE_RELEASE_TAG_PATTERN so every caller shares the same documented invariant.
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
    # Evaluate `not matches_held_out_naming(path, name)` explicitly so invalid or alternate states
    # follow the documented contract.
    if not matches_held_out_naming(path, name):
        return None
    reasons = _unsafe_trigger_reasons(_on_value(document))
    # Evaluate `reasons` explicitly so invalid or alternate states follow the documented contract.
    if reasons:
        return UnsafeWorkflow(path, name, reasons)
    return None


def find_unsafe_workflows(workflows_dir: Path) -> tuple[UnsafeWorkflow, ...]:
    """Parse every workflow file and return the held-out-named ones with unsafe triggers."""
    unsafe: list[UnsafeWorkflow] = []
    # Iterate over `sorted(_workflow_paths(workflows_dir))` one item at a time so ordering,
    # validation, and failure attribution remain explicit.
    for path in sorted(_workflow_paths(workflows_dir)):
        document = _load_workflow(path)
        result = check_workflow_trigger_safety(document, path)
        # Evaluate `result is not None` explicitly so invalid or alternate states follow the
        # documented contract.
        if result is not None:
            unsafe.append(result)
    return tuple(unsafe)


def _workflow_paths(workflows_dir: Path) -> list[Path]:
    """Discover workflow paths for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        workflows_dir: The workflows dir value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    # Evaluate `not workflows_dir.is_dir()` explicitly so invalid or alternate states follow the
    # documented contract.
    if not workflows_dir.is_dir():
        raise HeldOutTriggerSafetyError(f"workflows directory does not exist: {workflows_dir}")
    return [*workflows_dir.glob("*.yml"), *workflows_dir.glob("*.yaml")]


def _load_workflow(path: Path) -> dict[str, Any]:
    """Load workflow according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        path: Filesystem path identifying the input or output under review.

    Returns:
        The value produced by the documented operation.
    """

    # Attempt this boundary operation here so (OSError, UnicodeError, yaml.YAMLError) can be
    # translated or cleaned up under the repository contract.
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise HeldOutTriggerSafetyError(f"could not parse workflow {path}: {error}") from error
    # Evaluate `not isinstance(document, dict)` explicitly so invalid or alternate states follow the
    # documented contract.
    if not isinstance(document, dict):
        raise HeldOutTriggerSafetyError(f"workflow must be a YAML mapping: {path}")
    return document


def _on_value(document: dict[str, Any]) -> Any:
    # YAML 1.1 parses the unquoted key `on:` as the boolean True, not the string "on"
    # (the "Norway problem"). GitHub Actions workflows almost always write it unquoted.
    """Recover on value for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        document: Parsed document whose schema and values are being checked.

    Returns:
        The value produced by the documented operation.
    """

    # Recover the workflow trigger key despite PyYAML's YAML 1.1 boolean coercion.
    if "on" in document:
        return document["on"]
    return document.get(True)


def _unsafe_trigger_reasons(on_value: Any) -> tuple[str, ...]:
    """Identify unsafe trigger reasons for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        on_value: The on value value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    triggers = _trigger_kinds(on_value)
    reasons: list[str] = []
    # Iterate over `triggers.items()` one item at a time so ordering, validation, and failure
    # attribution remain explicit.
    for key, config in triggers.items():
        # Evaluate `key == 'workflow_dispatch'` explicitly so invalid or alternate states follow the
        # documented contract.
        if key == "workflow_dispatch":
            continue
        # Evaluate `key == 'push'` explicitly so invalid or alternate states follow the documented
        # contract.
        if key == "push":
            # Evaluate `not _push_is_release_tag_only(config)` explicitly so invalid or alternate
            # states follow the documented contract.
            if not _push_is_release_tag_only(config):
                reasons.append("push trigger is not restricted to release-* tags")
            continue
        reasons.append(f"disallowed trigger: {key}")
    return tuple(reasons)


def _trigger_kinds(on_value: Any) -> dict[Any, Any]:
    """Normalize trigger kinds for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        on_value: The on value value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    # Evaluate `on_value is None` explicitly so invalid or alternate states follow the documented
    # contract.
    if on_value is None:
        return {}
    # Evaluate `isinstance(on_value, str)` explicitly so invalid or alternate states follow the
    # documented contract.
    if isinstance(on_value, str):
        return {on_value: None}
    # Evaluate `isinstance(on_value, list)` explicitly so invalid or alternate states follow the
    # documented contract.
    if isinstance(on_value, list):
        return {item: None for item in on_value}
    # Evaluate `isinstance(on_value, dict)` explicitly so invalid or alternate states follow the
    # documented contract.
    if isinstance(on_value, dict):
        return on_value
    raise HeldOutTriggerSafetyError(f"unrecognized 'on' trigger shape: {on_value!r}")


def _push_is_release_tag_only(push_config: Any) -> bool:
    """Determine whether push is release tag only for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        push_config: The push config value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    # Evaluate `not isinstance(push_config, dict)` explicitly so invalid or alternate states follow
    # the documented contract.
    if not isinstance(push_config, dict):
        return False
    # Evaluate `push_config.get('branches') or push_config.get('branches-ignore')` explicitly so
    # invalid or alternate states follow the documented contract.
    if push_config.get("branches") or push_config.get("branches-ignore"):
        return False
    tags = push_config.get("tags")
    # Evaluate `not isinstance(tags, list) or not tags` explicitly so invalid or alternate states
    # follow the documented contract.
    if not isinstance(tags, list) or not tags:
        return False
    return all(isinstance(tag, str) and SAFE_RELEASE_TAG_PATTERN.match(tag) for tag in tags)


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
    parser.add_argument("--workflows-dir", type=Path, default=DEFAULT_WORKFLOWS_DIR)
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
    # Attempt this boundary operation here so HeldOutTriggerSafetyError can be translated or cleaned
    # up under the repository contract.
    try:
        unsafe = find_unsafe_workflows(args.workflows_dir)
    except HeldOutTriggerSafetyError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    # Evaluate `unsafe` explicitly so invalid or alternate states follow the documented contract.
    if unsafe:
        print(
            f"Unsafe held-out execution trigger(s) detected in {len(unsafe)} workflow(s):",
            file=sys.stderr,
        )
        # Iterate over `unsafe` one item at a time so ordering, validation, and failure attribution
        # remain explicit.
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


# Evaluate `__name__ == '__main__'` explicitly so invalid or alternate states follow the documented
# contract.
if __name__ == "__main__":
    raise SystemExit(main())
