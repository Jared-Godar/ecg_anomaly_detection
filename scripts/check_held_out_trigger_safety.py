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

# Repository root, resolved from this script's own location rather than the current
# working directory, so the script behaves identically regardless of where it's invoked from.
ROOT = Path(__file__).resolve().parents[1]
# The standard location GitHub Actions reads workflow files from; overridable via
# --workflows-dir for tests that exercise this script against a fixture directory.
DEFAULT_WORKFLOWS_DIR = ROOT / ".github" / "workflows"
# Matches "held-out"/"held_out"/"heldout" and "benchmark-execution"/"benchmark_execution"/etc.,
# case-insensitively, in either a workflow's filename or its declared `name:` -- this is
# the naming convention this script uses to decide which workflows are even in scope,
# since it never inspects job contents to detect held-out execution semantically.
HELD_OUT_NAME_PATTERN = re.compile(r"held.?out|benchmark.?execution", re.IGNORECASE)
# The one trigger shape a `push:` restriction is allowed to take: tags starting with
# "release-", never a branch push.
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
    # Workflows that don't match the held-out naming convention are entirely out of
    # scope for this check -- their triggers are never inspected.
    if not matches_held_out_naming(path, name):
        return None
    reasons = _unsafe_trigger_reasons(_on_value(document))
    # No unsafe-trigger reasons means this held-out-named workflow's triggers are
    # already correctly restricted; nothing to report.
    if reasons:
        return UnsafeWorkflow(path, name, reasons)
    return None


def find_unsafe_workflows(workflows_dir: Path) -> tuple[UnsafeWorkflow, ...]:
    """Parse every workflow file and return the held-out-named ones with unsafe triggers."""
    unsafe: list[UnsafeWorkflow] = []
    # Process workflow files in sorted order for deterministic output.
    for path in sorted(_workflow_paths(workflows_dir)):
        document = _load_workflow(path)
        result = check_workflow_trigger_safety(document, path)
        # None means this workflow either isn't held-out-named or is already safe.
        if result is not None:
            unsafe.append(result)
    return tuple(unsafe)


def _workflow_paths(workflows_dir: Path) -> list[Path]:
    """Discover every GitHub Actions workflow file (.yml and .yaml) in a directory.

    Args:
        workflows_dir: Path to the .github/workflows directory (or an override, in tests).

    Returns:
        Every workflow file path found, in glob order (sorted by the caller).
    """

    # A missing workflows directory means this check has nothing to validate against,
    # which likely indicates a misconfigured --workflows-dir rather than a repository
    # with genuinely zero workflows.
    if not workflows_dir.is_dir():
        raise HeldOutTriggerSafetyError(f"workflows directory does not exist: {workflows_dir}")
    return [*workflows_dir.glob("*.yml"), *workflows_dir.glob("*.yaml")]


def _load_workflow(path: Path) -> dict[str, Any]:
    """Load and parse one workflow YAML file as a top-level mapping.

    Args:
        path: Path to the workflow file to load.

    Returns:
        The parsed workflow document.
    """

    # Translate a missing, unreadable, or malformed-YAML file into
    # HeldOutTriggerSafetyError so callers only need to catch one exception type.
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise HeldOutTriggerSafetyError(f"could not parse workflow {path}: {error}") from error
    # Every field access downstream (name, on) assumes document is a dict; a workflow
    # file whose top level is a list or scalar isn't a valid GitHub Actions workflow.
    if not isinstance(document, dict):
        raise HeldOutTriggerSafetyError(f"workflow must be a YAML mapping: {path}")
    return document


def _on_value(document: dict[str, Any]) -> Any:
    """Recover a workflow's `on:` trigger value despite PyYAML's boolean coercion.

    YAML 1.1 parses the unquoted key `on:` as the boolean True, not the string "on"
    (the "Norway problem"). GitHub Actions workflows almost always write it unquoted.

    Args:
        document: The parsed workflow document.

    Returns:
        The trigger value, however it's actually keyed in the parsed dict.
    """

    # Prefer an explicit string "on" key if present (e.g. from an already-normalized
    # document); otherwise recover it from PyYAML's boolean-coerced True key.
    if "on" in document:
        return document["on"]
    return document.get(True)


def _unsafe_trigger_reasons(on_value: Any) -> tuple[str, ...]:
    """Identify every trigger on a workflow that isn't safely restricted.

    Args:
        on_value: The normalized `on:` value (see _on_value).

    Returns:
        Human-readable reasons for each unsafe trigger found; empty if all triggers
        are safe (workflow_dispatch, or push restricted to release-* tags).
    """

    triggers = _trigger_kinds(on_value)
    reasons: list[str] = []
    # Check every declared trigger kind independently, so a workflow with multiple
    # unsafe triggers gets every one reported, not just the first.
    for key, config in triggers.items():
        # workflow_dispatch is always safe: it requires an explicit manual invocation.
        if key == "workflow_dispatch":
            continue
        # "push" gets its own tag-restriction check below; every other trigger name
        # falls through to the disallowed-trigger message after this if/continue.
        if key == "push":
            # A push trigger is only safe when restricted to release-* tags; a bare
            # push (any branch) would let a held-out workflow run on routine commits.
            if not _push_is_release_tag_only(config):
                reasons.append("push trigger is not restricted to release-* tags")
            continue
        reasons.append(f"disallowed trigger: {key}")
    return tuple(reasons)


def _trigger_kinds(on_value: Any) -> dict[Any, Any]:
    """Normalize a workflow's `on:` value into a uniform trigger-name-to-config mapping.

    GitHub Actions accepts `on:` as a bare string, a list of strings, or a mapping with
    per-trigger config; this collapses all three shapes into one dict form so downstream
    code (_unsafe_trigger_reasons) only needs to handle one representation.

    Args:
        on_value: The raw `on:` value, in any of its three accepted shapes.

    Returns:
        A dict mapping each trigger name to its config (None if the trigger had no
        config, as with the bare-string/list forms).
    """

    # No `on:` key at all means no triggers are configured.
    if on_value is None:
        return {}
    # A bare string names exactly one trigger with no configuration.
    if isinstance(on_value, str):
        return {on_value: None}
    # A list names multiple triggers, none of which carry configuration.
    if isinstance(on_value, list):
        return {item: None for item in on_value}
    # A mapping already has the trigger-name-to-config shape this function returns.
    if isinstance(on_value, dict):
        return on_value
    raise HeldOutTriggerSafetyError(f"unrecognized 'on' trigger shape: {on_value!r}")


def _push_is_release_tag_only(push_config: Any) -> bool:
    """Return whether a `push:` trigger's config is restricted to release-* tags only.

    Args:
        push_config: The `push:` trigger's own config value (from _trigger_kinds).

    Returns:
        True only if push_config exclusively restricts to tags matching
        SAFE_RELEASE_TAG_PATTERN, with no branch filtering at all.
    """

    # A push trigger with no config (None, from the bare-string/list `on:` shapes) or
    # a non-dict config can't possibly declare a tags-only restriction.
    if not isinstance(push_config, dict):
        return False
    # Any branch filtering at all (branches or branches-ignore) means this push
    # trigger isn't tag-restricted, regardless of what tags: might also say.
    if push_config.get("branches") or push_config.get("branches-ignore"):
        return False
    tags = push_config.get("tags")
    # No tags list (or an empty one) means push isn't actually restricted to any tags.
    if not isinstance(tags, list) or not tags:
        return False
    return all(isinstance(tag, str) and SAFE_RELEASE_TAG_PATTERN.match(tag) for tag in tags)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the trigger-safety check entry point.

    Args:
        argv: Optional command-line arguments; defaults to the process arguments.

    Returns:
        Parsed arguments: the workflows directory to scan.
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
    # Every workflow-loading failure mode is collapsed into HeldOutTriggerSafetyError
    # by _load_workflow/_workflow_paths; catch it once here for uniform error reporting.
    try:
        unsafe = find_unsafe_workflows(args.workflows_dir)
    except HeldOutTriggerSafetyError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    # A non-empty unsafe list means at least one held-out-named workflow has an
    # unrestricted trigger; report every one explicitly and exit non-zero so this can gate CI.
    if unsafe:
        print(
            f"Unsafe held-out execution trigger(s) detected in {len(unsafe)} workflow(s):",
            file=sys.stderr,
        )
        # List every unsafe workflow's own reasons, so a reviewer can act on the
        # complete picture without re-running with different flags.
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


# Standard script entry-point guard: only run main() when executed directly, not when
# imported (e.g. by this script's own test module).
if __name__ == "__main__":
    raise SystemExit(main())
