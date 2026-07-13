#!/usr/bin/env python3
"""Fail if a `pull_request_target` workflow breaks the invariants that make it safe to run.

This is a governance-as-code guard, not an execution command: it never runs or dispatches a
workflow. A workflow triggered by `pull_request_target` executes in the *base* repository's
privileged context (secrets available, write-capable ambient token) while being triggered by
*untrusted* pull-request content. Its entire security argument is therefore a small set of
structural invariants that a future well-meaning edit could silently break. This script reads
every `.github/workflows/*.yml`/`*.yaml` file and, for any workflow whose triggers include
`pull_request_target` (in any of the accepted `on:` shapes), mechanically enforces five rules:

1. No PR-head checkout: no `actions/checkout` step may set `with.ref` to the pull request's
   head. Checking out attacker-controlled code inside the privileged context is remote code
   execution the moment any later step runs anything from the tree.
2. No `${{ ... }}` expression interpolation inside any `run:` body. Interpolation splices
   attacker-influenced values (branch names, PR titles, ...) directly into shell source --
   classic template/script injection. Dynamic values must cross into shell as `env:` variables.
3. Immutable-author gate: every job's `if:` must pin `github.event.pull_request.user.login`,
   and no `if:` may consult `github.actor` -- the actor changes to whoever clicks re-run, so an
   actor-based gate can be spoofed by re-running someone else's workflow run.
4. `persist-credentials: false` on every `actions/checkout` step, so the privileged token is
   never written into the on-disk Git config where later (possibly injected) steps could read it.
5. A top-level `permissions:` block must exist and must not grant `contents: write` -- the
   ambient GITHUB_TOKEN stays read-only, and any write path must be a scoped credential exposed
   to exactly one step via `env:`.

Every violation is reported independently. See the guard's sibling,
scripts/check_held_out_trigger_safety.py, for the architectural template this follows.
"""

from __future__ import annotations

import argparse
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
# The privileged trigger this guard governs: any workflow declaring it is in scope.
PRIVILEGED_TRIGGER = "pull_request_target"
# The exact expression fragment every job's if: gate must contain. The PR author's login is
# immutable for the lifetime of the pull request, unlike github.actor which changes on re-run.
REQUIRED_AUTHOR_GATE_FRAGMENT = "github.event.pull_request.user.login =="
# Substrings that mark a checkout ref as pointing at the untrusted PR head rather than the
# trusted base. Matched as substrings so every spelling (refs/pull/N/head,
# github.event.pull_request.head.sha, .head.ref, ...) is caught.
FORBIDDEN_REF_SUBSTRINGS = ("head", "github.event.pull_request")


class PrivilegedWorkflowSafetyError(RuntimeError):
    """Raised when a workflow file cannot be read or parsed as a GitHub Actions workflow."""


@dataclass(frozen=True, slots=True)
class UnsafeWorkflow:
    """One pull_request_target workflow that violates a privileged-trigger safety invariant."""

    path: Path
    name: str
    reasons: tuple[str, ...]


def uses_privileged_trigger(document: dict[str, Any]) -> bool:
    """Return whether a workflow's triggers include pull_request_target in any accepted shape."""
    return PRIVILEGED_TRIGGER in _trigger_kinds(_on_value(document))


def check_workflow_safety(document: dict[str, Any], path: Path) -> UnsafeWorkflow | None:
    """Return an UnsafeWorkflow if a pull_request_target workflow violates any safety rule.

    Args:
        document: The parsed workflow document.
        path: Path of the workflow file, used for scoping and reporting.

    Returns:
        An UnsafeWorkflow carrying every violation found, or None when the workflow is
        out of scope (no pull_request_target trigger) or fully compliant.
    """

    # Workflows without a pull_request_target trigger are entirely out of scope for this
    # guard -- their jobs, steps, and permissions are never inspected. Routine triggers
    # (push, pull_request, schedule, ...) run without base-repository privileges.
    if not uses_privileged_trigger(document):
        return None
    reasons = _violation_reasons(document)
    # No violation reasons means every privileged-trigger invariant holds; nothing to report.
    if reasons:
        name = document.get("name") if isinstance(document.get("name"), str) else path.stem
        return UnsafeWorkflow(path, name, reasons)
    return None


def find_unsafe_workflows(workflows_dir: Path) -> tuple[UnsafeWorkflow, ...]:
    """Parse every workflow file and return the pull_request_target ones with violations."""
    unsafe: list[UnsafeWorkflow] = []
    # Process workflow files in sorted order for deterministic output.
    for path in sorted(_workflow_paths(workflows_dir)):
        document = _load_workflow(path)
        result = check_workflow_safety(document, path)
        # None means this workflow either isn't privileged-triggered or is fully compliant.
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
        raise PrivilegedWorkflowSafetyError(f"workflows directory does not exist: {workflows_dir}")
    return [*workflows_dir.glob("*.yml"), *workflows_dir.glob("*.yaml")]


def _load_workflow(path: Path) -> dict[str, Any]:
    """Load and parse one workflow YAML file as a top-level mapping.

    Args:
        path: Path to the workflow file to load.

    Returns:
        The parsed workflow document.
    """

    # Translate a missing, unreadable, or malformed-YAML file into
    # PrivilegedWorkflowSafetyError so callers only need to catch one exception type.
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise PrivilegedWorkflowSafetyError(f"could not parse workflow {path}: {error}") from error
    # Every field access downstream (name, on, jobs, permissions) assumes document is a
    # dict; a workflow file whose top level is a list or scalar isn't a valid workflow.
    if not isinstance(document, dict):
        raise PrivilegedWorkflowSafetyError(f"workflow must be a YAML mapping: {path}")
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


def _trigger_kinds(on_value: Any) -> dict[Any, Any]:
    """Normalize a workflow's `on:` value into a uniform trigger-name-to-config mapping.

    GitHub Actions accepts `on:` as a bare string, a list of strings, or a mapping with
    per-trigger config; this collapses all three shapes into one dict form so scope
    detection (uses_privileged_trigger) only needs to handle one representation.

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
    raise PrivilegedWorkflowSafetyError(f"unrecognized 'on' trigger shape: {on_value!r}")


def _violation_reasons(document: dict[str, Any]) -> tuple[str, ...]:
    """Collect every privileged-trigger safety violation in one in-scope workflow.

    Args:
        document: The parsed workflow document (already confirmed in scope).

    Returns:
        A human-readable reason per violation, each naming the rule it belongs to;
        empty when all five invariants hold.
    """

    reasons: list[str] = [*_permissions_reasons(document)]
    jobs = document.get("jobs")
    # A workflow without a well-formed jobs mapping has no job-level surface to audit;
    # the per-job rules are then vacuously satisfied (actionlint owns schema validity).
    if not isinstance(jobs, dict):
        return tuple(reasons)
    # Audit every job independently so a multi-job workflow reports each job's own
    # violations rather than stopping at the first offender.
    for job_id, job in jobs.items():
        reasons.extend(_job_reasons(str(job_id), job))
    return tuple(reasons)


def _permissions_reasons(document: dict[str, Any]) -> tuple[str, ...]:
    """Check rule 5: the ambient GITHUB_TOKEN must be explicitly locked to read-only.

    Attack foreclosed: without an explicit top-level `permissions:` block the ambient
    token can default to write access, so any injected or compromised step could push
    commits, rewrite releases, or tamper with the repository. The write path (if one is
    needed at all) must be a scoped credential exposed via a single step's `env:`.

    Args:
        document: The parsed workflow document.

    Returns:
        Zero or one reason describing the ambient-token violation.
    """

    # An absent permissions block leaves the token at the repository/organization
    # default, which may be write-all -- the safety argument must not depend on that.
    if "permissions" not in document:
        return ("top-level permissions block is missing (rule: no ambient write token)",)
    permissions = document["permissions"]
    # The shorthand string form: "read-all" is safe; "write-all" (or anything else)
    # grants the ambient token write scopes, including contents.
    if isinstance(permissions, str):
        # read-all is the only shorthand that keeps every scope, contents included, read-only.
        if permissions.strip() == "read-all":
            return ()
        return (
            f"top-level permissions {permissions!r} grants write access"
            " (rule: no ambient write token)",
        )
    # The mapping form: only an explicit `contents: write` grant is forbidden; every
    # other scope combination leaves repository contents read-only.
    if isinstance(permissions, dict):
        # Only the contents scope guards repository writes; other scopes stay out of
        # this rule's remit.
        if permissions.get("contents") == "write":
            return ("top-level permissions grants contents: write (rule: no ambient write token)",)
        return ()
    # A bare `permissions:` key parses as None, which GitHub treats as granting nothing.
    if permissions is None:
        return ()
    return (
        f"top-level permissions has unrecognized shape {permissions!r}"
        " (rule: no ambient write token)",
    )


def _job_reasons(job_id: str, job: Any) -> tuple[str, ...]:
    """Collect every rule violation inside one job of an in-scope workflow.

    Args:
        job_id: The job's key under `jobs:`, used in reason messages.
        job: The job's parsed value (expected to be a mapping).

    Returns:
        A reason per violation found in this job's gate condition and steps.
    """

    # A non-mapping job value can't declare an if: gate, so the author-gate rule fails
    # by construction; deeper schema problems are actionlint's concern, not ours.
    if not isinstance(job, dict):
        return (
            f"job {job_id!r} has no if: gate pinning"
            " github.event.pull_request.user.login (rule: immutable author gate)",
        )
    reasons = [*_author_gate_reasons(job_id, job)]
    steps = job.get("steps")
    # Jobs without a well-formed steps list (e.g. reusable-workflow calls) have no
    # step-level surface for the checkout/run rules to inspect.
    if not isinstance(steps, list):
        return tuple(reasons)
    # Number steps from 1 in messages so they line up with how humans read the YAML.
    for position, step in enumerate(steps, start=1):
        reasons.extend(_step_reasons(job_id, position, step))
    return tuple(reasons)


def _author_gate_reasons(job_id: str, job: dict[str, Any]) -> tuple[str, ...]:
    """Check rule 3: every job gates on the immutable PR author, never on the actor.

    Attack foreclosed: `github.actor` is whoever *caused the current run* -- on a
    re-run, that's the person who clicked re-run, not the PR author. A gate written
    against the actor can therefore be satisfied by a trusted maintainer re-running an
    attacker's workflow run. `github.event.pull_request.user.login` is fixed at PR
    creation and cannot be changed by re-running.

    Args:
        job_id: The job's key under `jobs:`, used in reason messages.
        job: The job's parsed mapping.

    Returns:
        A reason for a missing author gate and/or for each if: that consults the actor.
    """

    reasons: list[str] = []
    job_condition = job.get("if")
    # The job-level if: must exist, be textual, and pin the PR author's login with an
    # equality check; anything else fails to prove only the intended author reaches
    # the privileged steps.
    if not (isinstance(job_condition, str) and REQUIRED_AUTHOR_GATE_FRAGMENT in job_condition):
        reasons.append(
            f"job {job_id!r} has no if: gate pinning"
            " github.event.pull_request.user.login (rule: immutable author gate)"
        )
    # Sweep every if: expression in the job -- the job's own and each step's -- for
    # github.actor, since an actor check anywhere reintroduces the re-run spoof.
    for location, condition in _all_conditions(job):
        # Non-string conditions (booleans, numbers) can't reference the actor at all.
        if isinstance(condition, str) and "github.actor" in condition:
            reasons.append(
                f"job {job_id!r} {location} if: relies on github.actor,"
                " which changes on re-run (rule: immutable author gate)"
            )
    return tuple(reasons)


def _all_conditions(job: dict[str, Any]) -> list[tuple[str, Any]]:
    """Enumerate every if: condition in a job, labeled by where it appears.

    Args:
        job: The job's parsed mapping.

    Returns:
        (location-label, condition-value) pairs for the job-level if: and each
        step-level if: found in a well-formed steps list.
    """

    conditions: list[tuple[str, Any]] = []
    # The job-level condition guards the entire job, so it's always in the sweep.
    if "if" in job:
        conditions.append(("job-level", job["if"]))
    steps = job.get("steps")
    # Only a well-formed list of mapping steps can carry step-level conditions.
    if isinstance(steps, list):
        # Number steps from 1 in labels so they line up with how humans read the YAML.
        for position, step in enumerate(steps, start=1):
            # A step-level if: only exists on mapping-shaped steps that declare one.
            if isinstance(step, dict) and "if" in step:
                conditions.append((f"step {position}", step["if"]))
    return conditions


def _step_reasons(job_id: str, position: int, step: Any) -> tuple[str, ...]:
    """Collect the checkout- and run-body violations for one step.

    Args:
        job_id: The enclosing job's key, used in reason messages.
        position: The step's 1-based position within the job's steps list.
        step: The step's parsed value (expected to be a mapping).

    Returns:
        A reason per rule-1/2/4 violation found in this step.
    """

    # A non-mapping step can't declare uses:/run:/with:, so no step rule can apply.
    if not isinstance(step, dict):
        return ()
    reasons: list[str] = []
    run_body = step.get("run")
    # Rule 2 -- attack foreclosed: `${{ ... }}` inside a run: body is expanded *before*
    # the shell parses the script, so attacker-influenced event fields (PR title,
    # branch name, body) become shell source verbatim: template injection. Requiring
    # every dynamic value to cross as an env: variable keeps it inert data. Strict by
    # design (no allowlisted expressions) -- that strictness is the price of a
    # privileged trigger.
    if isinstance(run_body, str) and "${{" in run_body:
        reasons.append(
            f"job {job_id!r} step {position} run: body contains '${{{{' expression"
            " interpolation; pass dynamic values via env (rule: no run interpolation)"
        )
    # Steps that don't invoke actions/checkout carry no checkout-specific obligations.
    if not _is_checkout_step(step):
        return tuple(reasons)
    with_config = step.get("with")
    # Normalize a missing/malformed with: block to an empty mapping: a checkout with
    # no with: at all still violates rule 4 (persist-credentials defaults to true).
    if not isinstance(with_config, dict):
        with_config = {}
    ref = with_config.get("ref")
    # Rule 1 -- attack foreclosed: pull_request_target runs with base-repository
    # privileges, and by default checks out the trusted *base* ref. Overriding ref: to
    # the PR head puts attacker-controlled files (scripts, configs, lockfiles) on disk
    # inside the privileged context -- one later `npm install`/`make`/`uv run` away
    # from remote code execution with secrets in the environment. A missing ref: is
    # the safe base default.
    if isinstance(ref, str) and any(marker in ref for marker in FORBIDDEN_REF_SUBSTRINGS):
        reasons.append(
            f"job {job_id!r} step {position} checks out the untrusted PR head via"
            f" with.ref: {ref!r} (rule: no PR-head checkout)"
        )
    # Rule 4 -- attack foreclosed: actions/checkout persists the workflow token into
    # .git/config by default, so any later step (including an injected one) could read
    # it off disk and reuse it after the gating conditions were evaluated. The step
    # must opt out explicitly with the boolean false.
    if with_config.get("persist-credentials") is not False:
        reasons.append(
            f"job {job_id!r} step {position} checkout does not set"
            " persist-credentials: false (rule: no persisted credentials)"
        )
    return tuple(reasons)


def _is_checkout_step(step: dict[str, Any]) -> bool:
    """Return whether a step invokes actions/checkout, at any version suffix.

    Args:
        step: The step's parsed mapping.

    Returns:
        True for `uses: actions/checkout` with or without an @-pinned version.
    """

    uses = step.get("uses")
    # Match the bare action name or any @-suffixed pin (tag, branch, or full SHA),
    # without also matching unrelated actions that merely share the prefix.
    return isinstance(uses, str) and (
        uses == "actions/checkout" or uses.startswith("actions/checkout@")
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the privileged-workflow safety check entry point.

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
        0 when every pull_request_target workflow is compliant (or none exists), 1 when
        at least one violation was found, 2 when a workflow file could not be read.
    """

    args = parse_args(argv)
    # Every workflow-loading failure mode is collapsed into PrivilegedWorkflowSafetyError
    # by _load_workflow/_workflow_paths; catch it once here for uniform error reporting.
    try:
        unsafe = find_unsafe_workflows(args.workflows_dir)
    except PrivilegedWorkflowSafetyError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    # A non-empty unsafe list means at least one pull_request_target workflow broke an
    # invariant; report every one explicitly and exit non-zero so this can gate CI.
    if unsafe:
        print(
            f"Privileged-workflow safety violation(s) detected in {len(unsafe)} workflow(s):",
            file=sys.stderr,
        )
        # List every unsafe workflow's own reasons, so a reviewer can act on the
        # complete picture without re-running with different flags.
        for item in unsafe:
            print(f"  - {item.path.name} ({item.name!r}):", file=sys.stderr)
            # One reason per line keeps multi-violation reports scannable in CI logs.
            for reason in item.reasons:
                print(f"      * {reason}", file=sys.stderr)
        print(
            "\npull_request_target workflows run untrusted PR events with base-repository"
            " privileges; every structural invariant above is load-bearing. See the module"
            " docstring of scripts/check_privileged_workflow_safety.py for the rules and"
            " the attacks they foreclose.",
            file=sys.stderr,
        )
        return 1

    print("No privileged-workflow safety violations detected.")
    return 0


# Standard script entry-point guard: only run main() when executed directly, not when
# imported (e.g. by this script's own test module).
if __name__ == "__main__":
    raise SystemExit(main())
