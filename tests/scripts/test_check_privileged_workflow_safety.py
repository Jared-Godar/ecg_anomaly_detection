"""Tests for the pull_request_target privileged-workflow safety guard.

scripts/ holds standalone operational tooling, not the installed package, so the module under
test is loaded directly from its file path rather than imported as `ecg_anomaly_detection.*`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

# Locate the script relative to this test file, not the current working
# directory, so the test suite works regardless of where pytest is invoked from.
_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "check_privileged_workflow_safety.py"
)
# Load the script as a module by file path, since it's not installed as part of the
# package (see this file's module docstring for why).
_SPEC = importlib.util.spec_from_file_location("check_privileged_workflow_safety", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
# The module object every test in this file calls into (e.g. cpws.check_workflow_safety).
cpws = importlib.util.module_from_spec(_SPEC)
# Register the loaded module in sys.modules before executing it, matching the
# standard importlib.util pattern.
sys.modules[_SPEC.name] = cpws
_SPEC.loader.exec_module(cpws)


# A pull_request_target workflow honoring all five invariants: read-only ambient token,
# author-pinned job gate, base-ref checkout without persisted credentials, and dynamic
# values crossing into run: bodies only through env.
COMPLIANT_PRIVILEGED_YAML = """
name: Compliant privileged workflow
on:
  pull_request_target:
permissions:
  contents: read
jobs:
  autofill:
    runs-on: ubuntu-latest
    if: github.event.pull_request.user.login == 'dependabot[bot]'
    steps:
      - uses: actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0
        with:
          persist-credentials: false
      - name: Echo the PR title safely
        env:
          PR_TITLE: ${{ github.event.pull_request.title }}
        run: echo "$PR_TITLE"
"""

# Rule 1 violation: the checkout step overrides ref: to the untrusted PR head,
# putting attacker-controlled code on disk inside the privileged context.
HEAD_CHECKOUT_YAML = """
name: Head checkout
on:
  pull_request_target:
permissions:
  contents: read
jobs:
  autofill:
    runs-on: ubuntu-latest
    if: github.event.pull_request.user.login == 'dependabot[bot]'
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.head.sha }}
          persist-credentials: false
"""

# Rule 2 violation: a run: body interpolates an event field directly into shell
# source instead of passing it through env.
RUN_INTERPOLATION_YAML = """
name: Run interpolation
on:
  pull_request_target:
permissions:
  contents: read
jobs:
  autofill:
    runs-on: ubuntu-latest
    if: github.event.pull_request.user.login == 'dependabot[bot]'
    steps:
      - run: echo "${{ github.event.pull_request.title }}"
"""

# Rule 3 violation (missing gate): the job has no if: pinning the immutable PR author.
MISSING_AUTHOR_GATE_YAML = """
name: Missing author gate
on:
  pull_request_target:
permissions:
  contents: read
jobs:
  autofill:
    runs-on: ubuntu-latest
    steps:
      - run: echo noop
"""

# Rule 3 violation (actor reliance): the gate names the author but a step-level if:
# also consults github.actor, which changes to whoever clicks re-run.
ACTOR_CONDITION_YAML = """
name: Actor condition
on:
  pull_request_target:
permissions:
  contents: read
jobs:
  autofill:
    runs-on: ubuntu-latest
    if: github.event.pull_request.user.login == 'dependabot[bot]'
    steps:
      - if: github.actor == 'dependabot[bot]'
        run: echo noop
"""

# Rule 4 violation: the checkout step never opts out of credential persistence, so
# the workflow token lands in .git/config on disk.
PERSIST_CREDENTIALS_YAML = """
name: Persisted credentials
on:
  pull_request_target:
permissions:
  contents: read
jobs:
  autofill:
    runs-on: ubuntu-latest
    if: github.event.pull_request.user.login == 'dependabot[bot]'
    steps:
      - uses: actions/checkout@v4
"""

# Rule 5 violation (missing block): no top-level permissions block, so the ambient
# token inherits the repository default, which may be write-all.
MISSING_PERMISSIONS_YAML = """
name: Missing permissions
on:
  pull_request_target:
jobs:
  autofill:
    runs-on: ubuntu-latest
    if: github.event.pull_request.user.login == 'dependabot[bot]'
    steps:
      - run: echo noop
"""

# Rule 5 violation (write grant): the top-level permissions block explicitly grants
# contents: write to the ambient token.
CONTENTS_WRITE_YAML = """
name: Contents write
on:
  pull_request_target:
permissions:
  contents: write
jobs:
  autofill:
    runs-on: ubuntu-latest
    if: github.event.pull_request.user.login == 'dependabot[bot]'
    steps:
      - run: echo noop
"""

# A plain pull_request workflow committing every violation above -- out of scope,
# because it runs without base-repository privileges.
OUT_OF_SCOPE_PULL_REQUEST_YAML = """
name: Ordinary CI
on:
  pull_request:
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.head.sha }}
      - if: github.actor == 'someone'
        run: echo "${{ github.event.pull_request.title }}"
"""

# A compliant workflow declaring pull_request_target in the bare-list `on:` shape,
# exercising trigger-shape normalization alongside the mapping-form fixtures above.
LIST_TRIGGER_COMPLIANT_YAML = """
name: List trigger
on: [pull_request_target]
permissions:
  contents: read
jobs:
  autofill:
    runs-on: ubuntu-latest
    if: github.event.pull_request.user.login == 'dependabot[bot]'
    steps:
      - run: echo noop
"""


def _check(fixture_yaml: str, filename: str = "workflow.yml") -> Any:
    """Parse one fixture and run check_workflow_safety on it.

    Args:
        fixture_yaml: The workflow fixture text to parse.
        filename: The synthetic filename to attribute the workflow to.

    Returns:
        check_workflow_safety's result: an UnsafeWorkflow or None.
    """

    return cpws.check_workflow_safety(yaml.safe_load(fixture_yaml), Path(filename))


def _reasons(fixture_yaml: str) -> tuple[str, ...]:
    """Parse one fixture, assert it is flagged, and return its violation reasons.

    Args:
        fixture_yaml: The workflow fixture text to parse.

    Returns:
        The flagged UnsafeWorkflow's reasons tuple.
    """

    result = _check(fixture_yaml)
    assert result is not None
    return result.reasons


# --- scope --------------------------------------------------------------------------


def test_compliant_privileged_workflow_passes() -> None:
    """A pull_request_target workflow honoring all five invariants is not flagged."""

    assert _check(COMPLIANT_PRIVILEGED_YAML) is None


def test_plain_pull_request_workflow_is_out_of_scope() -> None:
    """A pull_request-triggered workflow is never inspected, even when it would violate every rule.

    Ordinary pull_request runs execute without base-repository privileges, so
    the privileged-trigger invariants simply don't apply to them.
    """

    assert _check(OUT_OF_SCOPE_PULL_REQUEST_YAML) is None


def test_list_shaped_trigger_is_in_scope() -> None:
    """pull_request_target declared via the bare-list `on:` shape still puts the workflow in scope."""

    assert _check(LIST_TRIGGER_COMPLIANT_YAML) is None
    assert cpws.uses_privileged_trigger(yaml.safe_load(LIST_TRIGGER_COMPLIANT_YAML))


def test_on_key_parsed_as_yaml_boolean_true_is_still_handled() -> None:
    """Scope detection works even though PyYAML parses the unquoted `on:` key as boolean True.

    PyYAML follows YAML 1.1, under which the unquoted key `on` is parsed as
    the boolean True rather than the string "on" (the Norway problem); the
    guard must handle both possible key forms.
    """

    document = yaml.safe_load(COMPLIANT_PRIVILEGED_YAML)
    assert True in document or "on" in document
    assert cpws.uses_privileged_trigger(document)


# --- one fixture per rule -------------------------------------------------------------


def test_pr_head_checkout_is_flagged() -> None:
    """Rule 1: a checkout whose with.ref points at the PR head is flagged, naming the rule.

    Checking out attacker-controlled code inside the privileged context is one
    later build/install step away from remote code execution with secrets.
    """

    assert any("no PR-head checkout" in reason for reason in _reasons(HEAD_CHECKOUT_YAML))


def test_run_body_interpolation_is_flagged() -> None:
    """Rule 2: any '${{' inside a run: body is flagged, naming the rule.

    Expression interpolation splices attacker-influenced event fields into
    shell source before parsing -- template injection; env: is the safe path.
    """

    assert any("no run interpolation" in reason for reason in _reasons(RUN_INTERPOLATION_YAML))


def test_missing_author_gate_is_flagged() -> None:
    """Rule 3: a job with no if: pinning github.event.pull_request.user.login is flagged."""

    assert any("immutable author gate" in reason for reason in _reasons(MISSING_AUTHOR_GATE_YAML))


def test_actor_in_condition_is_flagged() -> None:
    """Rule 3: an if: consulting github.actor is flagged even when the author gate is present.

    github.actor changes to whoever clicks re-run, so an actor-based check can
    be satisfied by re-running an attacker's workflow run.
    """

    reasons = _reasons(ACTOR_CONDITION_YAML)
    assert any("github.actor" in reason for reason in reasons)
    assert any("immutable author gate" in reason for reason in reasons)


def test_missing_persist_credentials_false_is_flagged() -> None:
    """Rule 4: a checkout step that doesn't set persist-credentials: false is flagged.

    actions/checkout persists the workflow token into .git/config by default,
    where any later step could read it off disk.
    """

    assert any(
        "no persisted credentials" in reason for reason in _reasons(PERSIST_CREDENTIALS_YAML)
    )


def test_missing_top_level_permissions_is_flagged() -> None:
    """Rule 5: a workflow without a top-level permissions block is flagged."""

    assert any("no ambient write token" in reason for reason in _reasons(MISSING_PERMISSIONS_YAML))


def test_contents_write_permission_is_flagged() -> None:
    """Rule 5: a top-level permissions block granting contents: write is flagged.

    The ambient GITHUB_TOKEN must stay read-only; any write path belongs to a
    scoped credential exposed via one step's env.
    """

    assert any("contents: write" in reason for reason in _reasons(CONTENTS_WRITE_YAML))


# --- find_unsafe_workflows / main ------------------------------------------------------


def test_find_unsafe_workflows_flags_only_in_scope_violating_files(tmp_path: Path) -> None:
    """Of three workflow files (violating-and-privileged, violating-but-ordinary, compliant),
    only the first is reported by find_unsafe_workflows.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    workflows_dir = tmp_path / "workflows"
    workflows_dir.mkdir()
    (workflows_dir / "head-checkout.yml").write_text(HEAD_CHECKOUT_YAML, encoding="utf-8")
    (workflows_dir / "ordinary-ci.yml").write_text(OUT_OF_SCOPE_PULL_REQUEST_YAML, encoding="utf-8")
    (workflows_dir / "compliant.yml").write_text(COMPLIANT_PRIVILEGED_YAML, encoding="utf-8")

    unsafe = cpws.find_unsafe_workflows(workflows_dir)

    assert len(unsafe) == 1
    assert unsafe[0].path.name == "head-checkout.yml"


def test_find_unsafe_workflows_raises_on_missing_directory(tmp_path: Path) -> None:
    """Pointing find_unsafe_workflows at a directory that doesn't exist raises a specific, actionable error.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    # tmp_path / "missing" was never created.
    with pytest.raises(cpws.PrivilegedWorkflowSafetyError, match="does not exist"):
        cpws.find_unsafe_workflows(tmp_path / "missing")


def test_main_returns_zero_when_no_violations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With only a compliant privileged workflow present, main exits 0 and prints a confirmation.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
        capsys: Used to inspect main's printed stdout.
    """

    workflows_dir = tmp_path / "workflows"
    workflows_dir.mkdir()
    (workflows_dir / "compliant.yml").write_text(COMPLIANT_PRIVILEGED_YAML, encoding="utf-8")

    exit_code = cpws.main(["--workflows-dir", str(workflows_dir)])

    assert exit_code == 0
    assert "No privileged-workflow safety violations" in capsys.readouterr().out


def test_main_returns_one_when_a_violation_is_found(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With one violating privileged workflow present, main exits 1 and prints the violation to stderr.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
        capsys: Used to inspect main's printed stderr.
    """

    workflows_dir = tmp_path / "workflows"
    workflows_dir.mkdir()
    (workflows_dir / "head-checkout.yml").write_text(HEAD_CHECKOUT_YAML, encoding="utf-8")

    exit_code = cpws.main(["--workflows-dir", str(workflows_dir)])

    assert exit_code == 1
    assert "Privileged-workflow safety violation" in capsys.readouterr().err


def test_main_returns_two_on_malformed_yaml(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A workflow file that isn't parseable YAML makes main exit 2 (required data unreadable).

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
        capsys: Used to inspect main's printed stderr.
    """

    workflows_dir = tmp_path / "workflows"
    workflows_dir.mkdir()
    # An unclosed flow mapping is invalid YAML in any dialect.
    (workflows_dir / "broken.yml").write_text("on: {pull_request_target", encoding="utf-8")

    exit_code = cpws.main(["--workflows-dir", str(workflows_dir)])

    assert exit_code == 2
    assert "could not parse workflow" in capsys.readouterr().err


def test_main_against_the_real_workflows_directory_currently_passes() -> None:
    """Running main with no arguments against this repository's actual .github/workflows/ passes.

    Any pull_request_target workflow this repository carries must satisfy all
    five invariants; this assertion turns the guard into a live regression
    gate the moment such a workflow is added or edited.
    """

    assert cpws.main([]) == 0
