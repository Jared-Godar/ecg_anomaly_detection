"""Tests for the held-out execution CI trigger safety guard.

scripts/ holds standalone operational tooling, not the installed package, so the module under
test is loaded directly from its file path rather than imported as `ecg_anomaly_detection.*`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

# Locate the script relative to this test file, not the current working
# directory, so the test suite works regardless of where pytest is invoked from.
_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "check_held_out_trigger_safety.py"
# Load the script as a module by file path, since it's not installed as part of the
# package (see this file's module docstring for why).
_SPEC = importlib.util.spec_from_file_location("check_held_out_trigger_safety", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
# The module object every test in this file calls into (e.g. chts.check_workflow_trigger_safety).
chts = importlib.util.module_from_spec(_SPEC)
# Register the loaded module in sys.modules before executing it, matching the
# standard importlib.util pattern.
sys.modules[_SPEC.name] = chts
_SPEC.loader.exec_module(chts)


# A held-out-named workflow with only safe triggers (manual dispatch, release tags).
SAFE_HELD_OUT_YAML = """
name: Held-out execution
on:
  workflow_dispatch:
  push:
    tags: ["release-*"]
jobs:
  execute:
    runs-on: ubuntu-latest
    steps:
      - run: echo noop
"""

# A held-out-named workflow with a pull_request trigger, which could run on
# any contributor's PR without human approval.
UNSAFE_PULL_REQUEST_YAML = """
name: Held-out execution
on:
  pull_request:
  workflow_dispatch:
jobs:
  execute:
    runs-on: ubuntu-latest
    steps:
      - run: echo noop
"""

# A held-out-named workflow with an unrestricted `push:` trigger (no branch or tag filter).
UNSAFE_BARE_PUSH_YAML = """
name: Benchmark execution
on:
  push:
jobs:
  execute:
    runs-on: ubuntu-latest
    steps:
      - run: echo noop
"""

# A held-out-named workflow whose push trigger is filtered by tag, but which also
# allows pushes to "main" -- release tags alone are not enough to make this safe.
UNSAFE_PUSH_BRANCHES_YAML = """
name: held-out-run
on:
  push:
    branches: [main]
    tags: ["release-*"]
jobs:
  execute:
    runs-on: ubuntu-latest
    steps:
      - run: echo noop
"""

# A workflow with the same unsafe triggers as the fixtures above, but whose name and
# filename don't match held-out/benchmark-execution naming at all.
UNRELATED_WORKFLOW_WITH_UNSAFE_TRIGGERS_YAML = """
name: Quality gates
on:
  pull_request:
  push:
    branches: [main]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: echo noop
"""


# --- matches_held_out_naming ---------------------------------------------------------


def test_matches_held_out_naming_by_filename() -> None:
    """A workflow file named "held-out-execution.yml" matches by filename alone, regardless of its declared name."""

    assert chts.matches_held_out_naming(Path("held-out-execution.yml"), "Quality gates")


def test_matches_held_out_naming_by_declared_name() -> None:
    """A workflow whose declared `name:` contains "Benchmark Execution" matches even with an unrelated filename."""

    assert chts.matches_held_out_naming(Path("run-model.yml"), "Benchmark Execution")


def test_matches_held_out_naming_false_for_unrelated_workflow() -> None:
    """A workflow matching neither the filename nor the name pattern is not treated as held-out-related."""

    assert not chts.matches_held_out_naming(Path("quality.yml"), "Quality gates")


# --- check_workflow_trigger_safety ----------------------------------------------------


def test_safe_held_out_workflow_is_not_flagged() -> None:
    """A held-out workflow triggered only by workflow_dispatch and tagged pushes passes with no result."""

    document = yaml.safe_load(SAFE_HELD_OUT_YAML)
    assert chts.check_workflow_trigger_safety(document, Path("held-out-execution.yml")) is None


def test_pull_request_trigger_is_flagged() -> None:
    """A held-out workflow with a pull_request trigger is flagged, naming "pull_request" in its reasons.

    A pull_request trigger would let any contributor's PR run the held-out
    workflow without a separate human approval step, defeating the point of
    the governance gate.
    """

    document = yaml.safe_load(UNSAFE_PULL_REQUEST_YAML)
    result = chts.check_workflow_trigger_safety(document, Path("held-out-execution.yml"))
    assert result is not None
    assert any("pull_request" in reason for reason in result.reasons)


def test_bare_push_trigger_is_flagged() -> None:
    """A held-out workflow with an unfiltered `push:` trigger is flagged, its reason naming the missing tag filter."""

    document = yaml.safe_load(UNSAFE_BARE_PUSH_YAML)
    result = chts.check_workflow_trigger_safety(document, Path("benchmark-execution.yml"))
    assert result is not None
    assert any("release-*" in reason for reason in result.reasons)


def test_push_with_branches_is_flagged_even_with_release_tags() -> None:
    """A push trigger filtered by release tags is still flagged if it also allows pushes to a branch.

    Any ordinary branch push (e.g. to "main") must not be able to trigger
    held-out execution, even if the same trigger also happens to restrict by
    tag.
    """

    document = yaml.safe_load(UNSAFE_PUSH_BRANCHES_YAML)
    result = chts.check_workflow_trigger_safety(document, Path("held-out-run.yml"))
    assert result is not None


def test_unrelated_workflow_with_unsafe_triggers_is_not_flagged() -> None:
    """A workflow with the same unsafe triggers is not flagged at all when it isn't held-out-named.

    This guard only governs held-out/benchmark-execution workflows; an
    ordinary CI workflow with a pull_request trigger is expected and out of
    scope.
    """

    document = yaml.safe_load(UNRELATED_WORKFLOW_WITH_UNSAFE_TRIGGERS_YAML)
    assert chts.check_workflow_trigger_safety(document, Path("quality.yml")) is None


def test_on_key_parsed_as_yaml_boolean_true_is_still_handled() -> None:
    """A safe workflow is still recognized correctly even though PyYAML parses its unquoted `on:` key as boolean True.

    PyYAML follows YAML 1.1, under which the unquoted key `on` is parsed as
    the boolean True rather than the string "on"; the safety check must
    handle both possible key forms.
    """

    document = yaml.safe_load(SAFE_HELD_OUT_YAML)
    assert True in document or "on" in document
    assert chts.check_workflow_trigger_safety(document, Path("held-out-execution.yml")) is None


# --- find_unsafe_workflows / main ------------------------------------------------------


def test_find_unsafe_workflows_flags_only_held_out_named_unsafe_files(tmp_path: Path) -> None:
    """Of three workflow files (unsafe-and-named, unsafe-but-unrelated, safe-and-named), only the
    first is reported by find_unsafe_workflows.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    workflows_dir = tmp_path / "workflows"
    workflows_dir.mkdir()
    (workflows_dir / "held-out-execution.yml").write_text(
        UNSAFE_PULL_REQUEST_YAML, encoding="utf-8"
    )
    (workflows_dir / "quality.yml").write_text(
        UNRELATED_WORKFLOW_WITH_UNSAFE_TRIGGERS_YAML, encoding="utf-8"
    )
    (workflows_dir / "safe-held-out.yml").write_text(SAFE_HELD_OUT_YAML, encoding="utf-8")

    unsafe = chts.find_unsafe_workflows(workflows_dir)

    assert len(unsafe) == 1
    assert unsafe[0].path.name == "held-out-execution.yml"


def test_find_unsafe_workflows_raises_on_missing_directory(tmp_path: Path) -> None:
    """Pointing find_unsafe_workflows at a directory that doesn't exist raises a specific, actionable error.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    # tmp_path / "missing" was never created.
    with pytest.raises(chts.HeldOutTriggerSafetyError, match="does not exist"):
        chts.find_unsafe_workflows(tmp_path / "missing")


def test_main_returns_zero_when_no_unsafe_workflows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With only a safe held-out workflow present, main exits 0 and prints a "No unsafe" confirmation.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
        capsys: Used to inspect main's printed stdout.
    """

    workflows_dir = tmp_path / "workflows"
    workflows_dir.mkdir()
    (workflows_dir / "safe-held-out.yml").write_text(SAFE_HELD_OUT_YAML, encoding="utf-8")

    exit_code = chts.main(["--workflows-dir", str(workflows_dir)])

    assert exit_code == 0
    assert "No unsafe" in capsys.readouterr().out


def test_main_returns_one_when_an_unsafe_workflow_is_found(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With one unsafe held-out workflow present, main exits 1 and prints the violation to stderr.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
        capsys: Used to inspect main's printed stderr.
    """

    workflows_dir = tmp_path / "workflows"
    workflows_dir.mkdir()
    (workflows_dir / "held-out-execution.yml").write_text(UNSAFE_BARE_PUSH_YAML, encoding="utf-8")

    exit_code = chts.main(["--workflows-dir", str(workflows_dir)])

    assert exit_code == 1
    assert "Unsafe held-out execution trigger" in capsys.readouterr().err


def test_main_against_the_real_workflows_directory_currently_passes() -> None:
    """Running main with no arguments against this repository's actual .github/workflows/ currently passes.

    No held-out/benchmark-execution workflow exists in this repository yet,
    so this assertion is trivially true today -- its purpose is to guard
    against one being added later with an unsafe trigger, since it would
    fail this test immediately.
    """

    assert chts.main([]) == 0
