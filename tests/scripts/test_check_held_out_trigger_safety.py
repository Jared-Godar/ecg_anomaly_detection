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

# Centralize _SCRIPT_PATH so every caller shares the same documented invariant.
_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "check_held_out_trigger_safety.py"
# Centralize _SPEC so every caller shares the same documented invariant.
_SPEC = importlib.util.spec_from_file_location("check_held_out_trigger_safety", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
# Construct chts once so the module exposes one stable shared definition.
chts = importlib.util.module_from_spec(_SPEC)
# Construct this module object once so the module exposes one stable shared definition.
sys.modules[_SPEC.name] = chts
_SPEC.loader.exec_module(chts)


# Centralize SAFE_HELD_OUT_YAML so every caller shares the same documented invariant.
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

# Centralize UNSAFE_PULL_REQUEST_YAML so every caller shares the same documented invariant.
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

# Centralize UNSAFE_BARE_PUSH_YAML so every caller shares the same documented invariant.
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

# Centralize UNSAFE_PUSH_BRANCHES_YAML so every caller shares the same documented invariant.
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

# Centralize UNRELATED_WORKFLOW_WITH_UNSAFE_TRIGGERS_YAML so every caller shares the same documented invariant.
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
    """Verify that matches held out naming by filename.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

    assert chts.matches_held_out_naming(Path("held-out-execution.yml"), "Quality gates")


def test_matches_held_out_naming_by_declared_name() -> None:
    """Verify that matches held out naming by declared name.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

    assert chts.matches_held_out_naming(Path("run-model.yml"), "Benchmark Execution")


def test_matches_held_out_naming_false_for_unrelated_workflow() -> None:
    """Verify that matches held out naming false for unrelated workflow.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

    assert not chts.matches_held_out_naming(Path("quality.yml"), "Quality gates")


# --- check_workflow_trigger_safety ----------------------------------------------------


def test_safe_held_out_workflow_is_not_flagged() -> None:
    """Verify that safe held out workflow is not flagged.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

    document = yaml.safe_load(SAFE_HELD_OUT_YAML)
    assert chts.check_workflow_trigger_safety(document, Path("held-out-execution.yml")) is None


def test_pull_request_trigger_is_flagged() -> None:
    """Verify that pull request trigger is flagged.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

    document = yaml.safe_load(UNSAFE_PULL_REQUEST_YAML)
    result = chts.check_workflow_trigger_safety(document, Path("held-out-execution.yml"))
    assert result is not None
    assert any("pull_request" in reason for reason in result.reasons)


def test_bare_push_trigger_is_flagged() -> None:
    """Verify that bare push trigger is flagged.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

    document = yaml.safe_load(UNSAFE_BARE_PUSH_YAML)
    result = chts.check_workflow_trigger_safety(document, Path("benchmark-execution.yml"))
    assert result is not None
    assert any("release-*" in reason for reason in result.reasons)


def test_push_with_branches_is_flagged_even_with_release_tags() -> None:
    """Verify that push with branches is flagged even with release tags.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

    document = yaml.safe_load(UNSAFE_PUSH_BRANCHES_YAML)
    result = chts.check_workflow_trigger_safety(document, Path("held-out-run.yml"))
    assert result is not None


def test_unrelated_workflow_with_unsafe_triggers_is_not_flagged() -> None:
    """Verify that unrelated workflow with unsafe triggers is not flagged.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

    document = yaml.safe_load(UNRELATED_WORKFLOW_WITH_UNSAFE_TRIGGERS_YAML)
    assert chts.check_workflow_trigger_safety(document, Path("quality.yml")) is None


def test_on_key_parsed_as_yaml_boolean_true_is_still_handled() -> None:
    # PyYAML follows YAML 1.1 and parses the unquoted key `on:` as boolean True.
    """Verify that on key parsed as yaml boolean true is still handled.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

    document = yaml.safe_load(SAFE_HELD_OUT_YAML)
    assert True in document or "on" in document
    assert chts.check_workflow_trigger_safety(document, Path("held-out-execution.yml")) is None


# --- find_unsafe_workflows / main ------------------------------------------------------


def test_find_unsafe_workflows_flags_only_held_out_named_unsafe_files(tmp_path: Path) -> None:
    """Verify that find unsafe workflows flags only held out named unsafe files.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
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
    """Verify that find unsafe workflows raises on missing directory.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
    """

    # Scope `pytest.raises(chts.HeldOutTriggerSafetyError, match='does not exist')` here so the
    # expected failure and fixture cleanup stay scoped to this assertion.
    with pytest.raises(chts.HeldOutTriggerSafetyError, match="does not exist"):
        chts.find_unsafe_workflows(tmp_path / "missing")


def test_main_returns_zero_when_no_unsafe_workflows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify that main returns zero when no unsafe workflows.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
        capsys: Pytest capture fixture used to inspect terminal output.
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
    """Verify that main returns one when an unsafe workflow is found.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
        capsys: Pytest capture fixture used to inspect terminal output.
    """

    workflows_dir = tmp_path / "workflows"
    workflows_dir.mkdir()
    (workflows_dir / "held-out-execution.yml").write_text(UNSAFE_BARE_PUSH_YAML, encoding="utf-8")

    exit_code = chts.main(["--workflows-dir", str(workflows_dir)])

    assert exit_code == 1
    assert "Unsafe held-out execution trigger" in capsys.readouterr().err


def test_main_against_the_real_workflows_directory_currently_passes() -> None:
    # No held-out/benchmark-execution workflow exists yet, so this should not fail --
    # it guards against one being added later with an unsafe trigger.
    """Verify that main against the real workflows directory currently passes.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

    assert chts.main([]) == 0
