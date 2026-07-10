"""Tests for the disabled-by-default held-out execution config loader."""

from pathlib import Path

import pytest

from ecg_anomaly_detection.held_out_config import HeldOutConfigError, load_held_out_config


def _config_text() -> str:
    """Compute and return config text for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Returns:
        The value produced by the documented operation.
    """

    root = Path(__file__).parents[2]
    return (root / "configs" / "evaluation-heldout.toml").read_text(encoding="utf-8")


def test_committed_held_out_config_loads_successfully() -> None:
    """Verify that committed held out config loads successfully.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

    root = Path(__file__).parents[2]
    config = load_held_out_config(root / "configs" / "evaluation-heldout.toml")

    assert config.schema_version == 1
    assert config.partition == "test"
    assert config.execution_enabled is False
    assert config.requires_recorded_approval is True


def test_config_rejects_wrong_schema_version(tmp_path: Path) -> None:
    """Verify that config rejects wrong schema version.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
    """

    path = tmp_path / "config.toml"
    path.write_text(
        _config_text().replace("schema_version = 1", "schema_version = 2"), encoding="utf-8"
    )

    # Scope `pytest.raises(HeldOutConfigError, match='schema_version = 1')` here so the expected
    # failure and fixture cleanup stay scoped to this assertion.
    with pytest.raises(HeldOutConfigError, match="schema_version = 1"):
        load_held_out_config(path)


def test_config_rejects_missing_execution_table(tmp_path: Path) -> None:
    """Verify that config rejects missing execution table.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
    """

    path = tmp_path / "config.toml"
    path.write_text("schema_version = 1\n", encoding="utf-8")

    # Scope `pytest.raises(HeldOutConfigError, match='\\[execution\\] table')` here so the expected
    # failure and fixture cleanup stay scoped to this assertion.
    with pytest.raises(HeldOutConfigError, match=r"\[execution\] table"):
        load_held_out_config(path)


def test_config_rejects_execution_enabled_true(tmp_path: Path) -> None:
    """Verify that config rejects execution enabled true.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
    """

    path = tmp_path / "config.toml"
    path.write_text(
        _config_text().replace("execution_enabled = false", "execution_enabled = true"),
        encoding="utf-8",
    )

    # Scope `pytest.raises(HeldOutConfigError, match='execution_enabled must be false')` here so the
    # expected failure and fixture cleanup stay scoped to this assertion.
    with pytest.raises(HeldOutConfigError, match="execution_enabled must be false"):
        load_held_out_config(path)


def test_config_rejects_requires_recorded_approval_false(tmp_path: Path) -> None:
    """Verify that config rejects requires recorded approval false.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
    """

    path = tmp_path / "config.toml"
    path.write_text(
        _config_text().replace(
            "requires_recorded_approval = true", "requires_recorded_approval = false"
        ),
        encoding="utf-8",
    )

    # Scope `pytest.raises(HeldOutConfigError, match='requires_recorded_approval must be true')`
    # here so the expected failure and fixture cleanup stay scoped to this assertion.
    with pytest.raises(HeldOutConfigError, match="requires_recorded_approval must be true"):
        load_held_out_config(path)


def test_config_rejects_wrong_partition(tmp_path: Path) -> None:
    """Verify that config rejects wrong partition.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
    """

    path = tmp_path / "config.toml"
    path.write_text(
        _config_text().replace('partition = "test"', 'partition = "validation"'), encoding="utf-8"
    )

    # Scope `pytest.raises(HeldOutConfigError, match="execution.partition must be 'test'")` here so
    # the expected failure and fixture cleanup stay scoped to this assertion.
    with pytest.raises(HeldOutConfigError, match="execution.partition must be 'test'"):
        load_held_out_config(path)


def test_config_rejects_wrong_evaluator(tmp_path: Path) -> None:
    """Verify that config rejects wrong evaluator.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
    """

    path = tmp_path / "config.toml"
    path.write_text(
        _config_text().replace(
            'evaluator = "random-projection-nearest-centroid"', 'evaluator = "some-other-model"'
        ),
        encoding="utf-8",
    )

    # Scope `pytest.raises(HeldOutConfigError, match='execution.evaluator must be')` here so the
    # expected failure and fixture cleanup stay scoped to this assertion.
    with pytest.raises(HeldOutConfigError, match="execution.evaluator must be"):
        load_held_out_config(path)


def test_config_rejects_missing_file(tmp_path: Path) -> None:
    """Verify that config rejects missing file.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
    """

    # Scope `pytest.raises(HeldOutConfigError, match='could not load held-out config')` here so the
    # expected failure and fixture cleanup stay scoped to this assertion.
    with pytest.raises(HeldOutConfigError, match="could not load held-out config"):
        load_held_out_config(tmp_path / "missing.toml")
