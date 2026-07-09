"""Tests for the disabled-by-default held-out execution config loader."""

from pathlib import Path

import pytest

from ecg_anomaly_detection.held_out_config import HeldOutConfigError, load_held_out_config


def _config_text() -> str:
    root = Path(__file__).parents[2]
    return (root / "configs" / "evaluation-heldout.toml").read_text(encoding="utf-8")


def test_committed_held_out_config_loads_successfully() -> None:
    root = Path(__file__).parents[2]
    config = load_held_out_config(root / "configs" / "evaluation-heldout.toml")

    assert config.schema_version == 1
    assert config.partition == "test"
    assert config.execution_enabled is False
    assert config.requires_recorded_approval is True


def test_config_rejects_wrong_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        _config_text().replace("schema_version = 1", "schema_version = 2"), encoding="utf-8"
    )

    with pytest.raises(HeldOutConfigError, match="schema_version = 1"):
        load_held_out_config(path)


def test_config_rejects_missing_execution_table(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("schema_version = 1\n", encoding="utf-8")

    with pytest.raises(HeldOutConfigError, match=r"\[execution\] table"):
        load_held_out_config(path)


def test_config_rejects_execution_enabled_true(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        _config_text().replace("execution_enabled = false", "execution_enabled = true"),
        encoding="utf-8",
    )

    with pytest.raises(HeldOutConfigError, match="execution_enabled must be false"):
        load_held_out_config(path)


def test_config_rejects_requires_recorded_approval_false(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        _config_text().replace(
            "requires_recorded_approval = true", "requires_recorded_approval = false"
        ),
        encoding="utf-8",
    )

    with pytest.raises(HeldOutConfigError, match="requires_recorded_approval must be true"):
        load_held_out_config(path)


def test_config_rejects_wrong_partition(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        _config_text().replace('partition = "test"', 'partition = "validation"'), encoding="utf-8"
    )

    with pytest.raises(HeldOutConfigError, match="execution.partition must be 'test'"):
        load_held_out_config(path)


def test_config_rejects_wrong_evaluator(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        _config_text().replace(
            'evaluator = "random-projection-nearest-centroid"', 'evaluator = "some-other-model"'
        ),
        encoding="utf-8",
    )

    with pytest.raises(HeldOutConfigError, match="execution.evaluator must be"):
        load_held_out_config(path)


def test_config_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(HeldOutConfigError, match="could not load held-out config"):
        load_held_out_config(tmp_path / "missing.toml")
