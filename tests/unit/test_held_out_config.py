"""Tests for the disabled-by-default held-out execution config loader."""

from pathlib import Path

import pytest

from ecg_anomaly_detection.held_out_config import HeldOutConfigError, load_held_out_config


def _config_text() -> str:
    """Read the repository's real, committed configs/evaluation-heldout.toml as text.

    Tests mutate this text (via string replacement) to construct otherwise-valid
    configs with exactly one field violating the held-out governance invariants.

    Returns:
        The committed held-out config file's full text.
    """

    root = Path(__file__).parents[2]
    return (root / "configs" / "evaluation-heldout.toml").read_text(encoding="utf-8")


def test_committed_held_out_config_loads_successfully() -> None:
    """The actual committed configs/evaluation-heldout.toml loads and is disabled by default.

    This is the config governance.py enforces before any code path is allowed
    near the protected test partition; execution_enabled=False and
    requires_recorded_approval=True are the two flags that keep it inert.
    """

    root = Path(__file__).parents[2]
    config = load_held_out_config(root / "configs" / "evaluation-heldout.toml")

    assert config.schema_version == 1
    assert config.partition == "test"
    assert config.execution_enabled is False
    assert config.requires_recorded_approval is True


def test_config_rejects_wrong_schema_version(tmp_path: Path) -> None:
    """A config declaring schema_version = 2 is rejected; only version 1 is understood.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    path = tmp_path / "config.toml"
    path.write_text(
        _config_text().replace("schema_version = 1", "schema_version = 2"), encoding="utf-8"
    )

    # schema_version was rewritten to 2 above; only 1 is a supported schema.
    with pytest.raises(HeldOutConfigError, match="schema_version = 1"):
        load_held_out_config(path)


def test_config_rejects_missing_execution_table(tmp_path: Path) -> None:
    """A config with only schema_version and no [execution] table at all is rejected.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    path = tmp_path / "config.toml"
    path.write_text("schema_version = 1\n", encoding="utf-8")

    # This fixture's TOML has no [execution] table whatsoever.
    with pytest.raises(HeldOutConfigError, match=r"\[execution\] table"):
        load_held_out_config(path)


def test_config_rejects_execution_enabled_true(tmp_path: Path) -> None:
    """A config that flips execution_enabled to true is rejected outright.

    This is the primary governance gate: no on-disk config is permitted to
    enable held-out execution, regardless of any other field's value.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    path = tmp_path / "config.toml"
    path.write_text(
        _config_text().replace("execution_enabled = false", "execution_enabled = true"),
        encoding="utf-8",
    )

    # execution_enabled was rewritten to true above, which is never permitted.
    with pytest.raises(HeldOutConfigError, match="execution_enabled must be false"):
        load_held_out_config(path)


def test_config_rejects_requires_recorded_approval_false(tmp_path: Path) -> None:
    """A config that turns off the recorded-approval requirement is rejected.

    This flag exists so enabling held-out execution (in some future,
    separately governed change) can never happen without a recorded human
    approval step; a config disabling that requirement defeats the gate's
    purpose and must fail regardless of execution_enabled's own value.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    path = tmp_path / "config.toml"
    path.write_text(
        _config_text().replace(
            "requires_recorded_approval = true", "requires_recorded_approval = false"
        ),
        encoding="utf-8",
    )

    # requires_recorded_approval was rewritten to false above, which is never permitted.
    with pytest.raises(HeldOutConfigError, match="requires_recorded_approval must be true"):
        load_held_out_config(path)


def test_config_rejects_wrong_partition(tmp_path: Path) -> None:
    """A config naming any partition other than "test" is rejected.

    This governance file exists specifically to gate access to the held-out
    test partition; a config pointing anywhere else has no reason to exist
    under this loader.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    path = tmp_path / "config.toml"
    path.write_text(
        _config_text().replace('partition = "test"', 'partition = "validation"'), encoding="utf-8"
    )

    # partition was rewritten to "validation" above, which this loader does not accept.
    with pytest.raises(HeldOutConfigError, match="execution.partition must be 'test'"):
        load_held_out_config(path)


def test_config_rejects_wrong_evaluator(tmp_path: Path) -> None:
    """A config naming an evaluator other than the pinned baseline estimator is rejected.

    Pinning the evaluator name prevents a future config edit from silently
    swapping in an untested or unapproved model for held-out execution.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    path = tmp_path / "config.toml"
    path.write_text(
        _config_text().replace(
            'evaluator = "random-projection-nearest-centroid"', 'evaluator = "some-other-model"'
        ),
        encoding="utf-8",
    )

    # evaluator was rewritten to "some-other-model" above, which isn't the pinned baseline.
    with pytest.raises(HeldOutConfigError, match="execution.evaluator must be"):
        load_held_out_config(path)


def test_config_rejects_missing_file(tmp_path: Path) -> None:
    """Pointing the loader at a path with no file produces a HeldOutConfigError, not a raw OSError.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    # tmp_path / "missing.toml" was never created.
    with pytest.raises(HeldOutConfigError, match="could not load held-out config"):
        load_held_out_config(tmp_path / "missing.toml")
