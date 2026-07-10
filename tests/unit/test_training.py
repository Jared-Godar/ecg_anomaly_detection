"""Tests for deterministic, training-only baseline fitting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ecg_anomaly_detection.training import (
    TrainingConfig,
    TrainingError,
    load_training_config,
    train_from_index,
)


def test_training_is_deterministic_and_does_not_open_held_out_shards(tmp_path: Path) -> None:
    """Verify that training is deterministic and does not open held out shards.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
    """

    index = _repository(tmp_path)
    config = _config()
    first_dir = tmp_path / "artifacts" / "runs" / "first"
    second_dir = tmp_path / "artifacts" / "runs" / "second"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)

    first = train_from_index(
        tmp_path, index, config, first_dir / "model.json", first_dir / "meta.json"
    )
    second = train_from_index(
        tmp_path, index, config, second_dir / "model.json", second_dir / "meta.json"
    )

    assert first.model_path.read_bytes() == second.model_path.read_bytes()
    metadata = json.loads(first.metadata_path.read_text(encoding="utf-8"))
    assert metadata["partition"] == "train"
    assert metadata["record_ids"] == ["100"]
    assert metadata["model"]["sha256"] == hashlib.sha256(first.model_path.read_bytes()).hexdigest()


def test_training_rejects_modified_training_shard(tmp_path: Path) -> None:
    """Verify that training rejects modified training shard.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
    """

    index = _repository(tmp_path)
    shard = tmp_path / "data" / "interim" / "runs" / "fixture" / "windows" / "100.npz"
    shard.write_bytes(shard.read_bytes() + b"changed")
    output = tmp_path / "artifacts" / "runs" / "output"
    output.mkdir(parents=True)

    # Scope `pytest.raises(TrainingError, match='digest does not match')` here so the expected
    # failure and fixture cleanup stay scoped to this assertion.
    with pytest.raises(TrainingError, match="digest does not match"):
        train_from_index(tmp_path, index, _config(), output / "model.json", output / "meta.json")

    assert not tuple(output.iterdir())


def test_training_config_rejects_unsupported_estimator(tmp_path: Path) -> None:
    """Verify that training config rejects unsupported estimator.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
    """

    path = tmp_path / "training.toml"
    path.write_text(
        """
schema_version = 1
[training]
name = "bad"
version = "1"
estimator = "unknown"
seed = 1
projection_components = 2
""".strip(),
        encoding="utf-8",
    )

    # Scope `pytest.raises(TrainingError, match='estimator')` here so the expected failure and
    # fixture cleanup stay scoped to this assertion.
    with pytest.raises(TrainingError, match="estimator"):
        load_training_config(path)


def test_training_fails_without_two_classes_before_writing(tmp_path: Path) -> None:
    """Verify that training fails without two classes before writing.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
    """

    index = _repository(tmp_path, labels=np.asarray([0, 0], dtype=np.int64))
    output = tmp_path / "artifacts" / "runs" / "output"
    output.mkdir(parents=True)

    # Scope `pytest.raises(TrainingError, match='at least two')` here so the expected failure and
    # fixture cleanup stay scoped to this assertion.
    with pytest.raises(TrainingError, match="at least two"):
        train_from_index(tmp_path, index, _config(), output / "model.json", output / "meta.json")

    assert not tuple(output.iterdir())


def _config() -> TrainingConfig:
    """Compute and return config for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Returns:
        The value produced by the documented operation.
    """

    return TrainingConfig(1, "fixture", "1.0.0", "random-projection-nearest-centroid", 11, 3)


def _repository(tmp_path: Path, *, labels: np.ndarray | None = None) -> Path:
    """Compute and return repository for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
        labels: Target labels retained for validation or metric calculation.

    Returns:
        The value produced by the documented operation.
    """

    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    windows_dir = tmp_path / "data" / "interim" / "runs" / "fixture" / "windows"
    index_dir = tmp_path / "data" / "processed" / "runs" / "fixture"
    windows_dir.mkdir(parents=True)
    index_dir.mkdir(parents=True)
    values = labels if labels is not None else np.asarray([0, 1], dtype=np.int64)
    shard = windows_dir / "100.npz"
    np.savez_compressed(
        shard,
        windows=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        target_values=values,
        record_ids=np.asarray(["100", "100"]),
    )
    content = shard.read_bytes()
    descriptor = {
        "record_id": "100",
        "window_count": 2,
        "target_value_counts": {
            str(value): int(np.count_nonzero(values == value)) for value in np.unique(values)
        },
        "file": {
            "path": shard.relative_to(tmp_path).as_posix(),
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        },
    }
    empty = {
        "record_count": 1,
        "window_count": 1,
        "target_value_counts": {"0": 1},
        "shards": [{"record_id": "held-out", "file": {"path": "data/interim/does-not-exist.npz"}}],
    }
    index = index_dir / "dataset-index.json"
    index.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "partitions": {
                    "train": {
                        "record_count": 1,
                        "window_count": 2,
                        "target_value_counts": descriptor["target_value_counts"],
                        "shards": [descriptor],
                    },
                    "validation": empty,
                    "test": empty,
                },
            }
        ),
        encoding="utf-8",
    )
    return index
