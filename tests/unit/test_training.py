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
    """Two training runs over the same seeded config produce byte-identical model files, and the
    written metadata records the "train" partition with the correct model digest.

    The fixture dataset index's validation/test partitions both point at a
    shard path ("data/interim/does-not-exist.npz") that is never written to
    disk; if training opened either one, this test would fail with a
    file-not-found error rather than an assertion failure.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
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
    """A training shard rewritten after the dataset index was built fails digest verification,
    and no partial model or metadata file is left behind.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    index = _repository(tmp_path)
    shard = tmp_path / "data" / "interim" / "runs" / "fixture" / "windows" / "100.npz"
    shard.write_bytes(shard.read_bytes() + b"changed")
    output = tmp_path / "artifacts" / "runs" / "output"
    output.mkdir(parents=True)

    # shard was overwritten above, so its SHA-256 no longer matches the dataset index.
    with pytest.raises(TrainingError, match="digest does not match"):
        train_from_index(tmp_path, index, _config(), output / "model.json", output / "meta.json")

    assert not tuple(output.iterdir())


def test_training_config_rejects_unsupported_estimator(tmp_path: Path) -> None:
    """A config naming an estimator other than the one supported baseline is rejected at load time.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
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

    # estimator = "unknown" above is not a recognized estimator name.
    with pytest.raises(TrainingError, match="estimator"):
        load_training_config(path)


def test_training_fails_without_two_classes_before_writing(tmp_path: Path) -> None:
    """A training partition with only one target class fails before any model or metadata is written.

    The nearest-centroid estimator requires at least two classes to compute
    a meaningful centroid separation; the fixture here supplies two windows
    that are both labeled 0.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    index = _repository(tmp_path, labels=np.asarray([0, 0], dtype=np.int64))
    output = tmp_path / "artifacts" / "runs" / "output"
    output.mkdir(parents=True)

    # labels=[0, 0] above supplies only one target class.
    with pytest.raises(TrainingError, match="at least two"):
        train_from_index(tmp_path, index, _config(), output / "model.json", output / "meta.json")

    assert not tuple(output.iterdir())


def _config() -> TrainingConfig:
    """The default fixture training config: seed 11, projected to 3 components.

    Returns:
        A TrainingConfig for the random-projection-nearest-centroid estimator.
    """

    return TrainingConfig(1, "fixture", "1.0.0", "random-projection-nearest-centroid", 11, 3)


def _repository(tmp_path: Path, *, labels: np.ndarray | None = None) -> Path:
    """Build a minimal fixture repository: one training shard and a matching dataset index.

    The validation and test partitions both reference a shard path that is
    never written to disk, so any accidental read of either partition during
    training fails loudly with a file-not-found error instead of silently
    succeeding.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory, used as
            the fixture repository root.
        labels: Target values for the two fixture windows; defaults to
            [0, 1] (two classes) unless a test needs to force a single-class
            failure.

    Returns:
        The path to the written dataset-index.json.
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
