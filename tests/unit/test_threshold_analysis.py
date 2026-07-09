"""Tests for the validation-only centroid-distance margin threshold sweep."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ecg_anomaly_detection.evaluation import (
    EvaluationError,
    ThresholdSweepConfig,
    evaluate_threshold_sweep_from_index,
    load_threshold_sweep_config,
)


def test_threshold_sweep_is_deterministic_and_test_shard_is_not_opened(tmp_path: Path) -> None:
    paths = _repository(tmp_path)
    first = _sweep(tmp_path, paths, "first")
    second = _sweep(tmp_path, paths, "second")

    first_bytes = first.read_bytes()
    metrics = json.loads(first_bytes)
    assert first_bytes == second.read_bytes()
    assert metrics["record_ids"] == ["validation-record"]
    assert metrics["window_count"] == 4
    assert all("test" not in item["path"] for item in metrics["validation_shards"])

    thresholds = {item["threshold"]: item for item in metrics["thresholds"]}
    # All four windows are covered at threshold 0.0, so macro precision/recall/F1 match
    # full-set evaluation's macro_average in test_evaluation.py's equivalent fixture.
    assert thresholds[0.0]["covered_window_count"] == 4
    assert thresholds[0.0]["precision"] == pytest.approx((2 / 3 + 1.0) / 2)
    assert thresholds[0.0]["recall"] == pytest.approx(0.75)
    assert thresholds[0.0]["f1"] == pytest.approx((0.8 + 2 / 3) / 2)
    # A very high threshold excludes every window from coverage.
    assert thresholds[1000.0]["covered_window_count"] == 0
    assert thresholds[1000.0]["precision"] == 0.0
    assert thresholds[1000.0]["recall"] == 0.0
    assert thresholds[1000.0]["f1"] == 0.0


def test_threshold_sweep_config_rejects_test_partition(tmp_path: Path) -> None:
    config = tmp_path / "threshold-sweep.toml"
    config.write_text(
        """
schema_version = 1
[threshold_sweep]
name = "bad"
version = "1"
partition = "test"
zero_division = 0.0
thresholds = [0.0, 1.0]
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(EvaluationError, match="must be 'validation'"):
        load_threshold_sweep_config(config)


def test_threshold_sweep_config_rejects_non_increasing_thresholds(tmp_path: Path) -> None:
    config = tmp_path / "threshold-sweep.toml"
    config.write_text(
        """
schema_version = 1
[threshold_sweep]
name = "bad"
version = "1"
partition = "validation"
zero_division = 0.0
thresholds = [1.0, 1.0, 2.0]
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(EvaluationError, match="strictly increasing"):
        load_threshold_sweep_config(config)


def test_threshold_sweep_config_rejects_empty_thresholds(tmp_path: Path) -> None:
    config = tmp_path / "threshold-sweep.toml"
    config.write_text(
        """
schema_version = 1
[threshold_sweep]
name = "bad"
version = "1"
partition = "validation"
zero_division = 0.0
thresholds = []
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(EvaluationError, match="non-empty numeric array"):
        load_threshold_sweep_config(config)


@pytest.mark.parametrize("target", ["dataset", "model", "shard"])
def test_digest_mismatch_fails_before_metrics_persistence(tmp_path: Path, target: str) -> None:
    paths = _repository(tmp_path)
    if target == "dataset":
        paths["index"].write_bytes(paths["index"].read_bytes() + b" ")
    elif target == "model":
        paths["model"].write_bytes(paths["model"].read_bytes() + b" ")
    else:
        paths["shard"].write_bytes(paths["shard"].read_bytes() + b"changed")
    output = _output(tmp_path, "digest")

    with pytest.raises(EvaluationError, match="digest does not match"):
        evaluate_threshold_sweep_from_index(
            tmp_path, paths["index"], paths["model"], paths["metadata"], _config(), output
        )

    assert not output.exists()


def test_malformed_config_fails_before_metrics_persistence(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed-threshold-sweep.toml"
    malformed.write_text(
        """
schema_version = 1
[threshold_sweep]
name = "bad"
version = "1"
partition = "validation"
zero_division = 0.5
thresholds = [0.0, 1.0]
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(EvaluationError, match="zero_division must be 0.0 or 1.0"):
        load_threshold_sweep_config(malformed)


def test_zero_division_behavior_is_explicit_at_a_fully_excluding_threshold(
    tmp_path: Path,
) -> None:
    paths = _repository(tmp_path)
    config = ThresholdSweepConfig(1, "fixture", "1.0.0", "validation", 1.0, (1000.0,))
    output = _output(tmp_path, "zero-division")
    metrics = json.loads(
        evaluate_threshold_sweep_from_index(
            tmp_path, paths["index"], paths["model"], paths["metadata"], config, output
        ).metrics_path.read_text(encoding="utf-8")
    )
    entry = metrics["thresholds"][0]
    assert entry["covered_window_count"] == 0
    assert entry["precision"] == 1.0
    assert entry["recall"] == 1.0
    assert entry["f1"] == 1.0


def _config() -> ThresholdSweepConfig:
    return ThresholdSweepConfig(1, "fixture-sweep", "1.0.0", "validation", 0.0, (0.0, 1000.0))


def _sweep(root: Path, paths: dict[str, Path], run_id: str) -> Path:
    output = _output(root, run_id)
    result = evaluate_threshold_sweep_from_index(
        root, paths["index"], paths["model"], paths["metadata"], _config(), output
    )
    assert result.window_count == json.loads(output.read_text())["window_count"]
    return output


def _output(root: Path, run_id: str) -> Path:
    directory = root / "artifacts" / "runs" / run_id / "evaluation"
    directory.mkdir(parents=True)
    return directory / "threshold-sweep-metrics.json"


def _repository(root: Path) -> dict[str, Path]:
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    shard_dir = root / "data" / "interim" / "runs" / "fixture" / "windows"
    index_dir = root / "data" / "processed" / "runs" / "fixture"
    training_dir = root / "artifacts" / "runs" / "fixture" / "training"
    shard_dir.mkdir(parents=True)
    index_dir.mkdir(parents=True)
    training_dir.mkdir(parents=True)
    values = np.asarray([0, 1, 1, 0], dtype=np.int64)
    windows = np.asarray([[0.0, 0.0], [10.0, 10.0], [1.0, 1.0], [0.0, 0.0]])
    shard = shard_dir / "validation-record.npz"
    np.savez_compressed(
        shard,
        windows=windows,
        target_values=values,
        record_ids=np.asarray(["validation-record"] * len(values)),
    )
    shard_digest = _identity(root, shard)
    validation_descriptor = {
        "record_id": "validation-record",
        "window_count": len(values),
        "target_value_counts": _counts(values),
        "file": shard_digest,
    }
    index = index_dir / "dataset-index.json"
    index.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "window_samples": 2,
                "partitions": {
                    "train": {"shards": []},
                    "validation": {
                        "record_count": 1,
                        "window_count": len(values),
                        "target_value_counts": _counts(values),
                        "shards": [validation_descriptor],
                    },
                    "test": {
                        "record_count": 1,
                        "window_count": 99,
                        "target_value_counts": {"0": 99},
                        "shards": [
                            {
                                "record_id": "test-must-not-open",
                                "file": {"path": "data/interim/test-must-not-open.npz"},
                            }
                        ],
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    model = training_dir / "model.json"
    model.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "estimator": "random-projection-nearest-centroid",
                "training_name": "fixture",
                "training_version": "1.0.0",
                "seed": 1,
                "input_features": 2,
                "projection_components": 2,
                "classes": [0, 1],
                "projection": [[1.0, 0.0], [0.0, 1.0]],
                "centroids": [[0.0, 0.0], [10.0, 10.0]],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    metadata = training_dir / "training-metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "partition": "train",
                "dataset_index": _identity(root, index),
                "model": _identity(root, model),
            }
        ),
        encoding="utf-8",
    )
    return {"index": index, "model": model, "metadata": metadata, "shard": shard}


def _identity(root: Path, path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _counts(values: np.ndarray) -> dict[str, int]:
    return {str(value): int(np.count_nonzero(values == value)) for value in np.unique(values)}
