"""Integration test for the evaluate-threshold-sweep CLI boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ecg_anomaly_detection.cli import main

# A minimal two-threshold, validation-partition sweep config shared by both tests below.
_CONFIG = (
    "schema_version = 1\n\n"
    "[threshold_sweep]\n"
    'name = "cli-fixture-sweep"\n'
    'version = "1.0.0"\n'
    'partition = "validation"\n'
    "zero_division = 0.0\n"
    "thresholds = [0.0, 1000.0]\n"
)


def test_evaluate_threshold_sweep_command_writes_metrics(tmp_path: Path) -> None:
    """`ecg-data evaluate-threshold-sweep` writes a metrics file with the sweep's own name,
    partition, threshold count, and never references a "test"-partition shard path.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    paths = _repository(tmp_path)
    config_path = tmp_path / "threshold-sweep.toml"
    config_path.write_text(_CONFIG, encoding="utf-8")
    output_dir = tmp_path / "artifacts" / "runs" / "cli-fixture" / "evaluation"
    output_dir.mkdir(parents=True)
    output_path = output_dir / "threshold-sweep-metrics.json"

    exit_code = main(
        [
            "evaluate-threshold-sweep",
            "--repository-root",
            str(tmp_path),
            "--dataset-index",
            str(paths["index"]),
            "--model",
            str(paths["model"]),
            "--training-metadata",
            str(paths["metadata"]),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    metrics = json.loads(output_path.read_text(encoding="utf-8"))
    assert metrics["sweep_name"] == "cli-fixture-sweep"
    assert metrics["partition"] == "validation"
    assert len(metrics["thresholds"]) == 2
    assert all("test" not in item["path"] for item in metrics["validation_shards"])


def test_evaluate_threshold_sweep_command_fails_closed_on_digest_mismatch(tmp_path: Path) -> None:
    """A corrupted model file makes `evaluate-threshold-sweep` exit 1 and leave no output file behind.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    paths = _repository(tmp_path)
    paths["model"].write_bytes(paths["model"].read_bytes() + b" ")
    config_path = tmp_path / "threshold-sweep.toml"
    config_path.write_text(_CONFIG, encoding="utf-8")
    output_dir = tmp_path / "artifacts" / "runs" / "cli-fixture" / "evaluation"
    output_dir.mkdir(parents=True)
    output_path = output_dir / "threshold-sweep-metrics.json"

    exit_code = main(
        [
            "evaluate-threshold-sweep",
            "--repository-root",
            str(tmp_path),
            "--dataset-index",
            str(paths["index"]),
            "--model",
            str(paths["model"]),
            "--training-metadata",
            str(paths["metadata"]),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 1
    assert not output_path.exists()


def _repository(root: Path) -> dict[str, Path]:
    """Build a minimal fixture repository: one validation shard, a dataset index, and a model.

    The dataset index's "test" partition intentionally points at a shard
    path that is never written to disk ("test-must-not-open.npz"), so any
    accidental read of the test partition fails loudly with a file-not-found
    error instead of silently succeeding.

    Args:
        root: Pytest's per-test isolated temporary directory, used as the
            fixture repository root.

    Returns:
        A dict with "index", "model", "metadata", and "shard" paths.
    """

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
    """Compute the relative path, byte size, and SHA-256 digest that the digest check compares against.

    Args:
        root: The fixture repository root, used to make the recorded path relative.
        path: The file to fingerprint.

    Returns:
        A dict with "path", "size_bytes", and "sha256" keys matching the
        digest-tracked-input shape the evaluation code expects.
    """

    content = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _counts(values: np.ndarray) -> dict[str, int]:
    """Tally how many times each distinct value in values occurs, keyed by its string form.

    Args:
        values: The target-value array to tally.

    Returns:
        A dict mapping each distinct value's string representation to its count.
    """

    return {str(value): int(np.count_nonzero(values == value)) for value in np.unique(values)}
