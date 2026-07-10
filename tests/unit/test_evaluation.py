"""Tests for deterministic validation-only baseline evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ecg_anomaly_detection.evaluation import (
    EvaluationConfig,
    EvaluationError,
    _margin,
    _threshold_metrics,
    evaluate_validation_from_index,
    load_evaluation_config,
    load_threshold_sweep_config,
)


def test_exact_metrics_are_deterministic_and_test_shard_is_not_opened(tmp_path: Path) -> None:
    """Verify that exact metrics are deterministic and test shard is not opened.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
    """

    paths = _repository(tmp_path)
    first = _evaluate(tmp_path, paths, "first")
    second = _evaluate(tmp_path, paths, "second")

    first_bytes = first.read_bytes()
    metrics = json.loads(first_bytes)
    assert first_bytes == second.read_bytes()
    assert metrics["confusion_matrix"] == [[2, 0], [1, 1]]
    assert metrics["accuracy"] == 0.75
    assert metrics["per_class"] == {
        "0": {"f1": 0.8, "precision": 2 / 3, "recall": 1.0, "support": 2},
        "1": {"f1": 2 / 3, "precision": 1.0, "recall": 0.5, "support": 2},
    }
    assert metrics["macro_average"] == {
        "f1": (0.8 + 2 / 3) / 2,
        "precision": (2 / 3 + 1.0) / 2,
        "recall": 0.75,
        "support": 4,
    }
    assert metrics["record_ids"] == ["validation-record"]
    assert all("test" not in item["path"] for item in metrics["validation_shards"])


def test_zero_division_behavior_is_explicit(tmp_path: Path) -> None:
    """Verify that zero division behavior is explicit.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
    """

    paths = _repository(tmp_path, labels=np.asarray([0, 0], dtype=np.int64))
    metrics = json.loads(_evaluate(tmp_path, paths, "zero").read_text(encoding="utf-8"))
    assert metrics["zero_division"] == 0.0
    assert metrics["per_class"]["1"] == {
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "support": 0,
    }


@pytest.mark.parametrize("target", ["dataset", "model", "shard"])
def test_digest_mismatch_fails_before_metrics_persistence(tmp_path: Path, target: str) -> None:
    """Verify that digest mismatch fails before metrics persistence.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
        target: The target value supplied by the caller or surrounding test fixture.
    """

    paths = _repository(tmp_path)
    # Exercise the `target == 'dataset'` branch so this regression documents every expected outcome.
    if target == "dataset":
        paths["index"].write_bytes(paths["index"].read_bytes() + b" ")
    elif target == "model":
        paths["model"].write_bytes(paths["model"].read_bytes() + b" ")
    else:
        paths["shard"].write_bytes(paths["shard"].read_bytes() + b"changed")
    output = _output(tmp_path, "digest")

    # Scope `pytest.raises(EvaluationError, match='digest does not match')` here so the expected
    # failure and fixture cleanup stay scoped to this assertion.
    with pytest.raises(EvaluationError, match="digest does not match"):
        evaluate_validation_from_index(
            tmp_path, paths["index"], paths["model"], paths["metadata"], _config(), output
        )

    assert not output.exists()


def test_unknown_validation_label_fails_before_persistence(tmp_path: Path) -> None:
    """Verify that unknown validation label fails before persistence.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
    """

    paths = _repository(tmp_path, labels=np.asarray([0, 2], dtype=np.int64))
    output = _output(tmp_path, "unknown")

    # Scope `pytest.raises(EvaluationError, match='unknown to the model')` here so the expected
    # failure and fixture cleanup stay scoped to this assertion.
    with pytest.raises(EvaluationError, match="unknown to the model"):
        evaluate_validation_from_index(
            tmp_path, paths["index"], paths["model"], paths["metadata"], _config(), output
        )

    assert not output.exists()


def test_malformed_model_with_matching_digest_fails_before_persistence(tmp_path: Path) -> None:
    """Verify that malformed model with matching digest fails before persistence.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
    """

    paths = _repository(tmp_path, malformed_model=True)
    output = _output(tmp_path, "malformed")

    # Scope `pytest.raises(EvaluationError, match='projection shape')` here so the expected failure
    # and fixture cleanup stay scoped to this assertion.
    with pytest.raises(EvaluationError, match="projection shape"):
        evaluate_validation_from_index(
            tmp_path, paths["index"], paths["model"], paths["metadata"], _config(), output
        )

    assert not output.exists()


def test_model_and_dataset_feature_width_must_be_compatible(tmp_path: Path) -> None:
    """Verify that model and dataset feature width must be compatible.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
    """

    paths = _repository(tmp_path)
    index = json.loads(paths["index"].read_text(encoding="utf-8"))
    index["window_samples"] = 3
    paths["index"].write_text(json.dumps(index), encoding="utf-8")
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    metadata["dataset_index"] = _identity(tmp_path, paths["index"])
    paths["metadata"].write_text(json.dumps(metadata), encoding="utf-8")
    output = _output(tmp_path, "incompatible")

    # Scope `pytest.raises(EvaluationError, match='input width')` here so the expected failure and
    # fixture cleanup stay scoped to this assertion.
    with pytest.raises(EvaluationError, match="input width"):
        evaluate_validation_from_index(
            tmp_path, paths["index"], paths["model"], paths["metadata"], _config(), output
        )

    assert not output.exists()


def test_evaluation_config_rejects_test_partition(tmp_path: Path) -> None:
    """Verify that evaluation config rejects test partition.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
    """

    config = tmp_path / "evaluation.toml"
    config.write_text(
        """
schema_version = 1
[evaluation]
name = "bad"
version = "1"
evaluator = "random-projection-nearest-centroid"
partition = "test"
zero_division = 0.0
""".strip(),
        encoding="utf-8",
    )
    # Scope `pytest.raises(EvaluationError, match="must be 'validation'")` here so the expected
    # failure and fixture cleanup stay scoped to this assertion.
    with pytest.raises(EvaluationError, match="must be 'validation'"):
        load_evaluation_config(config)


def test_threshold_margin_is_exact_for_known_centroid_distances() -> None:
    """Verify that threshold margin is exact for known centroid distances.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

    distances = np.asarray(
        [
            [9.0, 1.0],
            [4.0, 4.0],
            [0.25, 2.25],
        ],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(_margin(distances), np.asarray([8.0, 0.0, 2.0]))


def test_threshold_metrics_report_exact_coverage_and_macro_scores() -> None:
    """Verify that threshold metrics report exact coverage and macro scores.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

    classes = (0, 1)
    margins = np.asarray([0.0, 0.5, 1.0, 1.5], dtype=np.float64)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    predictions = np.asarray([0, 1, 1, 0], dtype=np.int64)

    all_windows = _threshold_metrics(0.0, classes, margins, labels, predictions, 0.0)
    assert all_windows.covered_window_count == 4
    assert all_windows.precision == pytest.approx(0.5)
    assert all_windows.recall == pytest.approx(0.5)
    assert all_windows.f1 == pytest.approx(0.5)

    high_margin = _threshold_metrics(1.0, classes, margins, labels, predictions, 0.0)
    assert high_margin.covered_window_count == 2
    assert high_margin.precision == pytest.approx(0.5)
    assert high_margin.recall == pytest.approx(0.25)
    assert high_margin.f1 == pytest.approx(1 / 3)

    no_windows = _threshold_metrics(2.0, classes, margins, labels, predictions, 1.0)
    assert no_windows.covered_window_count == 0
    assert no_windows.precision == 1.0
    assert no_windows.recall == 1.0
    assert no_windows.f1 == 1.0


@pytest.mark.parametrize(
    ("table", "message"),
    [
        (
            'version = "1"\npartition = "validation"\nzero_division = 0.0\nthresholds = [0.0]',
            "threshold_sweep.name must be a non-empty string",
        ),
        (
            'name = "bad"\nversion = "1"\npartition = "test"\n'
            "zero_division = 0.0\nthresholds = [0.0]",
            "threshold_sweep.partition must be 'validation'",
        ),
        (
            'name = "bad"\nversion = "1"\npartition = "validation"\n'
            "zero_division = 0.0\nthresholds = [1.0, 0.0]",
            "threshold_sweep.thresholds must be strictly increasing",
        ),
        (
            'name = "bad"\nversion = "1"\npartition = "validation"\n'
            "zero_division = 0.0\nthresholds = [0.0, inf]",
            "threshold_sweep.thresholds must be finite",
        ),
    ],
)
def test_threshold_sweep_config_rejects_invalid_values(
    tmp_path: Path, table: str, message: str
) -> None:
    """Verify that threshold sweep config rejects invalid values.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
        table: The table value supplied by the caller or surrounding test fixture.
        message: The message value supplied by the caller or surrounding test fixture.
    """

    config = tmp_path / "threshold-sweep.toml"
    config.write_text(
        f"schema_version = 1\n[threshold_sweep]\n{table}\n",
        encoding="utf-8",
    )

    # Scope `pytest.raises(EvaluationError, match=message)` here so the expected failure and fixture
    # cleanup stay scoped to this assertion.
    with pytest.raises(EvaluationError, match=message):
        load_threshold_sweep_config(config)


def _config() -> EvaluationConfig:
    """Compute and return config for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Returns:
        The value produced by the documented operation.
    """

    return EvaluationConfig(
        1, "fixture-validation", "1.0.0", "random-projection-nearest-centroid", "validation", 0.0
    )


def _evaluate(root: Path, paths: dict[str, Path], run_id: str) -> Path:
    """Compute and return evaluate for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        root: Repository root used to enforce path and trust boundaries.
        paths: The paths value supplied by the caller or surrounding test fixture.
        run_id: The run id value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    output = _output(root, run_id)
    result = evaluate_validation_from_index(
        root, paths["index"], paths["model"], paths["metadata"], _config(), output
    )
    assert result.window_count == json.loads(output.read_text())["window_count"]
    return output


def _output(root: Path, run_id: str) -> Path:
    """Resolve and validate output for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        root: Repository root used to enforce path and trust boundaries.
        run_id: The run id value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    directory = root / "artifacts" / "runs" / run_id / "evaluation"
    directory.mkdir(parents=True)
    return directory / "validation-metrics.json"


def _repository(
    root: Path,
    *,
    labels: np.ndarray | None = None,
    malformed_model: bool = False,
) -> dict[str, Path]:
    """Compute and return repository for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        root: Repository root used to enforce path and trust boundaries.
        labels: Target labels retained for validation or metric calculation.
        malformed_model: The malformed model value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    shard_dir = root / "data" / "interim" / "runs" / "fixture" / "windows"
    index_dir = root / "data" / "processed" / "runs" / "fixture"
    training_dir = root / "artifacts" / "runs" / "fixture" / "training"
    shard_dir.mkdir(parents=True)
    index_dir.mkdir(parents=True)
    training_dir.mkdir(parents=True)
    values = labels if labels is not None else np.asarray([0, 1, 1, 0], dtype=np.int64)
    windows = np.asarray([[0.0, 0.0], [10.0, 10.0], [1.0, 1.0], [0.0, 0.0]])[: len(values)]
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
                "projection": [[1.0]] if malformed_model else [[1.0, 0.0], [0.0, 1.0]],
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
    """Compute and return identity for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        root: Repository root used to enforce path and trust boundaries.
        path: Filesystem path identifying the input or output under review.

    Returns:
        The value produced by the documented operation.
    """

    content = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _counts(values: np.ndarray) -> dict[str, int]:
    """Compute and return counts for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        values: Structured values to validate, transform, or serialize.

    Returns:
        The value produced by the documented operation.
    """

    return {str(value): int(np.count_nonzero(values == value)) for value in np.unique(values)}
