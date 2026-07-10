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
    """Metrics are byte-identical across repeated runs, and the protected test shard is never opened.

    The fixture's dataset index (see _repository) declares a "test" partition shard
    pointing at a file that doesn't exist on disk ("test-must-not-open.npz"); if
    evaluation ever touched the test partition, this test would fail with a file-not-found
    error rather than succeeding twice with identical output.
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
    """Class 1 with zero support reports metrics as exactly zero_division, not NaN or an error.

    All 4 validation labels are class 0 here, so class 1 has zero true positives,
    zero predicted, and zero support -- every ratio in _divide would be 0/0 without
    the configured zero_division fallback.
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
    """A digest mismatch in any of the three cross-checked files fails before writing metrics.

    Parametrized across all three digest cross-checks evaluation performs (dataset
    index vs. training metadata, model vs. training metadata, and the validation
    shard vs. the dataset index's recorded digest), confirming each is independently
    enforced and that none of them ever leave a partial metrics file on disk.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
        target: Which file to corrupt after fixture setup ("dataset", "model", or "shard").
    """

    paths = _repository(tmp_path)
    # Corrupt exactly the file this parametrization is targeting.
    if target == "dataset":
        paths["index"].write_bytes(paths["index"].read_bytes() + b" ")
    elif target == "model":
        paths["model"].write_bytes(paths["model"].read_bytes() + b" ")
    else:
        paths["shard"].write_bytes(paths["shard"].read_bytes() + b"changed")
    output = _output(tmp_path, "digest")

    # Exactly one of dataset/model/shard was corrupted above, per this parametrization.
    with pytest.raises(EvaluationError, match="digest does not match"):
        evaluate_validation_from_index(
            tmp_path, paths["index"], paths["model"], paths["metadata"], _config(), output
        )

    assert not output.exists()


def test_unknown_validation_label_fails_before_persistence(tmp_path: Path) -> None:
    """A validation label the frozen model was never trained on is rejected, not silently misscored.

    The fixture model only knows classes {0, 1} (see _repository's model.json); label
    2 has no corresponding centroid, so scoring it would be meaningless.
    """

    paths = _repository(tmp_path, labels=np.asarray([0, 2], dtype=np.int64))
    output = _output(tmp_path, "unknown")

    # Label 2 is not among the fixture model's known classes {0, 1}.
    with pytest.raises(EvaluationError, match="unknown to the model"):
        evaluate_validation_from_index(
            tmp_path, paths["index"], paths["model"], paths["metadata"], _config(), output
        )

    assert not output.exists()


def test_malformed_model_with_matching_digest_fails_before_persistence(tmp_path: Path) -> None:
    """A model whose stored digest is valid but whose content is structurally malformed still fails.

    Digest verification alone can't catch a structurally invalid model (the digest
    just proves the bytes weren't tampered with after writing); load_baseline_model's
    own shape checks must still run and reject a projection matrix that doesn't match
    its declared input_features/projection_components.
    """

    paths = _repository(tmp_path, malformed_model=True)
    output = _output(tmp_path, "malformed")

    # The fixture's malformed_model=True writes a 1x1 projection matrix that doesn't
    # match the declared input_features=2/projection_components=2.
    with pytest.raises(EvaluationError, match="projection shape"):
        evaluate_validation_from_index(
            tmp_path, paths["index"], paths["model"], paths["metadata"], _config(), output
        )

    assert not output.exists()


def test_model_and_dataset_feature_width_must_be_compatible(tmp_path: Path) -> None:
    """A dataset index's window_samples that disagrees with the model's input_features is rejected.

    Rewrites the index's window_samples to 3 (the fixture model expects 2 features)
    and re-signs training-metadata's dataset_index digest to match the rewritten
    file, so this specifically isolates the width-compatibility check from a digest
    mismatch that would otherwise mask it.
    """

    paths = _repository(tmp_path)
    index = json.loads(paths["index"].read_text(encoding="utf-8"))
    index["window_samples"] = 3
    paths["index"].write_text(json.dumps(index), encoding="utf-8")
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    metadata["dataset_index"] = _identity(tmp_path, paths["index"])
    paths["metadata"].write_text(json.dumps(metadata), encoding="utf-8")
    output = _output(tmp_path, "incompatible")

    # window_samples was rewritten to 3, but the fixture model expects 2 features.
    with pytest.raises(EvaluationError, match="input width"):
        evaluate_validation_from_index(
            tmp_path, paths["index"], paths["model"], paths["metadata"], _config(), output
        )

    assert not output.exists()


def test_evaluation_config_rejects_test_partition(tmp_path: Path) -> None:
    """A config naming partition = "test" is rejected at load time, before it could run.

    This is the config-level half of the validation-only enforcement: a config that
    names any partition other than "validation" must fail here, never reaching
    _validation_partition's own runtime check.
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
    # partition = "test" above names the protected partition, not "validation".
    with pytest.raises(EvaluationError, match="must be 'validation'"):
        load_evaluation_config(config)


def test_threshold_margin_is_exact_for_known_centroid_distances() -> None:
    """_margin computes the exact nearest-to-second-nearest distance gap for hand-picked values.

    Three rows chosen to exercise distinct cases: a large gap (9.0 vs 1.0 -> 8.0), a
    zero gap (4.0 vs 4.0, a tie -> 0.0), and a small gap (2.25 vs 0.25 -> 2.0).
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
    """_threshold_metrics reports exact coverage counts and macro precision/recall/F1 at 3 thresholds.

    threshold=0.0 covers all 4 windows; threshold=1.0 covers only the 2 windows with
    margin >= 1.0; threshold=2.0 covers zero windows, falling back entirely to the
    configured zero_division value for every metric.
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
    """Each threshold-sweep field validation (name, partition, ordering, finiteness) is enforced.

    One parametrized sweep covering four independent load_threshold_sweep_config
    validation failures: a missing name, a non-"validation" partition, a
    non-strictly-increasing threshold sequence, and a non-finite (inf) threshold value.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
        table: The `[threshold_sweep]` table body, missing or containing exactly one
            invalid field per this parametrization.
        message: The expected error substring for that specific invalid field.
    """

    config = tmp_path / "threshold-sweep.toml"
    config.write_text(
        f"schema_version = 1\n[threshold_sweep]\n{table}\n",
        encoding="utf-8",
    )

    # `table` above contains exactly one invalid field per this parametrization.
    with pytest.raises(EvaluationError, match=message):
        load_threshold_sweep_config(config)


def _config() -> EvaluationConfig:
    """A fixed EvaluationConfig matching the fixture model's estimator and the validation partition.

    Returns:
        A ready-to-use EvaluationConfig with zero_division=0.0.
    """

    return EvaluationConfig(
        1, "fixture-validation", "1.0.0", "random-projection-nearest-centroid", "validation", 0.0
    )


def _evaluate(root: Path, paths: dict[str, Path], run_id: str) -> Path:
    """Run evaluate_validation_from_index against a fixture repository and return the output path.

    Args:
        root: The fake repository root.
        paths: The fixture's index/model/metadata/shard paths, from _repository.
        run_id: A distinct run ID, used to build a fresh output directory.

    Returns:
        Path to the written validation-metrics.json.
    """

    output = _output(root, run_id)
    result = evaluate_validation_from_index(
        root, paths["index"], paths["model"], paths["metadata"], _config(), output
    )
    assert result.window_count == json.loads(output.read_text())["window_count"]
    return output


def _output(root: Path, run_id: str) -> Path:
    """Build (and create) a fresh run-scoped evaluation output directory and metrics path.

    Args:
        root: The fake repository root.
        run_id: A distinct run ID, used to isolate this call's output from other tests.

    Returns:
        The metrics.json path within a freshly created evaluation/ directory.
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
    """Build a fake repository with a dataset index, frozen model, and matching training metadata.

    The dataset index's "test" partition deliberately points at a file that doesn't
    exist on disk ("test-must-not-open.npz"), so any test in this module that
    accidentally reads the test partition fails loudly with a file-not-found error.

    Args:
        root: The fake repository root to populate.
        labels: Override the 4 default validation window labels (used to test
            zero-support and unknown-label cases).
        malformed_model: If True, write a model whose projection matrix shape doesn't
            match its declared input_features/projection_components.

    Returns:
        A dict of the fixture's index/model/metadata/shard paths.
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
    """Build a FileDigest-shaped dict (path/size_bytes/sha256) for one fixture file.

    Args:
        root: The fake repository root, used to make `path` relative.
        path: The file to compute identity evidence for.

    Returns:
        A dict matching FileDigest's JSON shape.
    """

    content = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _counts(values: np.ndarray) -> dict[str, int]:
    """Build a target_value_counts-shaped dict (string label to occurrence count) for fixture labels.

    Args:
        values: The label array to count.

    Returns:
        A dict matching the dataset index's target_value_counts JSON shape.
    """

    return {str(value): int(np.count_nonzero(values == value)) for value in np.unique(values)}
