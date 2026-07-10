"""Deterministic validation-only evaluation of a frozen baseline model."""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np

from ecg_anomaly_detection.training import (
    SUPPORTED_ESTIMATOR,
    BaselineModel,
    FileDigest,
    TrainingError,
    load_baseline_model,
)

# Centralize BUFFER_SIZE so every caller shares the same documented invariant.
BUFFER_SIZE = 1024 * 1024
# Centralize SUPPORTED_PARTITION so every caller shares the same documented invariant.
SUPPORTED_PARTITION = "validation"


class EvaluationError(ValueError):
    """Raised when validation evaluation cannot satisfy its isolation contract."""


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Define the versioned evaluator, partition, and undefined-metric policy.

    The supported loader restricts these fields to the frozen baseline evaluator and the
    validation partition, preventing configuration from silently opening held-out data.
    """

    schema_version: int
    name: str
    version: str
    evaluator: str
    partition: str
    zero_division: float


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    """Store precision, recall, F1, and support for one configured target class.

    Keeping support beside rate metrics makes class imbalance and undefined values visible
    instead of allowing aggregate scores to hide the underlying sample count.
    """

    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True, slots=True)
class ValidationMetrics:
    """Capture validation-only scores together with complete artifact lineage.

    Dataset-index, shard, and model digests bind every reported metric to the exact frozen
    inputs that produced it; this structure never represents a protected-test benchmark.
    """

    schema_version: int
    evaluation_name: str
    evaluation_version: str
    evaluator: str
    partition: str
    zero_division: float
    record_ids: tuple[str, ...]
    record_count: int
    window_count: int
    classes: tuple[int, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]
    accuracy: float
    per_class: dict[str, ClassMetrics]
    macro_average: ClassMetrics
    dataset_index: FileDigest
    validation_shards: tuple[FileDigest, ...]
    model: FileDigest

    def to_json(self) -> str:
        """Serialize this structured record as deterministic JSON.

        The helper isolates this step so its assumptions, outputs, and failure behavior remain
        reviewable.

        Returns:
            The value produced by the documented operation.
        """

        return json.dumps(asdict(self), indent=2, sort_keys=True, allow_nan=False) + "\n"


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Report where validation metrics were written and how much data was scored.

    The lightweight return value lets orchestration report completion without reparsing the
    persisted metrics artifact.
    """

    metrics_path: Path
    window_count: int
    record_count: int


@dataclass(frozen=True, slots=True)
class ThresholdSweepConfig:
    """Define validation-only centroid-margin thresholds and undefined-metric handling.

    Thresholds are strictly increasing finite raw-distance gaps, not probabilities or
    calibrated confidence values.
    """

    schema_version: int
    name: str
    version: str
    partition: str
    zero_division: float
    thresholds: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ThresholdMetrics:
    """Store coverage and macro metrics for one centroid-distance margin threshold.

    Coverage is reported explicitly because stricter thresholds may improve visible metrics
    while excluding an increasing share of validation windows.
    """

    threshold: float
    covered_window_count: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True, slots=True)
class ThresholdSweepMetrics:
    """Capture a complete validation-only threshold sweep and its frozen lineage.

    This artifact is intentionally separate from ordinary validation metrics so downstream
    readers cannot mistake selective-coverage analysis for the baseline score.
    """

    schema_version: int
    sweep_name: str
    sweep_version: str
    partition: str
    zero_division: float
    record_ids: tuple[str, ...]
    record_count: int
    window_count: int
    thresholds: tuple[ThresholdMetrics, ...]
    dataset_index: FileDigest
    validation_shards: tuple[FileDigest, ...]
    model: FileDigest

    def to_json(self) -> str:
        """Serialize this structured record as deterministic JSON.

        The helper isolates this step so its assumptions, outputs, and failure behavior remain
        reviewable.

        Returns:
            The value produced by the documented operation.
        """

        return json.dumps(asdict(self), indent=2, sort_keys=True, allow_nan=False) + "\n"


@dataclass(frozen=True, slots=True)
class ThresholdSweepResult:
    """Report the threshold-sweep artifact location and evaluated validation volume.

    The result carries operational counts only; substantive metrics remain in the persisted,
    digest-linked artifact.
    """

    metrics_path: Path
    window_count: int
    record_count: int


def load_evaluation_config(path: Path) -> EvaluationConfig:
    """Load evaluation config according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        path: Filesystem path identifying the input or output under review.

    Returns:
        The value produced by the documented operation.
    """

    # Attempt this boundary operation here so (OSError, tomllib.TOMLDecodeError) can be translated
    # or cleaned up under the repository contract.
    try:
        # Scope `path.open('rb')` here so resource cleanup occurs on both success and failure paths.
        with path.open("rb") as source:
            document = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise EvaluationError(f"could not load evaluation config {path}: {error}") from error
    values = document.get("evaluation")
    # Evaluate `document.get('schema_version') != 1 or not isinstance(values, dict)` explicitly so
    # invalid or alternate states follow the documented contract.
    if document.get("schema_version") != 1 or not isinstance(values, dict):
        raise EvaluationError(
            "evaluation config must use schema_version = 1 and an [evaluation] table"
        )
    config = EvaluationConfig(
        1,
        _string(values, "name"),
        _string(values, "version"),
        _string(values, "evaluator"),
        _string(values, "partition"),
        _zero_division(values.get("zero_division")),
    )
    # Evaluate `config.evaluator != SUPPORTED_ESTIMATOR` explicitly so invalid or alternate states
    # follow the documented contract.
    if config.evaluator != SUPPORTED_ESTIMATOR:
        raise EvaluationError(f"evaluation.evaluator must be {SUPPORTED_ESTIMATOR!r}")
    # Evaluate `config.partition != SUPPORTED_PARTITION` explicitly so invalid or alternate states
    # follow the documented contract.
    if config.partition != SUPPORTED_PARTITION:
        raise EvaluationError("evaluation.partition must be 'validation'")
    return config


def load_threshold_sweep_config(path: Path) -> ThresholdSweepConfig:
    """Load threshold sweep config according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        path: Filesystem path identifying the input or output under review.

    Returns:
        The value produced by the documented operation.
    """

    # Attempt this boundary operation here so (OSError, tomllib.TOMLDecodeError) can be translated
    # or cleaned up under the repository contract.
    try:
        # Scope `path.open('rb')` here so resource cleanup occurs on both success and failure paths.
        with path.open("rb") as source:
            document = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise EvaluationError(f"could not load threshold sweep config {path}: {error}") from error
    values = document.get("threshold_sweep")
    # Evaluate `document.get('schema_version') != 1 or not isinstance(values, dict)` explicitly so
    # invalid or alternate states follow the documented contract.
    if document.get("schema_version") != 1 or not isinstance(values, dict):
        raise EvaluationError(
            "threshold sweep config must use schema_version = 1 and a [threshold_sweep] table"
        )
    config = ThresholdSweepConfig(
        1,
        _string(values, "name", table="threshold_sweep"),
        _string(values, "version", table="threshold_sweep"),
        _string(values, "partition", table="threshold_sweep"),
        _zero_division(values.get("zero_division"), table="threshold_sweep"),
        _thresholds(values.get("thresholds")),
    )
    # Evaluate `config.partition != SUPPORTED_PARTITION` explicitly so invalid or alternate states
    # follow the documented contract.
    if config.partition != SUPPORTED_PARTITION:
        raise EvaluationError("threshold_sweep.partition must be 'validation'")
    return config


def evaluate_validation_from_index(
    repository_root: Path,
    dataset_index_path: Path,
    model_path: Path,
    training_metadata_path: Path,
    config: EvaluationConfig,
    metrics_path: Path,
) -> EvaluationResult:
    """Score only indexed validation shards and persist metrics after all checks pass."""
    root = repository_root.resolve()
    output = _output_path(root, metrics_path)
    data = _load_validation_data(root, dataset_index_path, model_path, training_metadata_path)
    predictions = _predict(data.model, data.features)
    metrics = _build_metrics(
        config,
        data.model,
        data.records,
        data.labels,
        predictions,
        data.index_digest,
        data.shard_digests,
        data.model_digest,
    )
    _write_new(output, metrics.to_json())
    return EvaluationResult(output, len(data.labels), len(data.records))


def evaluate_threshold_sweep_from_index(
    repository_root: Path,
    dataset_index_path: Path,
    model_path: Path,
    training_metadata_path: Path,
    config: ThresholdSweepConfig,
    metrics_path: Path,
) -> ThresholdSweepResult:
    """Report coverage/precision/recall/F1 at each configured centroid-distance margin
    threshold, over only indexed validation shards.

    The margin is the raw squared-distance gap (in projected feature space) between a
    window's nearest and second-nearest class centroid. It is not a probability, is not
    calibrated, and does not support ROC/AUC analysis.
    """
    root = repository_root.resolve()
    output = _output_path(root, metrics_path, "threshold-sweep-metrics.json")
    data = _load_validation_data(root, dataset_index_path, model_path, training_metadata_path)
    distances = _distances(data.model, data.features)
    predictions = np.asarray(data.model.classes, dtype=np.int64)[np.argmin(distances, axis=1)]
    margins = _margin(distances)
    thresholds = tuple(
        _threshold_metrics(
            threshold, data.model.classes, margins, data.labels, predictions, config.zero_division
        )
        for threshold in config.thresholds
    )
    metrics = ThresholdSweepMetrics(
        1,
        config.name,
        config.version,
        config.partition,
        config.zero_division,
        tuple(data.records),
        len(data.records),
        len(data.labels),
        thresholds,
        data.index_digest,
        tuple(data.shard_digests),
        data.model_digest,
    )
    _write_new(output, metrics.to_json())
    return ThresholdSweepResult(output, len(data.labels), len(data.records))


def _threshold_metrics(
    threshold: float,
    classes: tuple[int, ...],
    margins: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    predictions: np.ndarray[Any, Any],
    zero_division: float,
) -> ThresholdMetrics:
    """Macro-averaged precision/recall/F1, restricted to windows covered at this threshold."""
    covered = margins >= threshold
    covered_count = int(np.count_nonzero(covered))
    covered_labels = labels[covered]
    covered_predictions = predictions[covered]
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    # Iterate over `classes` one item at a time so ordering, validation, and failure attribution
    # remain explicit.
    for value in classes:
        true_positive = int(
            np.count_nonzero((covered_labels == value) & (covered_predictions == value))
        )
        predicted = int(np.count_nonzero(covered_predictions == value))
        support = int(np.count_nonzero(covered_labels == value))
        precision = _divide(true_positive, predicted, zero_division)
        recall = _divide(true_positive, support, zero_division)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(_divide(2.0 * precision * recall, precision + recall, zero_division))
    return ThresholdMetrics(
        threshold,
        covered_count,
        float(np.mean(precisions)),
        float(np.mean(recalls)),
        float(np.mean(f1s)),
    )


@dataclass(frozen=True, slots=True)
class _ValidationData:
    """Bundle verified validation arrays, frozen model state, and input digests.

    The loader constructs this object only after proving all paths, hashes, counts, classes,
    and partition constraints, keeping metric code independent of filesystem trust checks.
    """

    model: BaselineModel
    records: list[str]
    labels: np.ndarray[Any, Any]
    features: np.ndarray[Any, Any]
    index_digest: FileDigest
    shard_digests: list[FileDigest]
    model_digest: FileDigest


def _load_validation_data(
    root: Path,
    dataset_index_path: Path,
    model_path: Path,
    training_metadata_path: Path,
) -> _ValidationData:
    """Load validation data according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        root: Repository root used to enforce path and trust boundaries.
        dataset_index_path: The dataset index path value supplied by the caller or surrounding test fixture.
        model_path: The model path value supplied by the caller or surrounding test fixture.
        training_metadata_path: The training metadata path value supplied by the caller or surrounding test
            fixture.

    Returns:
        The value produced by the documented operation.
    """

    index_path = _input_path(root, dataset_index_path, ("data", "processed"), "dataset index")
    frozen_model_path = _input_path(root, model_path, ("artifacts",), "baseline model")
    metadata_path = _input_path(root, training_metadata_path, ("artifacts",), "training metadata")
    index_digest = _digest(root, index_path)
    model_digest = _digest(root, frozen_model_path)
    index = _read_json(index_path, "dataset index")
    metadata = _read_json(metadata_path, "training metadata")
    _verify_training_digests(metadata, index_digest, model_digest)
    # Attempt this boundary operation here so TrainingError can be translated or cleaned up under
    # the repository contract.
    try:
        model = load_baseline_model(frozen_model_path)
    except TrainingError as error:
        raise EvaluationError(str(error)) from error
    validation = _validation_partition(index, model)

    matrices: list[np.ndarray[Any, Any]] = []
    targets: list[np.ndarray[Any, Any]] = []
    records: list[str] = []
    shard_digests: list[FileDigest] = []
    verified_shards: list[tuple[str, Path]] = []
    # Iterate over `validation['shards']` one item at a time so ordering, validation, and failure
    # attribution remain explicit.
    for descriptor in validation["shards"]:
        record_id, file_values = _shard_descriptor(descriptor)
        shard_path = _input_path(
            root, Path(file_values["path"]), ("data", "interim"), "validation shard"
        )
        digest = _digest(root, shard_path)
        # Evaluate `digest.size_bytes != file_values['size_bytes'] or digest.sha256 !=
        # file_values['sha256']` explicitly so invalid or alternate states follow the documented
        # contract.
        if digest.size_bytes != file_values["size_bytes"] or digest.sha256 != file_values["sha256"]:
            raise EvaluationError(
                f"validation shard digest does not match dataset index: {record_id}"
            )
        verified_shards.append((record_id, shard_path))
        shard_digests.append(digest)
    # Evaluate `not verified_shards` explicitly so invalid or alternate states follow the documented
    # contract.
    if not verified_shards:
        raise EvaluationError("validation partition must contain at least one shard")
    # Iterate over `verified_shards` one item at a time so ordering, validation, and failure
    # attribution remain explicit.
    for record_id, shard_path in verified_shards:
        windows, labels = _load_validation_shard(shard_path, record_id, model)
        matrices.append(windows)
        targets.append(labels)
        records.append(record_id)
    features = np.concatenate(matrices, axis=0)
    labels = np.concatenate(targets)
    _verify_partition_counts(validation, records, labels)
    return _ValidationData(
        model, records, labels, features, index_digest, shard_digests, model_digest
    )


def _distances(model: BaselineModel, features: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Calculate squared projected distances from every row to every centroid.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        model: Frozen model definition used by the current operation.
        features: Feature matrix consumed by the deterministic model operation.

    Returns:
        The value produced by the documented operation.
    """

    projection = np.asarray(model.projection, dtype=np.float64)
    centroids = np.asarray(model.centroids, dtype=np.float64)
    projected = features @ projection
    return np.sum((projected[:, None, :] - centroids[None, :, :]) ** 2, axis=2)


def _margin(distances: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Nearest-to-second-nearest centroid distance gap; raw squared distance, not a probability."""
    sorted_distances = np.sort(distances, axis=1)
    return sorted_distances[:, 1] - sorted_distances[:, 0]


def _predict(model: BaselineModel, features: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Assign each feature row to its nearest frozen class centroid.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        model: Frozen model definition used by the current operation.
        features: Feature matrix consumed by the deterministic model operation.

    Returns:
        The value produced by the documented operation.
    """

    distances = _distances(model, features)
    return np.asarray(model.classes, dtype=np.int64)[np.argmin(distances, axis=1)]


def _build_metrics(
    config: EvaluationConfig,
    model: BaselineModel,
    records: list[str],
    labels: np.ndarray[Any, Any],
    predictions: np.ndarray[Any, Any],
    index_digest: FileDigest,
    shard_digests: list[FileDigest],
    model_digest: FileDigest,
) -> ValidationMetrics:
    """Build metrics according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        config: Validated configuration controlling the documented operation.
        model: Frozen model definition used by the current operation.
        records: The records value supplied by the caller or surrounding test fixture.
        labels: Target labels retained for validation or metric calculation.
        predictions: The predictions value supplied by the caller or surrounding test fixture.
        index_digest: The index digest value supplied by the caller or surrounding test fixture.
        shard_digests: The shard digests value supplied by the caller or surrounding test fixture.
        model_digest: The model digest value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    classes = model.classes
    positions = {value: index for index, value in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    # Iterate over `zip(labels.tolist(), predictions.tolist(), strict=True)` one item at a time so
    # ordering, validation, and failure attribution remain explicit.
    for truth, prediction in zip(labels.tolist(), predictions.tolist(), strict=True):
        matrix[positions[int(truth)], positions[int(prediction)]] += 1
    per_class: dict[str, ClassMetrics] = {}
    # Iterate over `enumerate(classes)` one item at a time so ordering, validation, and failure
    # attribution remain explicit.
    for index, value in enumerate(classes):
        true_positive = int(matrix[index, index])
        predicted = int(matrix[:, index].sum())
        support = int(matrix[index, :].sum())
        precision = _divide(true_positive, predicted, config.zero_division)
        recall = _divide(true_positive, support, config.zero_division)
        f1 = _divide(2.0 * precision * recall, precision + recall, config.zero_division)
        per_class[str(value)] = ClassMetrics(precision, recall, f1, support)
    macro = ClassMetrics(
        precision=float(np.mean([item.precision for item in per_class.values()])),
        recall=float(np.mean([item.recall for item in per_class.values()])),
        f1=float(np.mean([item.f1 for item in per_class.values()])),
        support=len(labels),
    )
    return ValidationMetrics(
        1,
        config.name,
        config.version,
        config.evaluator,
        config.partition,
        config.zero_division,
        tuple(records),
        len(records),
        len(labels),
        classes,
        tuple(tuple(int(value) for value in row) for row in matrix),
        float(np.trace(matrix) / len(labels)),
        per_class,
        macro,
        index_digest,
        tuple(shard_digests),
        model_digest,
    )


def _validation_partition(index: dict[str, Any], model: BaselineModel) -> dict[str, Any]:
    """Compute and return validation partition for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        index: The index value supplied by the caller or surrounding test fixture.
        model: Frozen model definition used by the current operation.

    Returns:
        The value produced by the documented operation.
    """

    # Evaluate `index.get('schema_version') not in {1, 2}` explicitly so invalid or alternate states
    # follow the documented contract.
    if index.get("schema_version") not in {1, 2}:
        raise EvaluationError("dataset index must use schema_version 1 or 2")
    partitions = index.get("partitions")
    # Evaluate `not isinstance(partitions, dict) or set(partitions) != {'train', 'validation',
    # 'test'}` explicitly so invalid or alternate states follow the documented contract.
    if not isinstance(partitions, dict) or set(partitions) != {"train", "validation", "test"}:
        raise EvaluationError("dataset index must contain train, validation, and test partitions")
    validation = partitions.get(SUPPORTED_PARTITION)
    # Evaluate `not isinstance(validation, dict) or not isinstance(validation.get('shards'), list)`
    # explicitly so invalid or alternate states follow the documented contract.
    if not isinstance(validation, dict) or not isinstance(validation.get("shards"), list):
        raise EvaluationError("dataset index validation partition must contain shards")
    # Evaluate `index.get('window_samples') != model.input_features` explicitly so invalid or
    # alternate states follow the documented contract.
    if index.get("window_samples") != model.input_features:
        raise EvaluationError("baseline model input width does not match dataset index")
    return validation


def _verify_training_digests(
    metadata: dict[str, Any], index: FileDigest, model: FileDigest
) -> None:
    """Verify training digests according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        metadata: The metadata value supplied by the caller or surrounding test fixture.
        index: The index value supplied by the caller or surrounding test fixture.
        model: Frozen model definition used by the current operation.
    """

    # Evaluate `metadata.get('schema_version') != 1 or metadata.get('partition') != 'train'`
    # explicitly so invalid or alternate states follow the documented contract.
    if metadata.get("schema_version") != 1 or metadata.get("partition") != "train":
        raise EvaluationError("training metadata must use schema_version 1 for the train partition")
    # Iterate over `(('dataset_index', index), ('model', model))` one item at a time so ordering,
    # validation, and failure attribution remain explicit.
    for name, actual in (("dataset_index", index), ("model", model)):
        expected = metadata.get(name)
        # Evaluate `not isinstance(expected, dict) or any((expected.get(key) != getattr(actual, key)
        # for key in ('path', 'size_bytes', 's...` explicitly so invalid or alternate states follow
        # the documented contract.
        if not isinstance(expected, dict) or any(
            expected.get(key) != getattr(actual, key) for key in ("path", "size_bytes", "sha256")
        ):
            raise EvaluationError(
                f"{name.replace('_', ' ')} digest does not match training metadata"
            )


def _shard_descriptor(descriptor: Any) -> tuple[str, dict[str, Any]]:
    """Compute and return shard descriptor for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        descriptor: The descriptor value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    # Evaluate `not isinstance(descriptor, dict) or not isinstance(descriptor.get('file'), dict)`
    # explicitly so invalid or alternate states follow the documented contract.
    if not isinstance(descriptor, dict) or not isinstance(descriptor.get("file"), dict):
        raise EvaluationError("validation shard descriptor is invalid")
    record_id, values = descriptor.get("record_id"), descriptor["file"]
    # Evaluate `not isinstance(record_id, str) or not record_id` explicitly so invalid or alternate
    # states follow the documented contract.
    if not isinstance(record_id, str) or not record_id:
        raise EvaluationError("validation shard record_id must be a non-empty string")
    # Evaluate `not isinstance(values.get('path'), str) or not isinstance(values.get('size_bytes'),
    # int) or (not isinstance(values.ge...` explicitly so invalid or alternate states follow the
    # documented contract.
    if (
        not isinstance(values.get("path"), str)
        or not isinstance(values.get("size_bytes"), int)
        or not isinstance(values.get("sha256"), str)
    ):
        raise EvaluationError("validation shard file identity is invalid")
    return record_id, values


def _load_validation_shard(
    path: Path, record_id: str, model: BaselineModel
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Load validation shard according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        path: Filesystem path identifying the input or output under review.
        record_id: The record id value supplied by the caller or surrounding test fixture.
        model: Frozen model definition used by the current operation.

    Returns:
        The value produced by the documented operation.
    """

    # Attempt this boundary operation here so (BadZipFile, KeyError, OSError, ValueError) can be
    # translated or cleaned up under the repository contract.
    try:
        # Scope `np.load(path, allow_pickle=False)` here so resource cleanup occurs on both success
        # and failure paths.
        with np.load(path, allow_pickle=False) as artifact:
            windows = np.asarray(artifact["windows"])
            labels = np.asarray(artifact["target_values"])
            record_ids = np.asarray(artifact["record_ids"])
    except (BadZipFile, KeyError, OSError, ValueError) as error:
        raise EvaluationError(f"could not load validation shard {path}: {error}") from error
    # Evaluate `windows.ndim != 2 or windows.shape[0] == 0 or windows.shape[1] !=
    # model.input_features or (windows.dtype.kind != 'f')...` explicitly so invalid or alternate
    # states follow the documented contract.
    if (
        windows.ndim != 2
        or windows.shape[0] == 0
        or windows.shape[1] != model.input_features
        or windows.dtype.kind != "f"
        or not np.isfinite(windows).all()
    ):
        raise EvaluationError(
            "validation windows must be a non-empty compatible finite floating-point matrix"
        )
    # Evaluate `labels.ndim != 1 or labels.dtype.kind not in {'i', 'u'} or len(labels) !=
    # len(windows)` explicitly so invalid or alternate states follow the documented contract.
    if labels.ndim != 1 or labels.dtype.kind not in {"i", "u"} or len(labels) != len(windows):
        raise EvaluationError("validation targets must be an integer vector aligned with windows")
    unknown = sorted(set(int(value) for value in labels) - set(model.classes))
    # Evaluate `unknown` explicitly so invalid or alternate states follow the documented contract.
    if unknown:
        raise EvaluationError(f"validation targets contain labels unknown to the model: {unknown}")
    # Evaluate `record_ids.ndim != 1 or len(record_ids) != len(windows) or set(record_ids.tolist())
    # != {record_id}` explicitly so invalid or alternate states follow the documented contract.
    if (
        record_ids.ndim != 1
        or len(record_ids) != len(windows)
        or set(record_ids.tolist()) != {record_id}
    ):
        raise EvaluationError("validation shard record lineage does not match the dataset index")
    return windows, labels


def _verify_partition_counts(
    partition: dict[str, Any], records: list[str], labels: np.ndarray[Any, Any]
) -> None:
    """Verify partition counts according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        partition: The partition value supplied by the caller or surrounding test fixture.
        records: The records value supplied by the caller or surrounding test fixture.
        labels: Target labels retained for validation or metric calculation.
    """

    # Evaluate `len(records) != partition.get('record_count') or len(set(records)) != len(records)`
    # explicitly so invalid or alternate states follow the documented contract.
    if len(records) != partition.get("record_count") or len(set(records)) != len(records):
        raise EvaluationError("loaded validation records do not match dataset index")
    # Evaluate `len(labels) != partition.get('window_count')` explicitly so invalid or alternate
    # states follow the documented contract.
    if len(labels) != partition.get("window_count"):
        raise EvaluationError("loaded validation window count does not match dataset index")
    counts = {
        str(key): value for key, value in sorted(Counter(int(value) for value in labels).items())
    }
    # Evaluate `counts != partition.get('target_value_counts')` explicitly so invalid or alternate
    # states follow the documented contract.
    if counts != partition.get("target_value_counts"):
        raise EvaluationError("loaded validation target counts do not match dataset index")


def _read_json(path: Path, description: str) -> dict[str, Any]:
    """Read json according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        path: Filesystem path identifying the input or output under review.
        description: The description value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    # Attempt this boundary operation here so (OSError, UnicodeError, json.JSONDecodeError) can be
    # translated or cleaned up under the repository contract.
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvaluationError(f"could not read {description} {path}: {error}") from error
    # Evaluate `not isinstance(value, dict)` explicitly so invalid or alternate states follow the
    # documented contract.
    if not isinstance(value, dict):
        raise EvaluationError(f"{description} must be a JSON object")
    return value


def _string(values: dict[str, Any], key: str, *, table: str = "evaluation") -> str:
    """Require and return a non-empty string from the requested structured field.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        values: Structured values to validate, transform, or serialize.
        key: The key value supplied by the caller or surrounding test fixture.
        table: The table value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    value = values.get(key)
    # Evaluate `not isinstance(value, str) or not value.strip()` explicitly so invalid or alternate
    # states follow the documented contract.
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(f"{table}.{key} must be a non-empty string")
    return value.strip()


def _zero_division(value: Any, *, table: str = "evaluation") -> float:
    """Compute and return zero division for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        value: Candidate value whose contract is being enforced.
        table: The table value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    # Evaluate `isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) not
    # in {0.0, 1.0}` explicitly so invalid or alternate states follow the documented contract.
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or float(value) not in {0.0, 1.0}
    ):
        raise EvaluationError(f"{table}.zero_division must be 0.0 or 1.0")
    return float(value)


def _thresholds(value: Any) -> tuple[float, ...]:
    """Compute and return thresholds for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        value: Candidate value whose contract is being enforced.

    Returns:
        The value produced by the documented operation.
    """

    # Evaluate `not isinstance(value, list) or not value or any((isinstance(item, bool) or not
    # isinstance(item, (int, float)) for ite...` explicitly so invalid or alternate states follow
    # the documented contract.
    if (
        not isinstance(value, list)
        or not value
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
    ):
        raise EvaluationError("threshold_sweep.thresholds must be a non-empty numeric array")
    thresholds = tuple(float(item) for item in value)
    # Evaluate `any((not np.isfinite(item) for item in thresholds))` explicitly so invalid or
    # alternate states follow the documented contract.
    if any(not np.isfinite(item) for item in thresholds):
        raise EvaluationError("threshold_sweep.thresholds must be finite")
    # Evaluate `any((earlier >= later for earlier, later in zip(thresholds, thresholds[1:],
    # strict=False)))` explicitly so invalid or alternate states follow the documented contract.
    if any(earlier >= later for earlier, later in zip(thresholds, thresholds[1:], strict=False)):
        raise EvaluationError("threshold_sweep.thresholds must be strictly increasing")
    return thresholds


def _divide(numerator: float, denominator: float, zero_division: float) -> float:
    """Divide one metric component using the configured zero-division policy.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        numerator: The numerator value supplied by the caller or surrounding test fixture.
        denominator: The denominator value supplied by the caller or surrounding test fixture.
        zero_division: The zero division value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    return zero_division if denominator == 0 else float(numerator / denominator)


def _input_path(root: Path, path: Path, prefix: tuple[str, ...], description: str) -> Path:
    """Resolve and validate input path for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        root: Repository root used to enforce path and trust boundaries.
        path: Filesystem path identifying the input or output under review.
        prefix: The prefix value supplied by the caller or surrounding test fixture.
        description: The description value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    candidate = path if path.is_absolute() else root / path
    # Evaluate `candidate.is_symlink()` explicitly so invalid or alternate states follow the
    # documented contract.
    if candidate.is_symlink():
        raise EvaluationError(f"{description} must not be a symbolic link")
    resolved = candidate.resolve()
    # Attempt this boundary operation here so ValueError can be translated or cleaned up under the
    # repository contract.
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise EvaluationError(f"{description} must stay within repository root") from error
    # Evaluate `relative.parts[:len(prefix)] != prefix or not resolved.is_file()` explicitly so
    # invalid or alternate states follow the documented contract.
    if relative.parts[: len(prefix)] != prefix or not resolved.is_file():
        raise EvaluationError(f"{description} must be a regular file under {'/'.join(prefix)}/")
    return resolved


def _output_path(root: Path, path: Path, filename: str = "validation-metrics.json") -> Path:
    """Resolve and validate output path for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        root: Repository root used to enforce path and trust boundaries.
        path: Filesystem path identifying the input or output under review.
        filename: The filename value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    candidate = path if path.is_absolute() else root / path
    # Evaluate `candidate.is_symlink()` explicitly so invalid or alternate states follow the
    # documented contract.
    if candidate.is_symlink():
        raise EvaluationError("evaluation metrics output must not be a symbolic link")
    resolved = candidate.resolve()
    # Attempt this boundary operation here so ValueError can be translated or cleaned up under the
    # repository contract.
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise EvaluationError(
            "evaluation metrics output must stay within repository root"
        ) from error
    # Evaluate `relative.parts[:2] != ('artifacts', 'runs') or relative.parts[-2:] != ('evaluation',
    # filename)` explicitly so invalid or alternate states follow the documented contract.
    if relative.parts[:2] != ("artifacts", "runs") or relative.parts[-2:] != (
        "evaluation",
        filename,
    ):
        raise EvaluationError(
            f"evaluation metrics must be artifacts/runs/<run-id>/evaluation/{filename}"
        )
    # Evaluate `not resolved.parent.is_dir()` explicitly so invalid or alternate states follow the
    # documented contract.
    if not resolved.parent.is_dir():
        raise EvaluationError(f"evaluation metrics output parent does not exist: {resolved.parent}")
    return resolved


def _write_new(path: Path, content: str) -> None:
    """Write new according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        path: Filesystem path identifying the input or output under review.
        content: The content value supplied by the caller or surrounding test fixture.
    """

    # Attempt this boundary operation here so FileExistsError, OSError can be translated or cleaned
    # up under the repository contract.
    try:
        # Scope `path.open('x', encoding='utf-8')` here so resource cleanup occurs on both success
        # and failure paths.
        with path.open("x", encoding="utf-8") as output:
            output.write(content)
    except FileExistsError as error:
        raise EvaluationError(f"evaluation output already exists: {path}") from error
    except OSError as error:
        raise EvaluationError(f"could not write evaluation output {path}: {error}") from error


def _digest(root: Path, path: Path) -> FileDigest:
    """Calculate stable size and SHA-256 evidence for one repository file.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        root: Repository root used to enforce path and trust boundaries.
        path: Filesystem path identifying the input or output under review.

    Returns:
        The value produced by the documented operation.
    """

    digest, size = hashlib.sha256(), 0
    # Scope `path.open('rb')` here so resource cleanup occurs on both success and failure paths.
    with path.open("rb") as source:
        # Continue while `(chunk := source.read(BUFFER_SIZE))` so the loop's termination rule
        # remains visible to reviewers.
        while chunk := source.read(BUFFER_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return FileDigest(path.relative_to(root).as_posix(), size, digest.hexdigest())
