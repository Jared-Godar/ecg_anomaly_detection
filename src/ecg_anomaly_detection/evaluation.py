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

# Chunk size for streaming file reads during digest computation. 1 MiB balances syscall
# overhead against peak memory for shard files that can be tens of megabytes.
BUFFER_SIZE = 1024 * 1024
# The only partition this module is permitted to read. Hard-coded (not configurable)
# because it is the enforcement mechanism for this repository's core invariant that
# evaluation never opens the protected test partition -- see _validation_partition,
# where this constant is the single place that name is looked up from the dataset index.
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

        Returns:
            The metrics as a JSON string with sorted, deterministic key ordering.
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

        Returns:
            The sweep metrics as a JSON string with sorted, deterministic key ordering.
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
    """Load and validate a versioned evaluation configuration.

    Args:
        path: Path to the evaluation config TOML file.

    Returns:
        The validated config, pinned to the supported estimator and validation partition.
    """

    # Translate a missing, unreadable, or malformed-TOML file into EvaluationError.
    try:
        # The `with` block ensures the file handle closes even if tomllib.load raises.
        with path.open("rb") as source:
            document = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise EvaluationError(f"could not load evaluation config {path}: {error}") from error
    values = document.get("evaluation")
    # schema_version pins this loader's understanding of the [evaluation] table's shape.
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
    # This module only implements evaluation against the frozen baseline estimator; a
    # config naming any other estimator would otherwise pass structural checks while
    # requesting behavior this module doesn't provide.
    if config.evaluator != SUPPORTED_ESTIMATOR:
        raise EvaluationError(f"evaluation.evaluator must be {SUPPORTED_ESTIMATOR!r}")
    # This is the config-level half of the validation-only enforcement: a config that
    # names any partition other than "validation" is rejected before it ever reaches
    # the dataset-index lookup in _validation_partition.
    if config.partition != SUPPORTED_PARTITION:
        raise EvaluationError("evaluation.partition must be 'validation'")
    return config


def load_threshold_sweep_config(path: Path) -> ThresholdSweepConfig:
    """Load and validate a versioned threshold-sweep configuration.

    Args:
        path: Path to the threshold-sweep config TOML file.

    Returns:
        The validated config, with thresholds checked as finite and strictly increasing.
    """

    # Translate a missing, unreadable, or malformed-TOML file into EvaluationError.
    try:
        # The `with` block ensures the file handle closes even if tomllib.load raises.
        with path.open("rb") as source:
            document = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise EvaluationError(f"could not load threshold sweep config {path}: {error}") from error
    values = document.get("threshold_sweep")
    # schema_version pins this loader's understanding of the [threshold_sweep] table.
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
    # Same validation-only enforcement as load_evaluation_config, for the sweep config.
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
    # Compute each class's metrics only over the covered subset, then macro-average --
    # a class with zero covered support falls back to the configured zero_division value
    # via _divide, rather than raising or silently excluding that class from the average.
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
    """Load, verify, and assemble every input needed to evaluate, reading only validation.

    This is the single choke point through which both evaluate_validation_from_index and
    evaluate_threshold_sweep_from_index obtain their data; every shard opened here is
    drawn exclusively from the dataset index's "validation" partition (see
    _validation_partition), which is what makes "evaluation never opens the test
    partition" an enforced property of this module rather than a convention callers
    must remember to follow.

    Args:
        root: Repository root used to enforce path and trust boundaries.
        dataset_index_path: Path to the dataset index JSON produced by dataset_index.py.
        model_path: Path to the frozen baseline model JSON produced by training.py.
        training_metadata_path: Path to the training metadata JSON, used to cross-check
            that this model/index pair actually corresponds to a real completed training run.

    Returns:
        Verified feature matrix, labels, records, and lineage digests for evaluation.
    """

    index_path = _input_path(root, dataset_index_path, ("data", "processed"), "dataset index")
    frozen_model_path = _input_path(root, model_path, ("artifacts",), "baseline model")
    metadata_path = _input_path(root, training_metadata_path, ("artifacts",), "training metadata")
    index_digest = _digest(root, index_path)
    model_digest = _digest(root, frozen_model_path)
    index = _read_json(index_path, "dataset index")
    metadata = _read_json(metadata_path, "training metadata")
    _verify_training_digests(metadata, index_digest, model_digest)
    # load_baseline_model raises TrainingError for a malformed model; re-raise as this
    # module's own exception type so callers only need to catch EvaluationError.
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
    # First pass: resolve and digest-verify every validation shard path before loading
    # any window data, so a digest mismatch fails fast without partially loading shards.
    for descriptor in validation["shards"]:
        record_id, file_values = _shard_descriptor(descriptor)
        shard_path = _input_path(
            root, Path(file_values["path"]), ("data", "interim"), "validation shard"
        )
        digest = _digest(root, shard_path)
        # Recompute the shard's digest from the file actually on disk and compare against
        # what the index recorded -- this catches a shard that changed after indexing,
        # which would otherwise silently evaluate against different data than the
        # index's own lineage claims.
        if digest.size_bytes != file_values["size_bytes"] or digest.sha256 != file_values["sha256"]:
            raise EvaluationError(
                f"validation shard digest does not match dataset index: {record_id}"
            )
        verified_shards.append((record_id, shard_path))
        shard_digests.append(digest)
    # Evaluating with zero shards would otherwise silently produce empty metrics.
    if not verified_shards:
        raise EvaluationError("validation partition must contain at least one shard")
    # Second pass: only load window data after every shard has passed digest
    # verification above, keeping the "fail before loading anything untrusted" ordering.
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

    Shared by both _predict (which takes the argmin) and evaluate_threshold_sweep_from_index
    (which additionally needs the full distance matrix to compute margins), so the
    projection arithmetic is only implemented once.

    Args:
        model: The frozen model supplying the projection matrix and centroids.
        features: Raw feature matrix to project and measure distances for.

    Returns:
        A (rows, classes) matrix of squared distances from each row to each centroid.
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

    Args:
        model: The frozen model supplying the projection matrix, centroids, and classes.
        features: Raw feature matrix to classify.

    Returns:
        Predicted class label for each row.
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
    """Assemble the confusion matrix, per-class, and macro-averaged validation metrics.

    Args:
        config: Evaluation config supplying the zero_division policy and run identity.
        model: The frozen model whose classes define the confusion matrix's axes.
        records: Record IDs included in this evaluation.
        labels: Ground-truth labels for every scored window, aligned with predictions.
        predictions: Predicted labels for every scored window, aligned with labels.
        index_digest: Digest of the dataset index this evaluation read from.
        shard_digests: Digests of every validation shard actually opened.
        model_digest: Digest of the frozen model file used for prediction.

    Returns:
        Complete validation metrics with full artifact lineage attached.
    """

    classes = model.classes
    positions = {value: index for index, value in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    # Build the confusion matrix by incrementing one cell per (truth, prediction) pair;
    # strict=True guards against labels/predictions silently having different lengths.
    for truth, prediction in zip(labels.tolist(), predictions.tolist(), strict=True):
        matrix[positions[int(truth)], positions[int(prediction)]] += 1
    per_class: dict[str, ClassMetrics] = {}
    # Derive precision/recall/F1 for each class directly from the confusion matrix's
    # rows/columns/diagonal, rather than recomputing from labels/predictions again.
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
    """Extract and validate the dataset index's "validation" partition entry.

    This is the enforcement point for the repository's core evaluation invariant: it
    reads SUPPORTED_PARTITION ("validation") specifically and never any other key from
    `partitions`, so no code path in this module can be pointed at the test partition
    without changing this function itself.

    Args:
        index: The parsed dataset index document.
        model: The frozen model, used to cross-check input width against the index.

    Returns:
        The validation partition's shard/count entry from the index.
    """

    # dataset_index.py currently writes schema_version 1 or 2; anything else means this
    # index predates or postdates what this loader understands.
    if index.get("schema_version") not in {1, 2}:
        raise EvaluationError("dataset index must use schema_version 1 or 2")
    partitions = index.get("partitions")
    # An index must always declare all three fixed partitions, even though only
    # "validation" is ever read below -- their presence is evidence the index is
    # well-formed, not permission to read train or test.
    if not isinstance(partitions, dict) or set(partitions) != {"train", "validation", "test"}:
        raise EvaluationError("dataset index must contain train, validation, and test partitions")
    validation = partitions.get(SUPPORTED_PARTITION)
    # The validation entry itself must carry a shards list before anything downstream
    # can iterate over it.
    if not isinstance(validation, dict) or not isinstance(validation.get("shards"), list):
        raise EvaluationError("dataset index validation partition must contain shards")
    # A model trained on windows of one width can't meaningfully score windows of a
    # different width; catching this here produces a clear error instead of a numpy
    # shape-mismatch exception deep inside the projection arithmetic.
    if index.get("window_samples") != model.input_features:
        raise EvaluationError("baseline model input width does not match dataset index")
    return validation


def _verify_training_digests(
    metadata: dict[str, Any], index: FileDigest, model: FileDigest
) -> None:
    """Confirm the model and index being evaluated match the training run that made them.

    training.py's TrainingMetadata records the exact dataset-index and model digests it
    produced; cross-checking those recorded digests against the files actually being
    evaluated here prevents accidentally scoring a model against an index (or vice
    versa) from an unrelated run.

    Args:
        metadata: The parsed training metadata document.
        index: Digest of the dataset index file currently being evaluated.
        model: Digest of the model file currently being evaluated.
    """

    # Training metadata is only ever written for the train partition; anything else
    # means this file isn't what it claims to be.
    if metadata.get("schema_version") != 1 or metadata.get("partition") != "train":
        raise EvaluationError("training metadata must use schema_version 1 for the train partition")
    # Cross-check both the dataset_index and model digests the same way, since a
    # mismatch in either would mean this evaluation is scoring the wrong artifacts.
    for name, actual in (("dataset_index", index), ("model", model)):
        expected = metadata.get(name)
        # All three identity fields must match exactly for the digests to be equal.
        if not isinstance(expected, dict) or any(
            expected.get(key) != getattr(actual, key) for key in ("path", "size_bytes", "sha256")
        ):
            raise EvaluationError(
                f"{name.replace('_', ' ')} digest does not match training metadata"
            )


def _shard_descriptor(descriptor: Any) -> tuple[str, dict[str, Any]]:
    """Extract and validate one shard entry's record ID and file identity fields.

    Args:
        descriptor: One raw shard entry from the dataset index's validation partition.

    Returns:
        The record ID and its nested file-identity dict (path/size_bytes/sha256).
    """

    # Every descriptor must carry a nested file object; a malformed descriptor here
    # would otherwise raise a bare KeyError further down instead of a clear message.
    if not isinstance(descriptor, dict) or not isinstance(descriptor.get("file"), dict):
        raise EvaluationError("validation shard descriptor is invalid")
    record_id, values = descriptor.get("record_id"), descriptor["file"]
    # A shard is meaningless without knowing which record it belongs to.
    if not isinstance(record_id, str) or not record_id:
        raise EvaluationError("validation shard record_id must be a non-empty string")
    # These three fields are what _load_validation_data compares against a freshly
    # computed digest; if any is the wrong type, that comparison would be meaningless.
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
    """Load and validate one validation shard's windows and labels from an NPZ file.

    Args:
        path: Path to the shard NPZ file, already resolved and digest-verified.
        record_id: The record ID the dataset index claims this shard belongs to,
            cross-checked against the shard's own embedded record_ids field.
        model: The frozen model, used to check window width and known class labels.

    Returns:
        The shard's window matrix and aligned target-label vector.
    """

    # allow_pickle=False is a security boundary against arbitrary code execution from an
    # untrusted NPZ file; collapse every load/parse failure mode into EvaluationError.
    try:
        # The `with` block ensures the lazy NpzFile handle closes even if a field access
        # below raises.
        with np.load(path, allow_pickle=False) as artifact:
            windows = np.asarray(artifact["windows"])
            labels = np.asarray(artifact["target_values"])
            record_ids = np.asarray(artifact["record_ids"])
    except (BadZipFile, KeyError, OSError, ValueError) as error:
        raise EvaluationError(f"could not load validation shard {path}: {error}") from error
    # The window matrix must be a non-empty, finite 2-D float array whose width matches
    # what the frozen model expects -- a mismatch here would otherwise fail deep inside
    # the projection matrix multiplication with a much less specific numpy error.
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
    # Labels must be a 1-D integer vector with one entry per window row.
    if labels.ndim != 1 or labels.dtype.kind not in {"i", "u"} or len(labels) != len(windows):
        raise EvaluationError("validation targets must be an integer vector aligned with windows")
    unknown = sorted(set(int(value) for value in labels) - set(model.classes))
    # A validation label the frozen model was never trained on can't be predicted for
    # (the model's centroids only cover its own classes), so it's rejected outright
    # rather than silently producing an always-wrong prediction for that class.
    if unknown:
        raise EvaluationError(f"validation targets contain labels unknown to the model: {unknown}")
    # A shard is defined as one record's windows; this cross-checks the shard's own
    # embedded record_ids against what the dataset index claimed this shard contains.
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
    """Cross-check loaded validation data against the dataset index's own recorded counts.

    Args:
        partition: The validation partition's entry from the dataset index.
        records: Record IDs actually loaded.
        labels: Labels actually loaded, concatenated across every shard.
    """

    # Also guards against a duplicate record contributing windows twice.
    if len(records) != partition.get("record_count") or len(set(records)) != len(records):
        raise EvaluationError("loaded validation records do not match dataset index")
    # Same cross-check at the window-row-count level.
    if len(labels) != partition.get("window_count"):
        raise EvaluationError("loaded validation window count does not match dataset index")
    counts = {
        str(key): value for key, value in sorted(Counter(int(value) for value in labels).items())
    }
    # The strongest of the three cross-checks, since it depends on every label value
    # being correct, not just the row/record counts.
    if counts != partition.get("target_value_counts"):
        raise EvaluationError("loaded validation target counts do not match dataset index")


def _read_json(path: Path, description: str) -> dict[str, Any]:
    """Read and parse one JSON document, requiring it to be a top-level object.

    Args:
        path: Path to the JSON file to read.
        description: Human-readable label for this file, used in error messages.

    Returns:
        The parsed document.
    """

    # Translate a missing, unreadable, or malformed-JSON file into EvaluationError.
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvaluationError(f"could not read {description} {path}: {error}") from error
    # Every caller of this helper expects to call .get()/dict-index on the result;
    # a top-level JSON array or scalar would otherwise fail unpredictably downstream.
    if not isinstance(value, dict):
        raise EvaluationError(f"{description} must be a JSON object")
    return value


def _string(values: dict[str, Any], key: str, *, table: str = "evaluation") -> str:
    """Require and return a non-empty string from the requested structured field.

    The `table` parameter lets one implementation serve both [evaluation] and
    [threshold_sweep] fields while still raising a dotted-path error naming the right
    table.

    Args:
        values: The parsed config table to read from.
        key: The field name to extract.
        table: The TOML table name to use in the error message.

    Returns:
        The field's value with surrounding whitespace stripped.
    """

    value = values.get(key)
    # Reject a missing/wrong-typed value and a whitespace-only placeholder alike.
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(f"{table}.{key} must be a non-empty string")
    return value.strip()


def _zero_division(value: Any, *, table: str = "evaluation") -> float:
    """Require and return the zero-division policy, restricted to exactly 0.0 or 1.0.

    Only these two values are meaningful for a metric with an undefined (0/0)
    denominator: report it as the pessimistic 0.0 or optimistic 1.0. Anything else
    would be an arbitrary, hard-to-interpret metric value.

    Args:
        value: The raw TOML value for zero_division.
        table: The TOML table name to use in the error message.

    Returns:
        The validated policy value, exactly 0.0 or 1.0.
    """

    # bool is an int subclass in Python, so it's excluded explicitly.
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or float(value) not in {0.0, 1.0}
    ):
        raise EvaluationError(f"{table}.zero_division must be 0.0 or 1.0")
    return float(value)


def _thresholds(value: Any) -> tuple[float, ...]:
    """Require and return a non-empty, finite, strictly increasing threshold sequence.

    Strictly increasing is required because thresholds define a monotonic coverage
    sweep (each higher threshold should exclude a superset of what the previous one
    excluded); a non-increasing sequence wouldn't have that property and its coverage
    counts would be hard to interpret in order.

    Args:
        value: The raw TOML value for threshold_sweep.thresholds.

    Returns:
        The validated thresholds as a tuple of floats.
    """

    # bool is an int subclass in Python, so it's excluded explicitly from the numeric check.
    if (
        not isinstance(value, list)
        or not value
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
    ):
        raise EvaluationError("threshold_sweep.thresholds must be a non-empty numeric array")
    thresholds = tuple(float(item) for item in value)
    # A NaN/inf threshold would make the coverage comparison (margins >= threshold)
    # behave unpredictably rather than raising a clear configuration error.
    if any(not np.isfinite(item) for item in thresholds):
        raise EvaluationError("threshold_sweep.thresholds must be finite")
    # Compare each threshold to the next; any non-increasing adjacent pair violates the
    # strictly-increasing requirement explained in this function's docstring.
    if any(earlier >= later for earlier, later in zip(thresholds, thresholds[1:], strict=False)):
        raise EvaluationError("threshold_sweep.thresholds must be strictly increasing")
    return thresholds


def _divide(numerator: float, denominator: float, zero_division: float) -> float:
    """Divide one metric component using the configured zero-division policy.

    Args:
        numerator: The metric's numerator (e.g. true positives).
        denominator: The metric's denominator (e.g. predicted positives).
        zero_division: The value to report when denominator is exactly zero.

    Returns:
        The computed ratio, or zero_division if the denominator is zero.
    """

    return zero_division if denominator == 0 else float(numerator / denominator)


def _input_path(root: Path, path: Path, prefix: tuple[str, ...], description: str) -> Path:
    """Resolve a path and enforce it stays within a required subdirectory of the repo.

    Shared by every stage in this module that reads a repository-relative input
    (dataset index, model, training metadata, validation shards): resolving through
    symlinks and checking containment prevents a maliciously or accidentally crafted
    path from reading files outside the repository the pipeline operates on.

    Args:
        root: Repository root used to enforce path and trust boundaries.
        path: The candidate path, absolute or relative to root.
        prefix: The path segments the resolved file must be nested under.
        description: Human-readable label for this file, used in error messages.

    Returns:
        The resolved, validated absolute path.
    """

    candidate = path if path.is_absolute() else root / path
    # Reject a symlink before resolving it, so a link that points outside the required
    # prefix can't be validated against a resolved target it doesn't actually name.
    if candidate.is_symlink():
        raise EvaluationError(f"{description} must not be a symbolic link")
    resolved = candidate.resolve()
    # relative_to raises ValueError when resolved escapes root (e.g. via `..` segments);
    # translate that into this module's own exception type.
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise EvaluationError(f"{description} must stay within repository root") from error
    # Confirm both containment under the expected subtree and that it's a regular file
    # (not a directory or special file) in one combined check.
    if relative.parts[: len(prefix)] != prefix or not resolved.is_file():
        raise EvaluationError(f"{description} must be a regular file under {'/'.join(prefix)}/")
    return resolved


def _output_path(root: Path, path: Path, filename: str = "validation-metrics.json") -> Path:
    """Resolve a path and enforce it matches the fixed run-scoped evaluation layout.

    Every evaluation output lands at artifacts/runs/<run-id>/evaluation/<filename>,
    matching this repository's per-run artifact isolation contract (see run_manifest.py)
    -- an evaluation output can never be written outside its own run's directory.

    Args:
        root: Repository root used to enforce path and trust boundaries.
        path: The candidate output path, absolute or relative to root.
        filename: The expected final path segment (varies between plain evaluation
            metrics and threshold-sweep metrics).

    Returns:
        The resolved, validated absolute path.
    """

    candidate = path if path.is_absolute() else root / path
    # Reject a symlink before resolving it: resolving would silently follow the link and
    # write to wherever it points, defeating the repository-root containment check below.
    if candidate.is_symlink():
        raise EvaluationError("evaluation metrics output must not be a symbolic link")
    resolved = candidate.resolve()
    # relative_to raises ValueError when resolved escapes root.
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise EvaluationError(
            "evaluation metrics output must stay within repository root"
        ) from error
    # The path must match artifacts/runs/<run-id>/evaluation/<filename> exactly; the
    # middle <run-id> segment is intentionally unconstrained here (checked elsewhere by
    # run_manifest.py's run-ID validation) since this function only enforces layout shape.
    if relative.parts[:2] != ("artifacts", "runs") or relative.parts[-2:] != (
        "evaluation",
        filename,
    ):
        raise EvaluationError(
            f"evaluation metrics must be artifacts/runs/<run-id>/evaluation/{filename}"
        )
    # Fail before attempting the write rather than letting a missing parent directory
    # surface as a generic OSError from _write_new.
    if not resolved.parent.is_dir():
        raise EvaluationError(f"evaluation metrics output parent does not exist: {resolved.parent}")
    return resolved


def _write_new(path: Path, content: str) -> None:
    """Write content to a path that must not already exist.

    Args:
        path: Destination path; must not already exist.
        content: The text to write.
    """

    # Collapse "already exists" and other OS-level failures into EvaluationError.
    try:
        # Open with mode "x" (exclusive create) so this can never silently overwrite a
        # previous run's evaluation output.
        with path.open("x", encoding="utf-8") as output:
            output.write(content)
    except FileExistsError as error:
        raise EvaluationError(f"evaluation output already exists: {path}") from error
    except OSError as error:
        raise EvaluationError(f"could not write evaluation output {path}: {error}") from error


def _digest(root: Path, path: Path) -> FileDigest:
    """Calculate stable size and SHA-256 evidence for one repository file.

    Args:
        root: Repository root, used to store the digest's path relative rather than
            absolute (so manifests remain portable across checkouts).
        path: The already-resolved, validated file path to hash.

    Returns:
        The file's repository-relative path, size, and SHA-256 digest.
    """

    digest, size = hashlib.sha256(), 0
    # Read in fixed-size chunks rather than the whole file at once, since dataset
    # indexes and shard files can be tens of megabytes.
    with path.open("rb") as source:
        # The walrus operator lets both the read and the loop's termination condition
        # (an empty final chunk) live in one line without a separate `break`.
        while chunk := source.read(BUFFER_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return FileDigest(path.relative_to(root).as_posix(), size, digest.hexdigest())
