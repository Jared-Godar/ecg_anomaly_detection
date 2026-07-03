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

BUFFER_SIZE = 1024 * 1024
SUPPORTED_PARTITION = "validation"


class EvaluationError(ValueError):
    """Raised when validation evaluation cannot satisfy its isolation contract."""


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    schema_version: int
    name: str
    version: str
    evaluator: str
    partition: str
    zero_division: float


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True, slots=True)
class ValidationMetrics:
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
        return json.dumps(asdict(self), indent=2, sort_keys=True, allow_nan=False) + "\n"


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    metrics_path: Path
    window_count: int
    record_count: int


def load_evaluation_config(path: Path) -> EvaluationConfig:
    try:
        with path.open("rb") as source:
            document = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise EvaluationError(f"could not load evaluation config {path}: {error}") from error
    values = document.get("evaluation")
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
    if config.evaluator != SUPPORTED_ESTIMATOR:
        raise EvaluationError(f"evaluation.evaluator must be {SUPPORTED_ESTIMATOR!r}")
    if config.partition != SUPPORTED_PARTITION:
        raise EvaluationError("evaluation.partition must be 'validation'")
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
    index_path = _input_path(root, dataset_index_path, ("data", "processed"), "dataset index")
    frozen_model_path = _input_path(root, model_path, ("artifacts",), "baseline model")
    metadata_path = _input_path(root, training_metadata_path, ("artifacts",), "training metadata")
    index_digest = _digest(root, index_path)
    model_digest = _digest(root, frozen_model_path)
    index = _read_json(index_path, "dataset index")
    metadata = _read_json(metadata_path, "training metadata")
    _verify_training_digests(metadata, index_digest, model_digest)
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
    for descriptor in validation["shards"]:
        record_id, file_values = _shard_descriptor(descriptor)
        shard_path = _input_path(
            root, Path(file_values["path"]), ("data", "interim"), "validation shard"
        )
        digest = _digest(root, shard_path)
        if digest.size_bytes != file_values["size_bytes"] or digest.sha256 != file_values["sha256"]:
            raise EvaluationError(
                f"validation shard digest does not match dataset index: {record_id}"
            )
        verified_shards.append((record_id, shard_path))
        shard_digests.append(digest)
    if not verified_shards:
        raise EvaluationError("validation partition must contain at least one shard")
    for record_id, shard_path in verified_shards:
        windows, labels = _load_validation_shard(shard_path, record_id, model)
        matrices.append(windows)
        targets.append(labels)
        records.append(record_id)
    features = np.concatenate(matrices, axis=0)
    labels = np.concatenate(targets)
    _verify_partition_counts(validation, records, labels)
    predictions = _predict(model, features)
    metrics = _build_metrics(
        config, model, records, labels, predictions, index_digest, shard_digests, model_digest
    )
    _write_new(output, metrics.to_json())
    return EvaluationResult(output, len(labels), len(records))


def _predict(model: BaselineModel, features: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    projection = np.asarray(model.projection, dtype=np.float64)
    centroids = np.asarray(model.centroids, dtype=np.float64)
    projected = features @ projection
    distances = np.sum((projected[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
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
    classes = model.classes
    positions = {value: index for index, value in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for truth, prediction in zip(labels.tolist(), predictions.tolist(), strict=True):
        matrix[positions[int(truth)], positions[int(prediction)]] += 1
    per_class: dict[str, ClassMetrics] = {}
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
    if index.get("schema_version") != 1:
        raise EvaluationError("dataset index must use schema_version 1")
    partitions = index.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != {"train", "validation", "test"}:
        raise EvaluationError("dataset index must contain train, validation, and test partitions")
    validation = partitions.get(SUPPORTED_PARTITION)
    if not isinstance(validation, dict) or not isinstance(validation.get("shards"), list):
        raise EvaluationError("dataset index validation partition must contain shards")
    if index.get("window_samples") != model.input_features:
        raise EvaluationError("baseline model input width does not match dataset index")
    return validation


def _verify_training_digests(
    metadata: dict[str, Any], index: FileDigest, model: FileDigest
) -> None:
    if metadata.get("schema_version") != 1 or metadata.get("partition") != "train":
        raise EvaluationError("training metadata must use schema_version 1 for the train partition")
    for name, actual in (("dataset_index", index), ("model", model)):
        expected = metadata.get(name)
        if not isinstance(expected, dict) or any(
            expected.get(key) != getattr(actual, key) for key in ("path", "size_bytes", "sha256")
        ):
            raise EvaluationError(
                f"{name.replace('_', ' ')} digest does not match training metadata"
            )


def _shard_descriptor(descriptor: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(descriptor, dict) or not isinstance(descriptor.get("file"), dict):
        raise EvaluationError("validation shard descriptor is invalid")
    record_id, values = descriptor.get("record_id"), descriptor["file"]
    if not isinstance(record_id, str) or not record_id:
        raise EvaluationError("validation shard record_id must be a non-empty string")
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
    try:
        with np.load(path, allow_pickle=False) as artifact:
            windows = np.asarray(artifact["windows"])
            labels = np.asarray(artifact["target_values"])
            record_ids = np.asarray(artifact["record_ids"])
    except (BadZipFile, KeyError, OSError, ValueError) as error:
        raise EvaluationError(f"could not load validation shard {path}: {error}") from error
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
    if labels.ndim != 1 or labels.dtype.kind not in {"i", "u"} or len(labels) != len(windows):
        raise EvaluationError("validation targets must be an integer vector aligned with windows")
    unknown = sorted(set(int(value) for value in labels) - set(model.classes))
    if unknown:
        raise EvaluationError(f"validation targets contain labels unknown to the model: {unknown}")
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
    if len(records) != partition.get("record_count") or len(set(records)) != len(records):
        raise EvaluationError("loaded validation records do not match dataset index")
    if len(labels) != partition.get("window_count"):
        raise EvaluationError("loaded validation window count does not match dataset index")
    counts = {
        str(key): value for key, value in sorted(Counter(int(value) for value in labels).items())
    }
    if counts != partition.get("target_value_counts"):
        raise EvaluationError("loaded validation target counts do not match dataset index")


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvaluationError(f"could not read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise EvaluationError(f"{description} must be a JSON object")
    return value


def _string(values: dict[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(f"evaluation.{key} must be a non-empty string")
    return value.strip()


def _zero_division(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or float(value) not in {0.0, 1.0}
    ):
        raise EvaluationError("evaluation.zero_division must be 0.0 or 1.0")
    return float(value)


def _divide(numerator: float, denominator: float, zero_division: float) -> float:
    return zero_division if denominator == 0 else float(numerator / denominator)


def _input_path(root: Path, path: Path, prefix: tuple[str, ...], description: str) -> Path:
    candidate = path if path.is_absolute() else root / path
    if candidate.is_symlink():
        raise EvaluationError(f"{description} must not be a symbolic link")
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise EvaluationError(f"{description} must stay within repository root") from error
    if relative.parts[: len(prefix)] != prefix or not resolved.is_file():
        raise EvaluationError(f"{description} must be a regular file under {'/'.join(prefix)}/")
    return resolved


def _output_path(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    if candidate.is_symlink():
        raise EvaluationError("evaluation metrics output must not be a symbolic link")
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise EvaluationError(
            "evaluation metrics output must stay within repository root"
        ) from error
    if relative.parts[:2] != ("artifacts", "runs") or relative.parts[-2:] != (
        "evaluation",
        "validation-metrics.json",
    ):
        raise EvaluationError(
            "evaluation metrics must be artifacts/runs/<run-id>/evaluation/validation-metrics.json"
        )
    if not resolved.parent.is_dir():
        raise EvaluationError(f"evaluation metrics output parent does not exist: {resolved.parent}")
    return resolved


def _write_new(path: Path, content: str) -> None:
    try:
        with path.open("x", encoding="utf-8") as output:
            output.write(content)
    except FileExistsError as error:
        raise EvaluationError(f"evaluation output already exists: {path}") from error
    except OSError as error:
        raise EvaluationError(f"could not write evaluation output {path}: {error}") from error


def _digest(root: Path, path: Path) -> FileDigest:
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as source:
        while chunk := source.read(BUFFER_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return FileDigest(path.relative_to(root).as_posix(), size, digest.hexdigest())
