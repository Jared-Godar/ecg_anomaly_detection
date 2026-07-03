"""Deterministic baseline fitting over the indexed training partition only."""

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

BUFFER_SIZE = 1024 * 1024
SUPPORTED_ESTIMATOR = "random-projection-nearest-centroid"


class TrainingError(ValueError):
    """Raised when baseline fitting cannot satisfy its isolation contract."""


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Versioned deterministic baseline configuration."""

    schema_version: int
    name: str
    version: str
    estimator: str
    seed: int
    projection_components: int


@dataclass(frozen=True, slots=True)
class FileDigest:
    """Repository-relative file identity."""

    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class BaselineModel:
    """Persistable random-projection nearest-centroid classifier."""

    schema_version: int
    estimator: str
    training_name: str
    training_version: str
    seed: int
    input_features: int
    projection_components: int
    classes: tuple[int, ...]
    projection: tuple[tuple[float, ...], ...]
    centroids: tuple[tuple[float, ...], ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True, allow_nan=False) + "\n"


@dataclass(frozen=True, slots=True)
class TrainingMetadata:
    """Fitting-only lineage without held-out metrics."""

    schema_version: int
    training_name: str
    training_version: str
    estimator: str
    seed: int
    partition: str
    record_ids: tuple[str, ...]
    record_count: int
    window_count: int
    target_value_counts: dict[str, int]
    dataset_index: FileDigest
    training_shards: tuple[FileDigest, ...]
    model: FileDigest

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True, slots=True)
class TrainingResult:
    model_path: Path
    metadata_path: Path
    window_count: int
    record_count: int


def load_training_config(path: Path) -> TrainingConfig:
    """Load a versioned baseline configuration."""
    try:
        with path.open("rb") as source:
            document = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise TrainingError(f"could not load training config {path}: {error}") from error
    values = document.get("training")
    if document.get("schema_version") != 1 or not isinstance(values, dict):
        raise TrainingError("training config must use schema_version = 1 and a [training] table")
    config = TrainingConfig(
        schema_version=1,
        name=_string(values, "name"),
        version=_string(values, "version"),
        estimator=_string(values, "estimator"),
        seed=_integer(values, "seed", minimum=0),
        projection_components=_integer(values, "projection_components", minimum=1),
    )
    if config.estimator != SUPPORTED_ESTIMATOR:
        raise TrainingError(f"training.estimator must be {SUPPORTED_ESTIMATOR!r}")
    return config


def train_from_index(
    repository_root: Path,
    dataset_index_path: Path,
    config: TrainingConfig,
    model_path: Path,
    metadata_path: Path,
) -> TrainingResult:
    """Fit and persist a baseline using only shards named by the train partition."""
    root = repository_root.resolve()
    index_path = _input_path(root, dataset_index_path, ("data", "processed"), "dataset index")
    try:
        document = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TrainingError(f"could not read dataset index {index_path}: {error}") from error
    if document.get("schema_version") != 1:
        raise TrainingError("dataset index must use schema_version 1")
    partitions = document.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != {"train", "validation", "test"}:
        raise TrainingError("dataset index must contain train, validation, and test partitions")
    train = partitions["train"]
    if not isinstance(train, dict) or not isinstance(train.get("shards"), list):
        raise TrainingError("dataset index train partition must contain shards")

    matrices: list[np.ndarray[Any, Any]] = []
    targets: list[np.ndarray[Any, Any]] = []
    records: list[str] = []
    shard_digests: list[FileDigest] = []
    for descriptor in train["shards"]:
        if not isinstance(descriptor, dict) or not isinstance(descriptor.get("file"), dict):
            raise TrainingError("training shard descriptor is invalid")
        record_id = descriptor.get("record_id")
        file_values = descriptor["file"]
        if not isinstance(record_id, str) or not record_id:
            raise TrainingError("training shard record_id must be a non-empty string")
        shard_path = _input_path(
            root, Path(str(file_values.get("path", ""))), ("data", "interim"), "training shard"
        )
        digest = _digest(root, shard_path)
        if digest.size_bytes != file_values.get("size_bytes") or digest.sha256 != file_values.get(
            "sha256"
        ):
            raise TrainingError(f"training shard digest does not match dataset index: {record_id}")
        windows, labels = _load_shard(shard_path, record_id)
        matrices.append(windows)
        targets.append(labels)
        records.append(record_id)
        shard_digests.append(digest)
    if not matrices:
        raise TrainingError("training partition must contain at least one shard")
    features = np.concatenate(matrices, axis=0)
    labels = np.concatenate(targets)
    expected_windows = train.get("window_count")
    if features.shape[0] != expected_windows:
        raise TrainingError("loaded training window count does not match dataset index")
    expected_records = train.get("record_count")
    if len(records) != expected_records or len(set(records)) != len(records):
        raise TrainingError("loaded training records do not match dataset index")
    counts = Counter(int(value) for value in labels)
    expected_counts = {str(key): value for key, value in sorted(counts.items())}
    if expected_counts != train.get("target_value_counts"):
        raise TrainingError("loaded training target counts do not match dataset index")
    if len(counts) < 2:
        raise TrainingError("baseline training requires at least two target classes")

    model = _fit(config, features, labels)
    resolved_model = _output_path(root, model_path, "model")
    resolved_metadata = _output_path(root, metadata_path, "training metadata")
    if resolved_model == resolved_metadata:
        raise TrainingError("model and training metadata outputs must be different")
    _write_new(resolved_model, model.to_json())
    model_digest = _digest(root, resolved_model)
    metadata = TrainingMetadata(
        schema_version=1,
        training_name=config.name,
        training_version=config.version,
        estimator=config.estimator,
        seed=config.seed,
        partition="train",
        record_ids=tuple(records),
        record_count=len(records),
        window_count=len(labels),
        target_value_counts=expected_counts,
        dataset_index=_digest(root, index_path),
        training_shards=tuple(shard_digests),
        model=model_digest,
    )
    try:
        _write_new(resolved_metadata, metadata.to_json())
    except Exception:
        resolved_model.unlink(missing_ok=True)
        raise
    return TrainingResult(resolved_model, resolved_metadata, len(labels), len(records))


def _fit(
    config: TrainingConfig, features: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]
) -> BaselineModel:
    rng = np.random.default_rng(config.seed)
    projection = rng.standard_normal((features.shape[1], config.projection_components))
    projection /= np.sqrt(config.projection_components)
    projected = features @ projection
    classes = tuple(int(value) for value in np.unique(labels))
    centroids = np.vstack([projected[labels == value].mean(axis=0) for value in classes])
    if not np.isfinite(centroids).all():
        raise TrainingError("fitted model contains non-finite parameters")
    return BaselineModel(
        schema_version=1,
        estimator=config.estimator,
        training_name=config.name,
        training_version=config.version,
        seed=config.seed,
        input_features=features.shape[1],
        projection_components=config.projection_components,
        classes=classes,
        projection=tuple(tuple(float(value) for value in row) for row in projection),
        centroids=tuple(tuple(float(value) for value in row) for row in centroids),
    )


def _load_shard(
    path: Path, expected_record: str
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    try:
        with np.load(path, allow_pickle=False) as artifact:
            windows = np.asarray(artifact["windows"])
            labels = np.asarray(artifact["target_values"])
            record_ids = np.asarray(artifact["record_ids"])
    except (BadZipFile, KeyError, OSError, ValueError) as error:
        raise TrainingError(f"could not load training shard {path}: {error}") from error
    if (
        windows.ndim != 2
        or windows.shape[0] == 0
        or windows.dtype.kind != "f"
        or not np.isfinite(windows).all()
    ):
        raise TrainingError("training windows must be a non-empty finite floating-point matrix")
    if labels.ndim != 1 or labels.dtype.kind not in {"i", "u"} or len(labels) != len(windows):
        raise TrainingError("training targets must be an integer vector aligned with windows")
    if (
        record_ids.ndim != 1
        or len(record_ids) != len(windows)
        or set(record_ids.tolist()) != {expected_record}
    ):
        raise TrainingError("training shard record lineage does not match the dataset index")
    return windows, labels


def _string(values: dict[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TrainingError(f"training.{key} must be a non-empty string")
    return value.strip()


def _integer(values: dict[str, Any], key: str, *, minimum: int) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise TrainingError(f"training.{key} must be an integer >= {minimum}")
    return value


def _input_path(root: Path, path: Path, prefix: tuple[str, ...], description: str) -> Path:
    candidate = path if path.is_absolute() else root / path
    if candidate.is_symlink():
        raise TrainingError(f"{description} must not be a symbolic link")
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise TrainingError(f"{description} must stay within repository root") from error
    if relative.parts[: len(prefix)] != prefix or not resolved.is_file():
        raise TrainingError(f"{description} must be a regular file under {'/'.join(prefix)}/")
    return resolved


def _output_path(root: Path, path: Path, description: str) -> Path:
    candidate = path if path.is_absolute() else root / path
    if candidate.is_symlink():
        raise TrainingError(f"{description} output must not be a symbolic link")
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise TrainingError(f"{description} output must stay within repository root") from error
    if relative.parts[:1] != ("artifacts",) or resolved.suffix != ".json":
        raise TrainingError(f"{description} output must be JSON under artifacts/")
    if not resolved.parent.is_dir():
        raise TrainingError(f"{description} output parent does not exist: {resolved.parent}")
    return resolved


def _write_new(path: Path, content: str) -> None:
    try:
        with path.open("x", encoding="utf-8") as output:
            output.write(content)
    except FileExistsError as error:
        raise TrainingError(f"training output already exists: {path}") from error
    except OSError as error:
        raise TrainingError(f"could not write training output {path}: {error}") from error


def _digest(root: Path, path: Path) -> FileDigest:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(BUFFER_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return FileDigest(path.relative_to(root).as_posix(), size, digest.hexdigest())
