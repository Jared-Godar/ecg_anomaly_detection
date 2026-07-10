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

# Centralize BUFFER_SIZE so every caller shares the same documented invariant.
BUFFER_SIZE = 1024 * 1024
# Centralize SUPPORTED_ESTIMATOR so every caller shares the same documented invariant.
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
        """Serialize this structured record as deterministic JSON.

        The helper isolates this step so its assumptions, outputs, and failure behavior remain
        reviewable.

        Returns:
            The value produced by the documented operation.
        """

        return json.dumps(asdict(self), indent=2, sort_keys=True, allow_nan=False) + "\n"


def load_baseline_model(path: Path) -> BaselineModel:
    """Load and strictly validate a persisted baseline without fitting it."""
    # Attempt this boundary operation here so (OSError, UnicodeError, json.JSONDecodeError) can be
    # translated or cleaned up under the repository contract.
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TrainingError(f"could not load baseline model {path}: {error}") from error
    # Evaluate `not isinstance(document, dict) or document.get('schema_version') != 1` explicitly so
    # invalid or alternate states follow the documented contract.
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise TrainingError("baseline model must be an object using schema_version 1")
    expected_fields = {
        "schema_version",
        "estimator",
        "training_name",
        "training_version",
        "seed",
        "input_features",
        "projection_components",
        "classes",
        "projection",
        "centroids",
    }
    # Evaluate `set(document) != expected_fields` explicitly so invalid or alternate states follow
    # the documented contract.
    if set(document) != expected_fields:
        raise TrainingError("baseline model fields do not match schema version 1")
    # Attempt this boundary operation here so (TypeError, ValueError) can be translated or cleaned
    # up under the repository contract.
    try:
        model = BaselineModel(
            schema_version=1,
            estimator=_model_string(document, "estimator"),
            training_name=_model_string(document, "training_name"),
            training_version=_model_string(document, "training_version"),
            seed=_model_integer(document, "seed", minimum=0),
            input_features=_model_integer(document, "input_features", minimum=1),
            projection_components=_model_integer(document, "projection_components", minimum=1),
            classes=_model_integer_vector(document, "classes"),
            projection=_model_matrix(document, "projection"),
            centroids=_model_matrix(document, "centroids"),
        )
    except (TypeError, ValueError) as error:
        raise TrainingError(f"malformed baseline model: {error}") from error
    # Evaluate `model.estimator != SUPPORTED_ESTIMATOR` explicitly so invalid or alternate states
    # follow the documented contract.
    if model.estimator != SUPPORTED_ESTIMATOR:
        raise TrainingError(f"baseline model estimator must be {SUPPORTED_ESTIMATOR!r}")
    # Evaluate `len(model.classes) < 2 or tuple(sorted(set(model.classes))) != model.classes`
    # explicitly so invalid or alternate states follow the documented contract.
    if len(model.classes) < 2 or tuple(sorted(set(model.classes))) != model.classes:
        raise TrainingError("baseline model classes must be at least two unique sorted integers")
    projection = np.asarray(model.projection, dtype=np.float64)
    centroids = np.asarray(model.centroids, dtype=np.float64)
    # Evaluate `projection.shape != (model.input_features, model.projection_components)` explicitly
    # so invalid or alternate states follow the documented contract.
    if projection.shape != (model.input_features, model.projection_components):
        raise TrainingError("baseline model projection shape does not match its metadata")
    # Evaluate `centroids.shape != (len(model.classes), model.projection_components)` explicitly so
    # invalid or alternate states follow the documented contract.
    if centroids.shape != (len(model.classes), model.projection_components):
        raise TrainingError("baseline model centroid shape does not match its classes")
    # Evaluate `not np.isfinite(projection).all() or not np.isfinite(centroids).all()` explicitly so
    # invalid or alternate states follow the documented contract.
    if not np.isfinite(projection).all() or not np.isfinite(centroids).all():
        raise TrainingError("baseline model parameters must be finite")
    return model


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
        """Serialize this structured record as deterministic JSON.

        The helper isolates this step so its assumptions, outputs, and failure behavior remain
        reviewable.

        Returns:
            The value produced by the documented operation.
        """

        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Report persisted training artifacts and the train-only data volume consumed.

    Orchestration uses this summary for progress and lineage without reopening the serialized
    model or its metadata.
    """

    model_path: Path
    metadata_path: Path
    window_count: int
    record_count: int


def load_training_config(path: Path) -> TrainingConfig:
    """Load a versioned baseline configuration."""
    # Attempt this boundary operation here so (OSError, tomllib.TOMLDecodeError) can be translated
    # or cleaned up under the repository contract.
    try:
        # Scope `path.open('rb')` here so resource cleanup occurs on both success and failure paths.
        with path.open("rb") as source:
            document = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise TrainingError(f"could not load training config {path}: {error}") from error
    values = document.get("training")
    # Evaluate `document.get('schema_version') != 1 or not isinstance(values, dict)` explicitly so
    # invalid or alternate states follow the documented contract.
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
    # Evaluate `config.estimator != SUPPORTED_ESTIMATOR` explicitly so invalid or alternate states
    # follow the documented contract.
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
    # Attempt this boundary operation here so (OSError, UnicodeError, json.JSONDecodeError) can be
    # translated or cleaned up under the repository contract.
    try:
        document = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TrainingError(f"could not read dataset index {index_path}: {error}") from error
    # Evaluate `document.get('schema_version') not in {1, 2}` explicitly so invalid or alternate
    # states follow the documented contract.
    if document.get("schema_version") not in {1, 2}:
        raise TrainingError("dataset index must use schema_version 1 or 2")
    partitions = document.get("partitions")
    # Evaluate `not isinstance(partitions, dict) or set(partitions) != {'train', 'validation',
    # 'test'}` explicitly so invalid or alternate states follow the documented contract.
    if not isinstance(partitions, dict) or set(partitions) != {"train", "validation", "test"}:
        raise TrainingError("dataset index must contain train, validation, and test partitions")
    train = partitions["train"]
    # Evaluate `not isinstance(train, dict) or not isinstance(train.get('shards'), list)` explicitly
    # so invalid or alternate states follow the documented contract.
    if not isinstance(train, dict) or not isinstance(train.get("shards"), list):
        raise TrainingError("dataset index train partition must contain shards")

    matrices: list[np.ndarray[Any, Any]] = []
    targets: list[np.ndarray[Any, Any]] = []
    records: list[str] = []
    shard_digests: list[FileDigest] = []
    # Iterate over `train['shards']` one item at a time so ordering, validation, and failure
    # attribution remain explicit.
    for descriptor in train["shards"]:
        # Evaluate `not isinstance(descriptor, dict) or not isinstance(descriptor.get('file'),
        # dict)` explicitly so invalid or alternate states follow the documented contract.
        if not isinstance(descriptor, dict) or not isinstance(descriptor.get("file"), dict):
            raise TrainingError("training shard descriptor is invalid")
        record_id = descriptor.get("record_id")
        file_values = descriptor["file"]
        # Evaluate `not isinstance(record_id, str) or not record_id` explicitly so invalid or
        # alternate states follow the documented contract.
        if not isinstance(record_id, str) or not record_id:
            raise TrainingError("training shard record_id must be a non-empty string")
        shard_path = _input_path(
            root, Path(str(file_values.get("path", ""))), ("data", "interim"), "training shard"
        )
        digest = _digest(root, shard_path)
        # Evaluate `digest.size_bytes != file_values.get('size_bytes') or digest.sha256 !=
        # file_values.get('sha256')` explicitly so invalid or alternate states follow the documented
        # contract.
        if digest.size_bytes != file_values.get("size_bytes") or digest.sha256 != file_values.get(
            "sha256"
        ):
            raise TrainingError(f"training shard digest does not match dataset index: {record_id}")
        windows, labels = _load_shard(shard_path, record_id)
        matrices.append(windows)
        targets.append(labels)
        records.append(record_id)
        shard_digests.append(digest)
    # Evaluate `not matrices` explicitly so invalid or alternate states follow the documented
    # contract.
    if not matrices:
        raise TrainingError("training partition must contain at least one shard")
    features = np.concatenate(matrices, axis=0)
    labels = np.concatenate(targets)
    expected_windows = train.get("window_count")
    # Evaluate `features.shape[0] != expected_windows` explicitly so invalid or alternate states
    # follow the documented contract.
    if features.shape[0] != expected_windows:
        raise TrainingError("loaded training window count does not match dataset index")
    expected_records = train.get("record_count")
    # Evaluate `len(records) != expected_records or len(set(records)) != len(records)` explicitly so
    # invalid or alternate states follow the documented contract.
    if len(records) != expected_records or len(set(records)) != len(records):
        raise TrainingError("loaded training records do not match dataset index")
    counts = Counter(int(value) for value in labels)
    expected_counts = {str(key): value for key, value in sorted(counts.items())}
    # Evaluate `expected_counts != train.get('target_value_counts')` explicitly so invalid or
    # alternate states follow the documented contract.
    if expected_counts != train.get("target_value_counts"):
        raise TrainingError("loaded training target counts do not match dataset index")
    # Evaluate `len(counts) < 2` explicitly so invalid or alternate states follow the documented
    # contract.
    if len(counts) < 2:
        raise TrainingError("baseline training requires at least two target classes")

    model = _fit(config, features, labels)
    resolved_model = _output_path(root, model_path, "model")
    resolved_metadata = _output_path(root, metadata_path, "training metadata")
    # Evaluate `resolved_model == resolved_metadata` explicitly so invalid or alternate states
    # follow the documented contract.
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
    # Attempt this boundary operation here so Exception can be translated or cleaned up under the
    # repository contract.
    try:
        _write_new(resolved_metadata, metadata.to_json())
    except Exception:
        resolved_model.unlink(missing_ok=True)
        raise
    return TrainingResult(resolved_model, resolved_metadata, len(labels), len(records))


def _fit(
    config: TrainingConfig, features: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]
) -> BaselineModel:
    """Fit the deterministic random-projection nearest-centroid baseline.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        config: Validated configuration controlling the documented operation.
        features: Feature matrix consumed by the deterministic model operation.
        labels: Target labels retained for validation or metric calculation.

    Returns:
        The value produced by the documented operation.
    """

    rng = np.random.default_rng(config.seed)
    projection = rng.standard_normal((features.shape[1], config.projection_components))
    # Johnson-Lindenstrauss scaling: preserves pairwise distances under the random projection.
    projection /= np.sqrt(config.projection_components)
    projected = features @ projection
    classes = tuple(int(value) for value in np.unique(labels))
    centroids = np.vstack([projected[labels == value].mean(axis=0) for value in classes])
    # Evaluate `not np.isfinite(centroids).all()` explicitly so invalid or alternate states follow
    # the documented contract.
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
    """Load shard according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        path: Filesystem path identifying the input or output under review.
        expected_record: The expected record value supplied by the caller or surrounding test fixture.

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
        raise TrainingError(f"could not load training shard {path}: {error}") from error
    # Evaluate `windows.ndim != 2 or windows.shape[0] == 0 or windows.dtype.kind != 'f' or (not
    # np.isfinite(windows).all())` explicitly so invalid or alternate states follow the documented
    # contract.
    if (
        windows.ndim != 2
        or windows.shape[0] == 0
        or windows.dtype.kind != "f"
        or not np.isfinite(windows).all()
    ):
        raise TrainingError("training windows must be a non-empty finite floating-point matrix")
    # Evaluate `labels.ndim != 1 or labels.dtype.kind not in {'i', 'u'} or len(labels) !=
    # len(windows)` explicitly so invalid or alternate states follow the documented contract.
    if labels.ndim != 1 or labels.dtype.kind not in {"i", "u"} or len(labels) != len(windows):
        raise TrainingError("training targets must be an integer vector aligned with windows")
    # Evaluate `record_ids.ndim != 1 or len(record_ids) != len(windows) or set(record_ids.tolist())
    # != {expected_record}` explicitly so invalid or alternate states follow the documented
    # contract.
    if (
        record_ids.ndim != 1
        or len(record_ids) != len(windows)
        or set(record_ids.tolist()) != {expected_record}
    ):
        raise TrainingError("training shard record lineage does not match the dataset index")
    return windows, labels


def _string(values: dict[str, Any], key: str) -> str:
    """Require and return a non-empty string from the requested structured field.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        values: Structured values to validate, transform, or serialize.
        key: The key value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    value = values.get(key)
    # Evaluate `not isinstance(value, str) or not value.strip()` explicitly so invalid or alternate
    # states follow the documented contract.
    if not isinstance(value, str) or not value.strip():
        raise TrainingError(f"training.{key} must be a non-empty string")
    return value.strip()


def _integer(values: dict[str, Any], key: str, *, minimum: int) -> int:
    """Require and return an integer from the requested structured field.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        values: Structured values to validate, transform, or serialize.
        key: The key value supplied by the caller or surrounding test fixture.
        minimum: The minimum value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    value = values.get(key)
    # Evaluate `not isinstance(value, int) or isinstance(value, bool) or value < minimum` explicitly
    # so invalid or alternate states follow the documented contract.
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise TrainingError(f"training.{key} must be an integer >= {minimum}")
    return value


def _model_string(values: dict[str, Any], key: str) -> str:
    """Compute and return model string for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        values: Structured values to validate, transform, or serialize.
        key: The key value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    value = values.get(key)
    # Evaluate `not isinstance(value, str) or not value` explicitly so invalid or alternate states
    # follow the documented contract.
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _model_integer(values: dict[str, Any], key: str, *, minimum: int) -> int:
    """Compute and return model integer for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        values: Structured values to validate, transform, or serialize.
        key: The key value supplied by the caller or surrounding test fixture.
        minimum: The minimum value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    value = values.get(key)
    # Evaluate `not isinstance(value, int) or isinstance(value, bool) or value < minimum` explicitly
    # so invalid or alternate states follow the documented contract.
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")
    return value


def _model_integer_vector(values: dict[str, Any], key: str) -> tuple[int, ...]:
    """Compute and return model integer vector for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        values: Structured values to validate, transform, or serialize.
        key: The key value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    value = values.get(key)
    # Evaluate `not isinstance(value, list) or any((not isinstance(item, int) or isinstance(item,
    # bool) for item in value))` explicitly so invalid or alternate states follow the documented
    # contract.
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise ValueError(f"{key} must be an integer array")
    return tuple(value)


def _model_matrix(values: dict[str, Any], key: str) -> tuple[tuple[float, ...], ...]:
    """Compute and return model matrix for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        values: Structured values to validate, transform, or serialize.
        key: The key value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    value = values.get(key)
    # Evaluate `not isinstance(value, list) or not value` explicitly so invalid or alternate states
    # follow the documented contract.
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty numeric matrix")
    rows: list[tuple[float, ...]] = []
    # Iterate over `value` one item at a time so ordering, validation, and failure attribution
    # remain explicit.
    for row in value:
        # Evaluate `not isinstance(row, list) or not row or any((not isinstance(item, (int, float))
        # or isinstance(item, bool) for item in...` explicitly so invalid or alternate states follow
        # the documented contract.
        if (
            not isinstance(row, list)
            or not row
            or any(not isinstance(item, (int, float)) or isinstance(item, bool) for item in row)
        ):
            raise ValueError(f"{key} must be a non-empty numeric matrix")
        rows.append(tuple(float(item) for item in row))
    # Evaluate `len({len(row) for row in rows}) != 1` explicitly so invalid or alternate states
    # follow the documented contract.
    if len({len(row) for row in rows}) != 1:
        raise ValueError(f"{key} rows must have equal length")
    return tuple(rows)


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
        raise TrainingError(f"{description} must not be a symbolic link")
    resolved = candidate.resolve()
    # Attempt this boundary operation here so ValueError can be translated or cleaned up under the
    # repository contract.
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise TrainingError(f"{description} must stay within repository root") from error
    # Evaluate `relative.parts[:len(prefix)] != prefix or not resolved.is_file()` explicitly so
    # invalid or alternate states follow the documented contract.
    if relative.parts[: len(prefix)] != prefix or not resolved.is_file():
        raise TrainingError(f"{description} must be a regular file under {'/'.join(prefix)}/")
    return resolved


def _output_path(root: Path, path: Path, description: str) -> Path:
    """Resolve and validate output path for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        root: Repository root used to enforce path and trust boundaries.
        path: Filesystem path identifying the input or output under review.
        description: The description value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    candidate = path if path.is_absolute() else root / path
    # Evaluate `candidate.is_symlink()` explicitly so invalid or alternate states follow the
    # documented contract.
    if candidate.is_symlink():
        raise TrainingError(f"{description} output must not be a symbolic link")
    resolved = candidate.resolve()
    # Attempt this boundary operation here so ValueError can be translated or cleaned up under the
    # repository contract.
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise TrainingError(f"{description} output must stay within repository root") from error
    # Evaluate `relative.parts[:1] != ('artifacts',) or resolved.suffix != '.json'` explicitly so
    # invalid or alternate states follow the documented contract.
    if relative.parts[:1] != ("artifacts",) or resolved.suffix != ".json":
        raise TrainingError(f"{description} output must be JSON under artifacts/")
    # Evaluate `not resolved.parent.is_dir()` explicitly so invalid or alternate states follow the
    # documented contract.
    if not resolved.parent.is_dir():
        raise TrainingError(f"{description} output parent does not exist: {resolved.parent}")
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
        raise TrainingError(f"training output already exists: {path}") from error
    except OSError as error:
        raise TrainingError(f"could not write training output {path}: {error}") from error


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

    digest = hashlib.sha256()
    size = 0
    # Scope `path.open('rb')` here so resource cleanup occurs on both success and failure paths.
    with path.open("rb") as source:
        # Continue while `(chunk := source.read(BUFFER_SIZE))` so the loop's termination rule
        # remains visible to reviewers.
        while chunk := source.read(BUFFER_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return FileDigest(path.relative_to(root).as_posix(), size, digest.hexdigest())
