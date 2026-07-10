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

# Chunk size for streaming file reads during digest computation. 1 MiB balances syscall
# overhead against peak memory for shard files that can be tens of megabytes.
BUFFER_SIZE = 1024 * 1024
# The only estimator this module currently implements; both the TOML config and any
# persisted model are validated against this exact string so a config or artifact that
# names a different (unsupported) estimator fails loudly instead of silently mismatching.
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

        allow_nan=False turns a NaN/inf parameter (which would otherwise serialize as
        non-standard JSON tokens few parsers accept) into an explicit error at write
        time, rather than persisting a model that would fail to round-trip elsewhere.

        Returns:
            The model as a JSON string with sorted, deterministic key ordering.
        """

        return json.dumps(asdict(self), indent=2, sort_keys=True, allow_nan=False) + "\n"


def load_baseline_model(path: Path) -> BaselineModel:
    """Load and strictly validate a persisted baseline without fitting it."""
    # Translate a missing, unreadable, or malformed-JSON file into TrainingError so
    # callers only need to catch one exception type for every load failure mode.
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TrainingError(f"could not load baseline model {path}: {error}") from error
    # This module has only ever written schema_version 1; any other value means the
    # file was produced by an incompatible tool or corrupted.
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
    # An exact field-set match (not just "at least these fields") catches both a
    # truncated file and one with unexpected extra keys from a future schema version.
    if set(document) != expected_fields:
        raise TrainingError("baseline model fields do not match schema version 1")
    # Collapse every per-field validation helper's TypeError/ValueError into one
    # TrainingError, so a malformed individual field (e.g. a non-numeric matrix row)
    # reports clearly instead of propagating a helper's raw exception type.
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
    # A model claiming an unsupported estimator can't be safely used for inference by
    # any code path that assumes the random-projection-nearest-centroid contract.
    if model.estimator != SUPPORTED_ESTIMATOR:
        raise TrainingError(f"baseline model estimator must be {SUPPORTED_ESTIMATOR!r}")
    # Nearest-centroid classification requires at least two distinct classes, and the
    # classes tuple is expected to be stored sorted-unique so downstream code (which
    # indexes centroids positionally by class) can rely on a stable ordering.
    if len(model.classes) < 2 or tuple(sorted(set(model.classes))) != model.classes:
        raise TrainingError("baseline model classes must be at least two unique sorted integers")
    projection = np.asarray(model.projection, dtype=np.float64)
    centroids = np.asarray(model.centroids, dtype=np.float64)
    # The projection matrix must map input_features to projection_components exactly;
    # a shape mismatch means the persisted metadata and the persisted matrix disagree.
    if projection.shape != (model.input_features, model.projection_components):
        raise TrainingError("baseline model projection shape does not match its metadata")
    # One centroid row per class, each in the projected (lower-dimensional) space.
    if centroids.shape != (len(model.classes), model.projection_components):
        raise TrainingError("baseline model centroid shape does not match its classes")
    # A NaN/inf parameter would silently corrupt every downstream distance calculation
    # rather than fail loudly, so it's rejected explicitly here.
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

        Returns:
            The metadata as a JSON string with sorted, deterministic key ordering.
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
    # Translate a missing, unreadable, or malformed-TOML file into TrainingError.
    try:
        # The `with` block ensures the file handle closes even if tomllib.load raises.
        with path.open("rb") as source:
            document = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise TrainingError(f"could not load training config {path}: {error}") from error
    values = document.get("training")
    # schema_version pins this loader's understanding of the [training] table's shape;
    # a mismatch or missing table means the config predates or postdates this loader.
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
    # This module only implements one estimator; a config naming any other one would
    # otherwise pass every structural check while requesting behavior that doesn't exist.
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
    # Translate a missing, unreadable, or malformed-JSON index into TrainingError.
    try:
        document = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TrainingError(f"could not read dataset index {index_path}: {error}") from error
    # dataset_index.py currently writes schema_version 1 or 2; anything else means this
    # index predates or postdates what this loader understands.
    if document.get("schema_version") not in {1, 2}:
        raise TrainingError("dataset index must use schema_version 1 or 2")
    partitions = document.get("partitions")
    # An index must always declare all three fixed partitions, even if this stage only
    # reads "train" -- their presence is itself evidence the index is well-formed.
    if not isinstance(partitions, dict) or set(partitions) != {"train", "validation", "test"}:
        raise TrainingError("dataset index must contain train, validation, and test partitions")
    train = partitions["train"]
    # This is the load-bearing boundary enforcing "training reads only the train
    # partition": only train["shards"] is ever touched below, never validation or test.
    if not isinstance(train, dict) or not isinstance(train.get("shards"), list):
        raise TrainingError("dataset index train partition must contain shards")

    matrices: list[np.ndarray[Any, Any]] = []
    targets: list[np.ndarray[Any, Any]] = []
    records: list[str] = []
    shard_digests: list[FileDigest] = []
    # Load every shard the index lists for "train", in the index's own declared order,
    # so the resulting feature matrix's row order is deterministic across runs.
    for descriptor in train["shards"]:
        # Every descriptor must carry a nested file object; a malformed descriptor here
        # would otherwise raise a bare KeyError further down instead of a clear message.
        if not isinstance(descriptor, dict) or not isinstance(descriptor.get("file"), dict):
            raise TrainingError("training shard descriptor is invalid")
        record_id = descriptor.get("record_id")
        file_values = descriptor["file"]
        # A shard is meaningless without knowing which record it belongs to.
        if not isinstance(record_id, str) or not record_id:
            raise TrainingError("training shard record_id must be a non-empty string")
        shard_path = _input_path(
            root, Path(str(file_values.get("path", ""))), ("data", "interim"), "training shard"
        )
        digest = _digest(root, shard_path)
        # Recompute the shard's digest from the file actually on disk and compare against
        # what the index recorded -- this catches a shard that changed (or was replaced)
        # after indexing, which would otherwise silently train on different data than the
        # index's own lineage claims.
        if digest.size_bytes != file_values.get("size_bytes") or digest.sha256 != file_values.get(
            "sha256"
        ):
            raise TrainingError(f"training shard digest does not match dataset index: {record_id}")
        windows, labels = _load_shard(shard_path, record_id)
        matrices.append(windows)
        targets.append(labels)
        records.append(record_id)
        shard_digests.append(digest)
    # Fitting with zero shards would otherwise silently produce a model with no data.
    if not matrices:
        raise TrainingError("training partition must contain at least one shard")
    features = np.concatenate(matrices, axis=0)
    labels = np.concatenate(targets)
    expected_windows = train.get("window_count")
    # Cross-check the actually-loaded row count against the index's own recorded count,
    # catching a shard that silently gained or lost rows since indexing.
    if features.shape[0] != expected_windows:
        raise TrainingError("loaded training window count does not match dataset index")
    expected_records = train.get("record_count")
    # Same cross-check at the record level, plus a duplicate-record guard: two shard
    # descriptors naming the same record would double-count that record's windows.
    if len(records) != expected_records or len(set(records)) != len(records):
        raise TrainingError("loaded training records do not match dataset index")
    counts = Counter(int(value) for value in labels)
    expected_counts = {str(key): value for key, value in sorted(counts.items())}
    # Same cross-check at the class-histogram level -- the strongest of the three
    # dataset-index cross-checks, since it depends on every label value being correct.
    if expected_counts != train.get("target_value_counts"):
        raise TrainingError("loaded training target counts do not match dataset index")
    # Nearest-centroid classification is undefined with fewer than two classes present.
    if len(counts) < 2:
        raise TrainingError("baseline training requires at least two target classes")

    model = _fit(config, features, labels)
    resolved_model = _output_path(root, model_path, "model")
    resolved_metadata = _output_path(root, metadata_path, "training metadata")
    # Writing both artifacts to the same path would make one silently overwrite (or
    # collide with) the other; _write_new's exclusive-create mode would then raise a
    # confusing "already exists" instead of this specific, actionable message.
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
    # If metadata fails to write after the model already did, remove the orphaned model
    # file rather than leaving a model on disk with no accompanying lineage record --
    # a bare `except Exception` here is intentional: any failure during this specific
    # write must trigger cleanup, not just the exception types anticipated in advance.
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

    Projects features into a lower-dimensional space via a seeded random projection
    (Johnson-Lindenstrauss), then computes one centroid per class in that projected
    space. Determinism comes entirely from config.seed: the same seed and features
    always produce bit-identical projection and centroid matrices, which is what makes
    this baseline reproducible across runs without needing to persist random state.

    Args:
        config: Training config supplying the seed and target projection dimensionality.
        features: The full training feature matrix (rows are windows, columns are
            raw signal samples).
        labels: Per-row target class labels, aligned with features' rows.

    Returns:
        The fitted model, including both the projection matrix and per-class centroids.
    """

    rng = np.random.default_rng(config.seed)
    projection = rng.standard_normal((features.shape[1], config.projection_components))
    # Johnson-Lindenstrauss scaling: preserves pairwise distances under the random projection.
    projection /= np.sqrt(config.projection_components)
    projected = features @ projection
    classes = tuple(int(value) for value in np.unique(labels))
    centroids = np.vstack([projected[labels == value].mean(axis=0) for value in classes])
    # A NaN/inf centroid would mean a class's projected features were somehow
    # non-finite; better to fail fitting outright than persist an unusable model.
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
    """Load and validate one training shard's windows and labels from an NPZ file.

    Args:
        path: Path to the shard NPZ file, already resolved and boundary-checked.
        expected_record: The record ID the dataset index claims this shard belongs to,
            cross-checked against the shard's own embedded record_ids field.

    Returns:
        The shard's window matrix and aligned target-label vector.
    """

    # allow_pickle=False is a security boundary against arbitrary code execution from an
    # untrusted NPZ file; collapse every load/parse failure mode into TrainingError.
    try:
        # The `with` block ensures the lazy NpzFile handle closes even if a field access
        # below raises.
        with np.load(path, allow_pickle=False) as artifact:
            windows = np.asarray(artifact["windows"])
            labels = np.asarray(artifact["target_values"])
            record_ids = np.asarray(artifact["record_ids"])
    except (BadZipFile, KeyError, OSError, ValueError) as error:
        raise TrainingError(f"could not load training shard {path}: {error}") from error
    # The window matrix must be a non-empty, finite 2-D float array -- a NaN/inf value
    # would silently corrupt the fit performed on it.
    if (
        windows.ndim != 2
        or windows.shape[0] == 0
        or windows.dtype.kind != "f"
        or not np.isfinite(windows).all()
    ):
        raise TrainingError("training windows must be a non-empty finite floating-point matrix")
    # Labels must be a 1-D integer vector with one entry per window row.
    if labels.ndim != 1 or labels.dtype.kind not in {"i", "u"} or len(labels) != len(windows):
        raise TrainingError("training targets must be an integer vector aligned with windows")
    # A shard is defined as one record's windows; this cross-checks the shard's own
    # embedded record_ids against what the dataset index claimed this shard contains,
    # catching a shard that was renamed or swapped after the index was built.
    if (
        record_ids.ndim != 1
        or len(record_ids) != len(windows)
        or set(record_ids.tolist()) != {expected_record}
    ):
        raise TrainingError("training shard record lineage does not match the dataset index")
    return windows, labels


def _string(values: dict[str, Any], key: str) -> str:
    """Require and return a non-empty string from the requested structured field.

    Used for [training] TOML fields; raises with a `training.{key}` error prefix
    matching TOML's dotted-path convention.

    Args:
        values: The parsed `[training]` table to read from.
        key: The field name to extract.

    Returns:
        The field's value with surrounding whitespace stripped.
    """

    value = values.get(key)
    # Reject a missing/wrong-typed value and a whitespace-only placeholder alike.
    if not isinstance(value, str) or not value.strip():
        raise TrainingError(f"training.{key} must be a non-empty string")
    return value.strip()


def _integer(values: dict[str, Any], key: str, *, minimum: int) -> int:
    """Require and return an integer at or above a minimum from a structured field.

    Used for [training] TOML fields; raises with a `training.{key}` error prefix.

    Args:
        values: The parsed `[training]` table to read from.
        key: The field name to extract.
        minimum: The smallest value considered valid (inclusive).

    Returns:
        The field's integer value.
    """

    value = values.get(key)
    # bool is an int subclass in Python, so it's excluded explicitly.
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise TrainingError(f"training.{key} must be an integer >= {minimum}")
    return value


def _model_string(values: dict[str, Any], key: str) -> str:
    """Require and return a non-empty string from a persisted baseline model document.

    Structurally similar to _string, but raises plain ValueError (not TrainingError)
    since load_baseline_model wraps every field-parser call in one try/except that
    re-raises as TrainingError with a unified "malformed baseline model" prefix.

    Args:
        values: The parsed model JSON document to read from.
        key: The field name to extract.

    Returns:
        The field's string value, unstripped (model JSON is machine-written, not
        hand-edited TOML, so no leading/trailing whitespace is expected or tolerated).
    """

    value = values.get(key)
    # Unlike TOML-sourced fields, no .strip() here: model JSON is machine-written.
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _model_integer(values: dict[str, Any], key: str, *, minimum: int) -> int:
    """Require and return an integer at or above a minimum from a baseline model document.

    Args:
        values: The parsed model JSON document to read from.
        key: The field name to extract.
        minimum: The smallest value considered valid (inclusive).

    Returns:
        The field's integer value.
    """

    value = values.get(key)
    # bool is an int subclass in Python, so it's excluded explicitly.
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")
    return value


def _model_integer_vector(values: dict[str, Any], key: str) -> tuple[int, ...]:
    """Require and return a vector of plain integers from a baseline model document.

    Used for the `classes` field; each element is validated individually rather than
    trusting the JSON array's apparent homogeneity, since JSON has no native distinction
    between int and float and a stray `1.0` would otherwise slip through as a class label.

    Args:
        values: The parsed model JSON document to read from.
        key: The field name to extract.

    Returns:
        The field's values as a tuple of Python ints.
    """

    value = values.get(key)
    # Each element is checked individually rather than trusting the list's apparent
    # homogeneity, since JSON has no int/float distinction and a stray float would
    # otherwise slip through.
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise ValueError(f"{key} must be an integer array")
    return tuple(value)


def _model_matrix(values: dict[str, Any], key: str) -> tuple[tuple[float, ...], ...]:
    """Require and return a rectangular numeric matrix from a baseline model document.

    Used for both `projection` and `centroids`; validates every row is non-empty,
    purely numeric, and that all rows share one consistent length, since a ragged
    "matrix" would break the shape checks load_baseline_model performs afterward.

    Args:
        values: The parsed model JSON document to read from.
        key: The field name to extract.

    Returns:
        The field's values as a tuple of equal-length float tuples.
    """

    value = values.get(key)
    # An empty or non-list value can't be a matrix at all.
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty numeric matrix")
    rows: list[tuple[float, ...]] = []
    # Validate and convert each row independently so a malformed row anywhere in the
    # matrix is caught, not just the first.
    for row in value:
        # Each row must itself be a non-empty list of plain numbers (int or float, but
        # not bool, which JSON has no distinct type for but Python would otherwise accept).
        if (
            not isinstance(row, list)
            or not row
            or any(not isinstance(item, (int, float)) or isinstance(item, bool) for item in row)
        ):
            raise ValueError(f"{key} must be a non-empty numeric matrix")
        rows.append(tuple(float(item) for item in row))
    # A ragged matrix (rows of different lengths) couldn't be interpreted as a
    # consistent feature-by-component or class-by-component array downstream.
    if len({len(row) for row in rows}) != 1:
        raise ValueError(f"{key} rows must have equal length")
    return tuple(rows)


def _input_path(root: Path, path: Path, prefix: tuple[str, ...], description: str) -> Path:
    """Resolve a path and enforce it stays within a required subdirectory of the repo.

    Shared by every stage in this module that reads a repository-relative input (the
    dataset index under data/processed/, training shards under data/interim/): resolving
    through symlinks and checking containment prevents a maliciously or accidentally
    crafted path from reading files outside the repository the pipeline operates on.

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
        raise TrainingError(f"{description} must not be a symbolic link")
    resolved = candidate.resolve()
    # relative_to raises ValueError when resolved escapes root (e.g. via `..` segments);
    # translate that into this module's own exception type.
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise TrainingError(f"{description} must stay within repository root") from error
    # Confirm both containment under the expected subtree and that it's a regular file
    # (not a directory or special file) in one combined check.
    if relative.parts[: len(prefix)] != prefix or not resolved.is_file():
        raise TrainingError(f"{description} must be a regular file under {'/'.join(prefix)}/")
    return resolved


def _output_path(root: Path, path: Path, description: str) -> Path:
    """Resolve a path and enforce it's a JSON output under the shared artifacts/ tree.

    Shared by both the model and training-metadata outputs; unlike _input_path (whose
    required prefix varies per call), every training output lands under artifacts/,
    matching this repository's directory contract for pipeline-generated evidence.

    Args:
        root: Repository root used to enforce path and trust boundaries.
        path: The candidate output path, absolute or relative to root.
        description: Human-readable label for this output, used in error messages.

    Returns:
        The resolved, validated absolute path.
    """

    candidate = path if path.is_absolute() else root / path
    # Reject a symlink before resolving it: resolving would silently follow the link and
    # write to wherever it points, defeating the repository-root containment check below.
    if candidate.is_symlink():
        raise TrainingError(f"{description} output must not be a symbolic link")
    resolved = candidate.resolve()
    # relative_to raises ValueError when resolved escapes root.
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise TrainingError(f"{description} output must stay within repository root") from error
    # Every training output is JSON under artifacts/, no exceptions.
    if relative.parts[:1] != ("artifacts",) or resolved.suffix != ".json":
        raise TrainingError(f"{description} output must be JSON under artifacts/")
    # Fail before attempting the write rather than letting a missing parent directory
    # surface as a generic OSError from _write_new.
    if not resolved.parent.is_dir():
        raise TrainingError(f"{description} output parent does not exist: {resolved.parent}")
    return resolved


def _write_new(path: Path, content: str) -> None:
    """Write content to a path that must not already exist.

    Used for both the model and metadata outputs, so a re-run can never silently
    overwrite a previous run's artifacts -- each output path is expected to be unique
    per run (typically via a run-ID-scoped directory chosen by the caller).

    Args:
        path: Destination path; must not already exist.
        content: The text to write.
    """

    # Collapse "already exists" and other OS-level failures into TrainingError.
    try:
        # Open with mode "x" (exclusive create) so this can never silently overwrite.
        with path.open("x", encoding="utf-8") as output:
            output.write(content)
    except FileExistsError as error:
        raise TrainingError(f"training output already exists: {path}") from error
    except OSError as error:
        raise TrainingError(f"could not write training output {path}: {error}") from error


def _digest(root: Path, path: Path) -> FileDigest:
    """Calculate stable size and SHA-256 evidence for one repository file.

    Args:
        root: Repository root, used to store the digest's path relative rather than
            absolute (so manifests remain portable across checkouts).
        path: The already-resolved, validated file path to hash.

    Returns:
        The file's repository-relative path, size, and SHA-256 digest.
    """

    digest = hashlib.sha256()
    size = 0
    # Read in fixed-size chunks rather than the whole file at once, since shard and
    # model files can be tens of megabytes.
    with path.open("rb") as source:
        # The walrus operator lets both the read and the loop's termination condition
        # (an empty final chunk) live in one line without a separate `break`.
        while chunk := source.read(BUFFER_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return FileDigest(path.relative_to(root).as_posix(), size, digest.hexdigest())
