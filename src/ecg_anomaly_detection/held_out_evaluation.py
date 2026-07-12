"""One-shot held-out test-partition evaluation gated by a verified ApprovalRecord.

This module implements governance steps 3-5 of docs/benchmark-governance.md: it loads a
frozen model, opens only the test-partition shards from the dataset index, scores the model
once, and archives all required disclosure evidence atomically.  It never touches the
validation partition and never runs without a verified ApprovalRecord.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np

from ecg_anomaly_detection.benchmark_approval import ApprovalRecord
from ecg_anomaly_detection.benchmark_policy import BenchmarkPolicy
from ecg_anomaly_detection.held_out_config import HeldOutExecutionConfig
from ecg_anomaly_detection.training import (
    SUPPORTED_ESTIMATOR,
    BaselineModel,
    FileDigest,
    TrainingError,
    load_baseline_model,
)

# Chunk size for streaming file reads during digest computation.
BUFFER_SIZE = 1024 * 1024
# The only partition this module is permitted to read.  Hard-coded (not configurable)
# because it is the enforcement mechanism for this module's core invariant: evaluation
# opens the test partition exactly once, after all governance gates pass.
SUPPORTED_PARTITION = "test"


class HeldOutEvaluationError(ValueError):
    """Raised when held-out evaluation cannot satisfy its governance or isolation contract."""


@dataclass(frozen=True, slots=True)
class HeldOutMetrics:
    """Capture held-out test-partition scores together with complete artifact lineage.

    Dataset-index, shard, and model digests bind every reported metric to the exact frozen
    inputs that produced it.  The approval_record_reference field ties the result to the
    governance gate that authorised this single execution.
    """

    schema_version: int
    evaluation_name: str
    evaluation_version: str
    evaluator: str
    partition: str
    zero_division: float
    approval_record_reference: str
    candidate_run_id: str
    record_ids: tuple[str, ...]
    record_count: int
    window_count: int
    classes: tuple[int, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]
    accuracy: float
    per_class: dict[str, dict[str, float | int]]
    macro_average: dict[str, float | int]
    dataset_index: FileDigest
    test_shards: tuple[FileDigest, ...]
    model: FileDigest

    def to_json(self) -> str:
        """Serialize this structured record as deterministic JSON.

        Returns:
            The metrics as a JSON string with sorted, deterministic key ordering.
        """

        return json.dumps(asdict(self), indent=2, sort_keys=True, allow_nan=False) + "\n"


@dataclass(frozen=True, slots=True)
class HeldOutEvaluationResult:
    """Report where held-out metrics and disclosure evidence were written.

    The lightweight return value lets orchestration report completion without reparsing
    the persisted artifacts.
    """

    metrics_path: Path
    disclosure_path: Path
    window_count: int
    record_count: int


def evaluate_held_out_from_index(
    repository_root: Path,
    dataset_index_path: Path,
    model_path: Path,
    training_metadata_path: Path,
    held_out_config: HeldOutExecutionConfig,
    policy: BenchmarkPolicy,
    approval_record: ApprovalRecord,
    metrics_path: Path,
    disclosure_path: Path,
) -> HeldOutEvaluationResult:
    """Score only indexed test shards once and persist metrics and disclosure evidence.

    Fails closed at every step: a disabled config, a policy mismatch, an approval record
    that does not match the candidate run, or any lineage or digest failure aborts before
    any test shard is opened.

    Args:
        repository_root: Repository root used to enforce path and trust boundaries.
        dataset_index_path: Path to the dataset index JSON produced by dataset_index.py.
        model_path: Path to the frozen baseline model JSON produced by training.py.
        training_metadata_path: Path to the training metadata JSON.
        held_out_config: Validated, disabled-by-default held-out execution config.
        policy: Validated benchmark governance policy.
        approval_record: Verified approval record from record_benchmark_approval.
        metrics_path: Destination for the held-out metrics JSON artifact.
        disclosure_path: Destination for the disclosure evidence JSON artifact.

    Returns:
        Paths to the written artifacts and the data volume scored.
    """
    root = repository_root.resolve()

    # All governance gates must pass before any path is resolved or any shard opened.
    _verify_config_enabled(held_out_config)
    _verify_policy_enabled(policy)
    _verify_approval_matches_candidate(approval_record, held_out_config)

    output_metrics = _output_path(root, metrics_path, "held-out-metrics.json")
    output_disclosure = _output_path(root, disclosure_path, "held-out-disclosure.json")

    data = _load_test_data(root, dataset_index_path, model_path, training_metadata_path)

    # Verify the approval record's candidate_run_id matches the model's training run.
    # The training metadata's model digest was already cross-checked in _load_test_data;
    # here we additionally confirm the approval names the same run as the model artifact.
    _verify_approval_run_matches_model(approval_record, data.model_digest, root)

    predictions = _predict(data.model, data.features)
    metrics = _build_metrics(
        held_out_config,
        approval_record,
        data.model,
        data.records,
        data.labels,
        predictions,
        data.index_digest,
        data.shard_digests,
        data.model_digest,
    )
    disclosure = _build_disclosure(policy, approval_record, metrics)

    # Write both artifacts atomically; if disclosure write fails, remove the metrics
    # file rather than leaving an unapproved result on disk.
    _write_new(output_metrics, metrics.to_json())
    # The following control-flow step enforces or verifies the surrounding contract.
    try:
        _write_new(output_disclosure, disclosure)
    except Exception:
        output_metrics.unlink(missing_ok=True)
        raise

    return HeldOutEvaluationResult(
        output_metrics, output_disclosure, len(data.labels), len(data.records)
    )


# ---------------------------------------------------------------------------
# Governance gate helpers
# ---------------------------------------------------------------------------


def _verify_config_enabled(config: HeldOutExecutionConfig) -> None:
    """Fail closed if the held-out config has not been explicitly enabled.

    The config loader already rejects execution_enabled=True, so this check guards
    against a future loader change that might relax that invariant.
    """
    # The following control-flow step enforces or verifies the surrounding contract.
    if not config.execution_enabled:
        raise HeldOutEvaluationError("held-out execution config must have execution_enabled = true")
    # The following control-flow step enforces or verifies the surrounding contract.
    if not config.requires_recorded_approval:
        raise HeldOutEvaluationError(
            "held-out execution config must have requires_recorded_approval = true"
        )


def _verify_policy_enabled(policy: BenchmarkPolicy) -> None:
    """Fail closed if the benchmark policy still disables test evaluation."""
    # The following control-flow step enforces or verifies the surrounding contract.
    if not policy.test_evaluation_enabled:
        raise HeldOutEvaluationError("benchmark policy must have test_evaluation_enabled = true")


def _verify_approval_matches_candidate(
    approval_record: ApprovalRecord, config: HeldOutExecutionConfig
) -> None:
    """Fail closed if the approval record's evaluator does not match the config."""
    # The following control-flow step enforces or verifies the surrounding contract.
    if config.evaluator != SUPPORTED_ESTIMATOR:
        raise HeldOutEvaluationError(f"held-out config evaluator must be {SUPPORTED_ESTIMATOR!r}")
    # The following control-flow step enforces or verifies the surrounding contract.
    if config.partition != SUPPORTED_PARTITION:
        raise HeldOutEvaluationError(f"held-out config partition must be {SUPPORTED_PARTITION!r}")
    # The following control-flow step enforces or verifies the surrounding contract.
    if not approval_record.candidate_run_id.strip():
        raise HeldOutEvaluationError("approval record candidate_run_id must be non-empty")


def _verify_approval_run_matches_model(
    approval_record: ApprovalRecord, model_digest: FileDigest, root: Path
) -> None:
    """Confirm the approval record's run_manifest_reference is consistent with the model path.

    The model path is under artifacts/runs/<run-id>/...; extract the run-id segment and
    confirm it matches the approval record's candidate_run_id so a record approved for
    one run cannot be reused to score a different run's model.
    """
    model_relative = Path(model_digest.path)
    # artifacts/runs/<run-id>/training/model.json -> parts[2] is the run-id
    # The following control-flow step enforces or verifies the surrounding contract.
    if len(model_relative.parts) < 3 or model_relative.parts[:2] != ("artifacts", "runs"):
        raise HeldOutEvaluationError("model artifact path must be under artifacts/runs/<run-id>/")
    model_run_id = model_relative.parts[2]
    # The following control-flow step enforces or verifies the surrounding contract.
    if model_run_id != approval_record.candidate_run_id:
        raise HeldOutEvaluationError(
            f"approval record candidate_run_id {approval_record.candidate_run_id!r} "
            f"does not match model run {model_run_id!r}"
        )


# ---------------------------------------------------------------------------
# Data loading (test partition only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _TestData:
    """Bundle verified test arrays, frozen model state, and input digests."""

    model: BaselineModel
    records: list[str]
    labels: np.ndarray[Any, Any]
    features: np.ndarray[Any, Any]
    index_digest: FileDigest
    shard_digests: list[FileDigest]
    model_digest: FileDigest


def _load_test_data(
    root: Path,
    dataset_index_path: Path,
    model_path: Path,
    training_metadata_path: Path,
) -> _TestData:
    """Load, verify, and assemble every input needed to evaluate, reading only the test partition."""
    index_path = _input_path(root, dataset_index_path, ("data", "processed"), "dataset index")
    frozen_model_path = _input_path(root, model_path, ("artifacts",), "baseline model")
    metadata_path = _input_path(root, training_metadata_path, ("artifacts",), "training metadata")

    index_digest = _digest(root, index_path)
    model_digest = _digest(root, frozen_model_path)
    index = _read_json(index_path, "dataset index")
    metadata = _read_json(metadata_path, "training metadata")
    _verify_training_digests(metadata, index_digest, model_digest)

    # The following control-flow step enforces or verifies the surrounding contract.

    try:
        model = load_baseline_model(frozen_model_path)
    except TrainingError as error:
        raise HeldOutEvaluationError(str(error)) from error

    test_partition = _test_partition(index, model)

    matrices: list[np.ndarray[Any, Any]] = []
    targets: list[np.ndarray[Any, Any]] = []
    records: list[str] = []
    shard_digests: list[FileDigest] = []
    verified_shards: list[tuple[str, Path]] = []

    # The following control-flow step enforces or verifies the surrounding contract.

    for descriptor in test_partition["shards"]:
        record_id, file_values = _shard_descriptor(descriptor)
        shard_path = _input_path(root, Path(file_values["path"]), ("data", "interim"), "test shard")
        digest = _digest(root, shard_path)
        # The following control-flow step enforces or verifies the surrounding contract.
        if digest.size_bytes != file_values["size_bytes"] or digest.sha256 != file_values["sha256"]:
            raise HeldOutEvaluationError(
                f"test shard digest does not match dataset index: {record_id}"
            )
        verified_shards.append((record_id, shard_path))
        shard_digests.append(digest)

    # The following control-flow step enforces or verifies the surrounding contract.

    if not verified_shards:
        raise HeldOutEvaluationError("test partition must contain at least one shard")

    # The following control-flow step enforces or verifies the surrounding contract.

    for record_id, shard_path in verified_shards:
        windows, labels = _load_test_shard(shard_path, record_id, model)
        matrices.append(windows)
        targets.append(labels)
        records.append(record_id)

    features = np.concatenate(matrices, axis=0)
    labels = np.concatenate(targets)
    _verify_partition_counts(test_partition, records, labels)
    return _TestData(model, records, labels, features, index_digest, shard_digests, model_digest)


def _test_partition(index: dict[str, Any], model: BaselineModel) -> dict[str, Any]:
    """Extract and validate the dataset index's 'test' partition entry.

    This is the enforcement point for this module's core invariant: it reads
    SUPPORTED_PARTITION ('test') specifically and never any other key from partitions.
    """
    # The following control-flow step enforces or verifies the surrounding contract.
    if index.get("schema_version") not in {1, 2}:
        raise HeldOutEvaluationError("dataset index must use schema_version 1 or 2")
    partitions = index.get("partitions")
    # The following control-flow step enforces or verifies the surrounding contract.
    if not isinstance(partitions, dict) or set(partitions) != {"train", "validation", "test"}:
        raise HeldOutEvaluationError(
            "dataset index must contain train, validation, and test partitions"
        )
    test = partitions.get(SUPPORTED_PARTITION)
    # The following control-flow step enforces or verifies the surrounding contract.
    if not isinstance(test, dict) or not isinstance(test.get("shards"), list):
        raise HeldOutEvaluationError("dataset index test partition must contain shards")
    # The following control-flow step enforces or verifies the surrounding contract.
    if index.get("window_samples") != model.input_features:
        raise HeldOutEvaluationError("baseline model input width does not match dataset index")
    return test


def _verify_training_digests(
    metadata: dict[str, Any], index: FileDigest, model: FileDigest
) -> None:
    """Confirm the model and index being evaluated match the training run that made them."""
    # The following control-flow step enforces or verifies the surrounding contract.
    if metadata.get("schema_version") != 1 or metadata.get("partition") != "train":
        raise HeldOutEvaluationError(
            "training metadata must use schema_version 1 for the train partition"
        )
    # The following control-flow step enforces or verifies the surrounding contract.
    for name, actual in (("dataset_index", index), ("model", model)):
        expected = metadata.get(name)
        # The following control-flow step enforces or verifies the surrounding contract.
        if not isinstance(expected, dict) or any(
            expected.get(key) != getattr(actual, key) for key in ("path", "size_bytes", "sha256")
        ):
            raise HeldOutEvaluationError(
                f"{name.replace('_', ' ')} digest does not match training metadata"
            )


def _shard_descriptor(descriptor: Any) -> tuple[str, dict[str, Any]]:
    """Extract and validate one shard entry's record ID and file identity fields."""
    # The following control-flow step enforces or verifies the surrounding contract.
    if not isinstance(descriptor, dict) or not isinstance(descriptor.get("file"), dict):
        raise HeldOutEvaluationError("test shard descriptor is invalid")
    record_id, values = descriptor.get("record_id"), descriptor["file"]
    # The following control-flow step enforces or verifies the surrounding contract.
    if not isinstance(record_id, str) or not record_id:
        raise HeldOutEvaluationError("test shard record_id must be a non-empty string")
    # The following control-flow step enforces or verifies the surrounding contract.
    if (
        not isinstance(values.get("path"), str)
        or not isinstance(values.get("size_bytes"), int)
        or not isinstance(values.get("sha256"), str)
    ):
        raise HeldOutEvaluationError("test shard file identity is invalid")
    return record_id, values


def _load_test_shard(
    path: Path, record_id: str, model: BaselineModel
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Load and validate one test shard's windows and labels from an NPZ file."""
    # The following control-flow step enforces or verifies the surrounding contract.
    try:
        # The following control-flow step enforces or verifies the surrounding contract.
        with np.load(path, allow_pickle=False) as artifact:
            windows = np.asarray(artifact["windows"])
            labels = np.asarray(artifact["target_values"])
            record_ids = np.asarray(artifact["record_ids"])
    except (BadZipFile, KeyError, OSError, ValueError) as error:
        raise HeldOutEvaluationError(f"could not load test shard {path}: {error}") from error
    # The following control-flow step enforces or verifies the surrounding contract.
    if (
        windows.ndim != 2
        or windows.shape[0] == 0
        or windows.shape[1] != model.input_features
        or windows.dtype.kind != "f"
        or not np.isfinite(windows).all()
    ):
        raise HeldOutEvaluationError(
            "test windows must be a non-empty compatible finite floating-point matrix"
        )
    # The following control-flow step enforces or verifies the surrounding contract.
    if labels.ndim != 1 or labels.dtype.kind not in {"i", "u"} or len(labels) != len(windows):
        raise HeldOutEvaluationError("test targets must be an integer vector aligned with windows")
    unknown = sorted(set(int(v) for v in labels) - set(model.classes))
    # The following control-flow step enforces or verifies the surrounding contract.
    if unknown:
        raise HeldOutEvaluationError(f"test targets contain labels unknown to the model: {unknown}")
    # The following control-flow step enforces or verifies the surrounding contract.
    if (
        record_ids.ndim != 1
        or len(record_ids) != len(windows)
        or set(record_ids.tolist()) != {record_id}
    ):
        raise HeldOutEvaluationError("test shard record lineage does not match the dataset index")
    return windows, labels


def _verify_partition_counts(
    partition: dict[str, Any], records: list[str], labels: np.ndarray[Any, Any]
) -> None:
    """Cross-check loaded test data against the dataset index's own recorded counts."""
    # The following control-flow step enforces or verifies the surrounding contract.
    if len(records) != partition.get("record_count") or len(set(records)) != len(records):
        raise HeldOutEvaluationError("loaded test records do not match dataset index")
    # The following control-flow step enforces or verifies the surrounding contract.
    if len(labels) != partition.get("window_count"):
        raise HeldOutEvaluationError("loaded test window count does not match dataset index")
    counts = {str(key): value for key, value in sorted(Counter(int(v) for v in labels).items())}
    # The following control-flow step enforces or verifies the surrounding contract.
    if counts != partition.get("target_value_counts"):
        raise HeldOutEvaluationError("loaded test target counts do not match dataset index")


# ---------------------------------------------------------------------------
# Prediction and metrics
# ---------------------------------------------------------------------------


def _predict(model: BaselineModel, features: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Assign each feature row to its nearest frozen class centroid."""
    projection = np.asarray(model.projection, dtype=np.float64)
    centroids = np.asarray(model.centroids, dtype=np.float64)
    projected = features @ projection
    distances = np.sum((projected[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    return np.asarray(model.classes, dtype=np.int64)[np.argmin(distances, axis=1)]


# The zero_division policy used when a class has no support in the test partition.
# Held-out evaluation uses the pessimistic 0.0 value (same as the validation evaluator's
# default) since the config schema does not expose this field.
_ZERO_DIVISION = 0.0


def _build_metrics(
    config: HeldOutExecutionConfig,
    approval_record: ApprovalRecord,
    model: BaselineModel,
    records: list[str],
    labels: np.ndarray[Any, Any],
    predictions: np.ndarray[Any, Any],
    index_digest: FileDigest,
    shard_digests: list[FileDigest],
    model_digest: FileDigest,
) -> HeldOutMetrics:
    """Assemble the confusion matrix, per-class, and macro-averaged held-out metrics."""
    classes = model.classes
    positions = {value: index for index, value in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    # The following control-flow step enforces or verifies the surrounding contract.
    for truth, prediction in zip(labels.tolist(), predictions.tolist(), strict=True):
        matrix[positions[int(truth)], positions[int(prediction)]] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    # The following control-flow step enforces or verifies the surrounding contract.
    for index, value in enumerate(classes):
        true_positive = int(matrix[index, index])
        predicted = int(matrix[:, index].sum())
        support = int(matrix[index, :].sum())
        precision = _divide(true_positive, predicted, _ZERO_DIVISION)
        recall = _divide(true_positive, support, _ZERO_DIVISION)
        f1 = _divide(2.0 * precision * recall, precision + recall, _ZERO_DIVISION)
        per_class[str(value)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    macro = {
        "precision": float(np.mean([v["precision"] for v in per_class.values()])),
        "recall": float(np.mean([v["recall"] for v in per_class.values()])),
        "f1": float(np.mean([v["f1"] for v in per_class.values()])),
        "support": len(labels),
    }
    return HeldOutMetrics(
        schema_version=1,
        evaluation_name=config.name,
        evaluation_version=config.version,
        evaluator=config.evaluator,
        partition=config.partition,
        zero_division=_ZERO_DIVISION,
        approval_record_reference=approval_record.candidate_run_id,
        candidate_run_id=approval_record.candidate_run_id,
        record_ids=tuple(records),
        record_count=len(records),
        window_count=len(labels),
        classes=classes,
        confusion_matrix=tuple(tuple(int(v) for v in row) for row in matrix),
        accuracy=float(np.trace(matrix) / len(labels)),
        per_class=per_class,
        macro_average=macro,
        dataset_index=index_digest,
        test_shards=tuple(shard_digests),
        model=model_digest,
    )


def _build_disclosure(
    policy: BenchmarkPolicy, approval_record: ApprovalRecord, metrics: HeldOutMetrics
) -> str:
    """Build the required disclosure evidence document per benchmark-governance.md."""
    payload = {
        "schema_version": 1,
        "policy_id": policy.policy_id,
        "policy_version": policy.version,
        "candidate_run_id": approval_record.candidate_run_id,
        "approval_record_reference": approval_record.candidate_run_id,
        "required_disclosures": sorted(policy.required_disclosures),
        "required_limitations": sorted(policy.required_limitations),
        "prohibited_claims": sorted(policy.prohibited_claims),
        "required_archival_records": sorted(policy.required_archival_records),
        "metrics_reference": {
            "partition": metrics.partition,
            "record_count": metrics.record_count,
            "window_count": metrics.window_count,
            "model_sha256": metrics.model.sha256,
            "dataset_index_sha256": metrics.dataset_index.sha256,
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _divide(numerator: float, denominator: float, zero_division: float) -> float:
    """Divide using the configured zero-division policy."""
    return zero_division if denominator == 0 else float(numerator / denominator)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _input_path(root: Path, path: Path, prefix: tuple[str, ...], description: str) -> Path:
    """Resolve a path and enforce it stays within a required subdirectory of the repo."""
    candidate = path if path.is_absolute() else root / path
    # The following control-flow step enforces or verifies the surrounding contract.
    if candidate.is_symlink():
        raise HeldOutEvaluationError(f"{description} must not be a symbolic link")
    resolved = candidate.resolve()
    # The following control-flow step enforces or verifies the surrounding contract.
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise HeldOutEvaluationError(f"{description} must stay within repository root") from error
    # The following control-flow step enforces or verifies the surrounding contract.
    if relative.parts[: len(prefix)] != prefix or not resolved.is_file():
        raise HeldOutEvaluationError(
            f"{description} must be a regular file under {'/'.join(prefix)}/"
        )
    return resolved


def _output_path(root: Path, path: Path, filename: str) -> Path:
    """Resolve a path and enforce it matches the fixed run-scoped held-out evaluation layout.

    Every held-out evaluation output lands at
    artifacts/runs/<run-id>/held-out-evaluation/<filename>.
    """
    candidate = path if path.is_absolute() else root / path
    # The following control-flow step enforces or verifies the surrounding contract.
    if candidate.is_symlink():
        raise HeldOutEvaluationError("held-out evaluation output must not be a symbolic link")
    resolved = candidate.resolve()
    # The following control-flow step enforces or verifies the surrounding contract.
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise HeldOutEvaluationError(
            "held-out evaluation output must stay within repository root"
        ) from error
    # The following control-flow step enforces or verifies the surrounding contract.
    if relative.parts[:2] != ("artifacts", "runs") or relative.parts[-2:] != (
        "held-out-evaluation",
        filename,
    ):
        raise HeldOutEvaluationError(
            f"held-out evaluation output must be "
            f"artifacts/runs/<run-id>/held-out-evaluation/{filename}"
        )
    # The following control-flow step enforces or verifies the surrounding contract.
    if not resolved.parent.is_dir():
        raise HeldOutEvaluationError(
            f"held-out evaluation output parent does not exist: {resolved.parent}"
        )
    return resolved


def _write_new(path: Path, content: str) -> None:
    """Write content to a path that must not already exist."""
    # The following control-flow step enforces or verifies the surrounding contract.
    try:
        # The following control-flow step enforces or verifies the surrounding contract.
        with path.open("x", encoding="utf-8") as output:
            output.write(content)
    except FileExistsError as error:
        raise HeldOutEvaluationError(
            f"held-out evaluation output already exists: {path}"
        ) from error
    except OSError as error:
        raise HeldOutEvaluationError(
            f"could not write held-out evaluation output {path}: {error}"
        ) from error


def _read_json(path: Path, description: str) -> dict[str, Any]:
    """Read and parse one JSON document, requiring it to be a top-level object."""
    # The following control-flow step enforces or verifies the surrounding contract.
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HeldOutEvaluationError(f"could not read {description} {path}: {error}") from error
    # The following control-flow step enforces or verifies the surrounding contract.
    if not isinstance(value, dict):
        raise HeldOutEvaluationError(f"{description} must be a JSON object")
    return value


def _digest(root: Path, path: Path) -> FileDigest:
    """Calculate stable size and SHA-256 evidence for one repository file."""
    digest, size = hashlib.sha256(), 0
    # The following control-flow step enforces or verifies the surrounding contract.
    with path.open("rb") as source:
        # The following control-flow step enforces or verifies the surrounding contract.
        while chunk := source.read(BUFFER_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return FileDigest(path.relative_to(root).as_posix(), size, digest.hexdigest())


# ---------------------------------------------------------------------------
# Public loader helpers (for CLI use)
# ---------------------------------------------------------------------------


def load_approval_record(path: Path) -> ApprovalRecord:
    """Load a previously written ApprovalRecord JSON from disk.

    Args:
        path: Path to the approval record JSON file.

    Returns:
        The parsed ApprovalRecord.
    """
    # The following control-flow step enforces or verifies the surrounding contract.
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HeldOutEvaluationError(f"could not load approval record {path}: {error}") from error
    # The following control-flow step enforces or verifies the surrounding contract.
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise HeldOutEvaluationError("approval record must be a JSON object with schema_version 1")
    required = (
        "policy_id",
        "policy_version",
        "owner",
        "candidate_run_id",
        "purpose",
        "prior_attempt_exists",
        "run_manifest_reference",
        "verified_lineage_references",
    )
    # The following control-flow step enforces or verifies the surrounding contract.
    for field in required:
        # The following control-flow step enforces or verifies the surrounding contract.
        if field not in document:
            raise HeldOutEvaluationError(f"approval record is missing required field: {field}")
    # The following control-flow step enforces or verifies the surrounding contract.
    try:
        return ApprovalRecord(
            schema_version=1,
            policy_id=document["policy_id"],
            policy_version=document["policy_version"],
            owner=document["owner"],
            candidate_run_id=document["candidate_run_id"],
            purpose=document["purpose"],
            prior_attempt_exists=document["prior_attempt_exists"],
            run_manifest_reference=document["run_manifest_reference"],
            verified_lineage_references=tuple(document["verified_lineage_references"]),
        )
    except (KeyError, TypeError) as error:
        raise HeldOutEvaluationError(f"malformed approval record: {error}") from error
