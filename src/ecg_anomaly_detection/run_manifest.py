"""Auditable run manifests linking code, environment, configuration, and artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
import string
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Sequence

from ecg_anomaly_detection.inventory import InventoryError, InventoryManifest, read_manifest

BUFFER_SIZE = 1024 * 1024
PARTITION_NAMES = ("train", "validation", "test")


class RunManifestError(ValueError):
    """Raised when run evidence cannot satisfy the manifest contract."""


@dataclass(frozen=True, slots=True)
class FileEvidence:
    """Repository-relative identity and digest for one run input or output."""

    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class GitState:
    """Source revision used for a run."""

    revision: str
    dirty: bool


@dataclass(frozen=True, slots=True)
class EnvironmentSnapshot:
    """Runtime and installed-package versions used for a run."""

    python_version: str
    python_implementation: str
    platform: str
    machine: str
    installed_packages: dict[str, str]


@dataclass(frozen=True, slots=True)
class DatasetEvidence:
    """Dataset identity and aggregate inventory evidence."""

    dataset_slug: str
    dataset_version: str
    inventory_created_at_utc: str
    file_count: int
    total_size_bytes: int
    source_files: tuple[FileEvidence, ...]
    inventory_manifest: FileEvidence


@dataclass(frozen=True, slots=True)
class PartitionEvidence:
    """Record and window counts copied from one split partition."""

    record_ids: tuple[str, ...]
    record_count: int
    window_count: int
    target_value_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class SplitEvidence:
    """Validated summary copied from a record-grouped split manifest."""

    split_name: str
    split_version: str
    strategy: str
    seed: int
    mapping_name: str
    mapping_version: str
    window_config_name: str
    window_config_version: str
    total_record_count: int
    total_window_count: int
    partitions: dict[str, PartitionEvidence]
    split_manifest: FileEvidence


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Machine-readable operational evidence for one supported pipeline run."""

    schema_version: int
    run_id: str
    created_at_utc: str
    git: GitState
    environment: EnvironmentSnapshot
    dependency_lock: FileEvidence
    dataset: DatasetEvidence
    split: SplitEvidence
    configuration_files: tuple[FileEvidence, ...]
    evidence_files: tuple[FileEvidence, ...]
    artifact_files: tuple[FileEvidence, ...]

    def to_json(self) -> str:
        """Serialize with deterministic keys and formatting."""
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


def create_run_manifest(
    repository_root: Path,
    inventory_manifest_path: Path,
    split_manifest_path: Path,
    config_paths: Sequence[Path],
    evidence_paths: Sequence[Path] = (),
    artifact_paths: Sequence[Path] = (),
    *,
    clock: Callable[[], datetime] | None = None,
    run_id_factory: Callable[[], str] | None = None,
    git_state_provider: Callable[[Path], GitState] | None = None,
    environment_provider: Callable[[], EnvironmentSnapshot] | None = None,
) -> RunManifest:
    """Create an auditable manifest without embedding source or artifact contents."""
    root = repository_root.resolve()
    if not (root / "pyproject.toml").is_file():
        raise RunManifestError(f"repository root does not contain pyproject.toml: {root}")
    if not config_paths:
        raise RunManifestError("at least one configuration file is required")

    all_paths = (
        inventory_manifest_path,
        split_manifest_path,
        *config_paths,
        *evidence_paths,
        *artifact_paths,
    )
    resolved_paths = [_resolve_evidence_path(root, path) for path in all_paths]
    lock_path = _resolve_evidence_path(root, root / "uv.lock")
    if len(set(resolved_paths)) != len(resolved_paths):
        raise RunManifestError("run inputs and artifacts must not contain duplicate paths")
    if lock_path in resolved_paths:
        raise RunManifestError("uv.lock is captured automatically and must not be repeated")

    try:
        inventory = read_manifest(resolved_paths[0])
    except InventoryError as error:
        raise RunManifestError(f"invalid inventory manifest: {error}") from error
    inventory_file = _file_evidence(root, resolved_paths[0])
    dataset_evidence = _dataset_evidence(inventory, inventory_file)
    split_file = _file_evidence(root, resolved_paths[1])
    split_evidence = _read_split_evidence(resolved_paths[1], split_file)
    config_end = 2 + len(config_paths)
    evidence_end = config_end + len(evidence_paths)

    now = (clock or (lambda: datetime.now(UTC)))()
    if now.tzinfo is None:
        raise RunManifestError("run manifest clock must return a timezone-aware datetime")
    run_id = (run_id_factory or (lambda: str(uuid.uuid4())))()
    try:
        uuid.UUID(run_id)
    except (AttributeError, TypeError, ValueError) as error:
        raise RunManifestError("run ID must be a valid UUID") from error

    return RunManifest(
        schema_version=1,
        run_id=run_id,
        created_at_utc=now.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        git=(git_state_provider or _capture_git_state)(root),
        environment=(environment_provider or _capture_environment)(),
        dependency_lock=_file_evidence(root, lock_path),
        dataset=dataset_evidence,
        split=split_evidence,
        configuration_files=tuple(
            _file_evidence(root, path) for path in resolved_paths[2:config_end]
        ),
        evidence_files=tuple(
            _file_evidence(root, path) for path in resolved_paths[config_end:evidence_end]
        ),
        artifact_files=tuple(_file_evidence(root, path) for path in resolved_paths[evidence_end:]),
    )


def write_run_manifest(manifest: RunManifest, repository_root: Path, output_path: Path) -> None:
    """Write an ignored JSON run manifest within the repository boundary."""
    root = repository_root.resolve()
    output_candidate = output_path if output_path.is_absolute() else root / output_path
    if output_candidate.is_symlink():
        raise RunManifestError("run manifest output must not be a symbolic link")
    resolved_output = output_candidate.resolve()
    _require_within_root(root, resolved_output, "run manifest output")
    try:
        relative_output = resolved_output.relative_to(root)
    except ValueError as error:  # pragma: no cover - guarded by _require_within_root
        raise RunManifestError("run manifest output must stay within repository root") from error
    if not relative_output.parts or relative_output.parts[0] != "artifacts":
        raise RunManifestError("run manifest output must be written under artifacts/")
    if resolved_output.suffix != ".json":
        raise RunManifestError("run manifest output must use the .json extension")
    if not resolved_output.parent.is_dir():
        raise RunManifestError(
            f"run manifest parent directory does not exist: {resolved_output.parent}"
        )
    resolved_output.write_text(manifest.to_json(), encoding="utf-8")


def _dataset_evidence(
    inventory: InventoryManifest, inventory_file: FileEvidence
) -> DatasetEvidence:
    return DatasetEvidence(
        dataset_slug=inventory.dataset_slug,
        dataset_version=inventory.dataset_version,
        inventory_created_at_utc=inventory.created_at_utc,
        file_count=len(inventory.files),
        total_size_bytes=sum(item.size_bytes for item in inventory.files),
        source_files=tuple(
            FileEvidence(path=item.path, size_bytes=item.size_bytes, sha256=item.sha256)
            for item in inventory.files
        ),
        inventory_manifest=inventory_file,
    )


def _read_split_evidence(path: Path, split_file: FileEvidence) -> SplitEvidence:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or document.get("schema_version") != 1:
            raise RunManifestError("split manifest must be a schema_version 1 object")
        partitions_document = document["partitions"]
        if not isinstance(partitions_document, dict) or set(partitions_document) != set(
            PARTITION_NAMES
        ):
            raise RunManifestError("split manifest must define train, validation, and test")
        partitions = {
            name: _parse_partition(name, partitions_document[name]) for name in PARTITION_NAMES
        }
        evidence = SplitEvidence(
            split_name=_required_string(document, "split_name"),
            split_version=_required_string(document, "split_version"),
            strategy=_required_string(document, "strategy"),
            seed=_required_nonnegative_int(document, "seed"),
            mapping_name=_required_string(document, "mapping_name"),
            mapping_version=_required_string(document, "mapping_version"),
            window_config_name=_required_string(document, "window_config_name"),
            window_config_version=_required_string(document, "window_config_version"),
            total_record_count=_required_nonnegative_int(document, "total_record_count"),
            total_window_count=_required_nonnegative_int(document, "total_window_count"),
            partitions=partitions,
            split_manifest=split_file,
        )
    except (KeyError, OSError, json.JSONDecodeError) as error:
        raise RunManifestError(f"invalid split manifest {path}: {error}") from error
    _validate_split_evidence(evidence)
    return evidence


def _parse_partition(name: str, value: Any) -> PartitionEvidence:
    if not isinstance(value, dict):
        raise RunManifestError(f"split partition {name} must be an object")
    record_ids_value = value.get("record_ids")
    target_counts_value = value.get("target_value_counts")
    if (
        not isinstance(record_ids_value, list)
        or not all(isinstance(item, str) and item for item in record_ids_value)
        or len(record_ids_value) != len(set(record_ids_value))
    ):
        raise RunManifestError(f"split partition {name} has invalid record_ids")
    if not isinstance(target_counts_value, dict) or not all(
        isinstance(key, str)
        and key
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count >= 0
        for key, count in target_counts_value.items()
    ):
        raise RunManifestError(f"split partition {name} has invalid target counts")
    return PartitionEvidence(
        record_ids=tuple(record_ids_value),
        record_count=_required_nonnegative_int(value, "record_count"),
        window_count=_required_nonnegative_int(value, "window_count"),
        target_value_counts=dict(sorted(target_counts_value.items())),
    )


def _validate_split_evidence(evidence: SplitEvidence) -> None:
    record_sets = [set(partition.record_ids) for partition in evidence.partitions.values()]
    if any(
        left & right for index, left in enumerate(record_sets) for right in record_sets[index + 1 :]
    ):
        raise RunManifestError("split manifest contains record leakage across partitions")
    for name, partition in evidence.partitions.items():
        if partition.record_count != len(partition.record_ids):
            raise RunManifestError(f"split partition {name} record count does not match membership")
        if partition.window_count != sum(partition.target_value_counts.values()):
            raise RunManifestError(f"split partition {name} window and target counts do not match")
    if evidence.total_record_count != sum(
        partition.record_count for partition in evidence.partitions.values()
    ):
        raise RunManifestError("split manifest total record count does not match partitions")
    if evidence.total_window_count != sum(
        partition.window_count for partition in evidence.partitions.values()
    ):
        raise RunManifestError("split manifest total window count does not match partitions")


def _capture_git_state(repository_root: Path) -> GitState:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RunManifestError(f"could not capture Git state: {error}") from error
    if len(revision) != 40 or any(character not in string.hexdigits for character in revision):
        raise RunManifestError("Git revision must be a full 40-character commit hash")
    return GitState(revision=revision.lower(), dirty=bool(status.strip()))


def _capture_environment() -> EnvironmentSnapshot:
    packages: dict[str, str] = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name.lower()] = distribution.version
    return EnvironmentSnapshot(
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        platform=sys.platform,
        machine=platform.machine(),
        installed_packages=dict(sorted(packages.items())),
    )


def _resolve_evidence_path(repository_root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else repository_root / path
    if candidate.is_symlink():
        raise RunManifestError(f"run evidence must not be a symbolic link: {candidate}")
    resolved = candidate.resolve()
    _require_within_root(repository_root, resolved, "run evidence")
    if not resolved.is_file():
        raise RunManifestError(f"run evidence must be a regular file: {resolved}")
    return resolved


def _file_evidence(repository_root: Path, path: Path) -> FileEvidence:
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as source:
        while chunk := source.read(BUFFER_SIZE):
            digest.update(chunk)
            size_bytes += len(chunk)
    return FileEvidence(
        path=path.relative_to(repository_root).as_posix(),
        size_bytes=size_bytes,
        sha256=digest.hexdigest(),
    )


def _require_within_root(repository_root: Path, path: Path, description: str) -> None:
    try:
        path.relative_to(repository_root)
    except ValueError as error:
        raise RunManifestError(f"{description} must stay within repository root") from error


def _required_string(values: dict[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RunManifestError(f"split manifest {key} must be a non-empty string")
    return value.strip()


def _required_nonnegative_int(values: dict[str, Any], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RunManifestError(f"split manifest {key} must be a nonnegative integer")
    return value
