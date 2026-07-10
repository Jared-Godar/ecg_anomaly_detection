"""Auditable run manifests linking code, environment, configuration, and artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
import string
import subprocess
import sys
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from ecg_anomaly_detection.inventory import InventoryError, InventoryManifest, read_manifest
from ecg_anomaly_detection.splitting import SplitError, read_split_manifest

BUFFER_SIZE = 1024 * 1024
UNKNOWN_GIT_REVISION = "unknown"
"""Sentinel `GitState.revision` recorded when Git state cannot be captured.

Not a valid commit hash (it is not 40 hex characters), so it is provably distinguishable from a
real revision and fails `benchmark_approval._valid_commit_hash()` by construction.
"""


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
    """Source revision used for a run.

    `revision` is `UNKNOWN_GIT_REVISION` and `dirty` is `None` when Git state could not be
    captured (Git unavailable, or a non-zero exit code) -- distinct from the case where Git
    succeeds but returns a malformed revision, which still raises `RunManifestError`.
    """

    revision: str
    dirty: bool | None


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
    """Subject, record, and window evidence copied from one split partition."""

    subject_ids: tuple[str, ...]
    subject_count: int
    record_ids: tuple[str, ...]
    record_subjects: dict[str, str]
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
    total_subject_count: int
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


def read_run_manifest(path: Path) -> RunManifest:
    """Load a previously written run manifest without recomputing any evidence."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RunManifestError(f"could not read run manifest {path}: {error}") from error
    if not isinstance(document, dict):
        raise RunManifestError(f"run manifest must be a JSON object: {path}")
    try:
        return _manifest_from_document(document)
    except (KeyError, TypeError, ValueError) as error:
        raise RunManifestError(f"invalid run manifest {path}: {error}") from error


def _manifest_from_document(document: dict[str, Any]) -> RunManifest:
    git = document["git"]
    environment = document["environment"]
    dataset = document["dataset"]
    split = document["split"]
    return RunManifest(
        schema_version=document["schema_version"],
        run_id=document["run_id"],
        created_at_utc=document["created_at_utc"],
        git=GitState(revision=git["revision"], dirty=git["dirty"]),
        environment=EnvironmentSnapshot(
            python_version=environment["python_version"],
            python_implementation=environment["python_implementation"],
            platform=environment["platform"],
            machine=environment["machine"],
            installed_packages=dict(environment["installed_packages"]),
        ),
        dependency_lock=_file_evidence_from_document(document["dependency_lock"]),
        dataset=DatasetEvidence(
            dataset_slug=dataset["dataset_slug"],
            dataset_version=dataset["dataset_version"],
            inventory_created_at_utc=dataset["inventory_created_at_utc"],
            file_count=dataset["file_count"],
            total_size_bytes=dataset["total_size_bytes"],
            source_files=tuple(
                _file_evidence_from_document(item) for item in dataset["source_files"]
            ),
            inventory_manifest=_file_evidence_from_document(dataset["inventory_manifest"]),
        ),
        split=SplitEvidence(
            split_name=split["split_name"],
            split_version=split["split_version"],
            strategy=split["strategy"],
            seed=split["seed"],
            mapping_name=split["mapping_name"],
            mapping_version=split["mapping_version"],
            window_config_name=split["window_config_name"],
            window_config_version=split["window_config_version"],
            total_subject_count=split["total_subject_count"],
            total_record_count=split["total_record_count"],
            total_window_count=split["total_window_count"],
            partitions={
                name: PartitionEvidence(
                    subject_ids=tuple(partition["subject_ids"]),
                    subject_count=partition["subject_count"],
                    record_ids=tuple(partition["record_ids"]),
                    record_subjects=dict(partition["record_subjects"]),
                    record_count=partition["record_count"],
                    window_count=partition["window_count"],
                    target_value_counts=dict(partition["target_value_counts"]),
                )
                for name, partition in split["partitions"].items()
            },
            split_manifest=_file_evidence_from_document(split["split_manifest"]),
        ),
        configuration_files=tuple(
            _file_evidence_from_document(item) for item in document["configuration_files"]
        ),
        evidence_files=tuple(
            _file_evidence_from_document(item) for item in document["evidence_files"]
        ),
        artifact_files=tuple(
            _file_evidence_from_document(item) for item in document["artifact_files"]
        ),
    )


def _file_evidence_from_document(value: dict[str, Any]) -> FileEvidence:
    return FileEvidence(path=value["path"], size_bytes=value["size_bytes"], sha256=value["sha256"])


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
        manifest = read_split_manifest(path)
    except SplitError as error:
        raise RunManifestError(f"invalid split manifest {path}: {error}") from error
    return SplitEvidence(
        split_name=manifest.split_name,
        split_version=manifest.split_version,
        strategy=manifest.strategy,
        seed=manifest.seed,
        mapping_name=manifest.mapping_name,
        mapping_version=manifest.mapping_version,
        window_config_name=manifest.window_config_name,
        window_config_version=manifest.window_config_version,
        total_subject_count=manifest.total_subject_count,
        total_record_count=manifest.total_record_count,
        total_window_count=manifest.total_window_count,
        partitions={
            name: PartitionEvidence(
                subject_ids=partition.subject_ids,
                subject_count=partition.subject_count,
                record_ids=partition.record_ids,
                record_subjects=partition.record_subjects,
                record_count=partition.record_count,
                window_count=partition.window_count,
                target_value_counts=partition.target_value_counts,
            )
            for name, partition in manifest.partitions.items()
        },
        split_manifest=split_file,
    )


def _capture_git_state(repository_root: Path) -> GitState:
    try:
        # both command lists below are fixed literals, not runtime/user-constructed input.
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],  # noqa: S607
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return GitState(revision=UNKNOWN_GIT_REVISION, dirty=None)
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
