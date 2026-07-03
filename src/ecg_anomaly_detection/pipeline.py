"""Configuration-driven orchestration for the supported local data stages."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from ecg_anomaly_detection.acquisition import Fetcher, acquire_dataset
from ecg_anomaly_detection.config import load_dataset_config
from ecg_anomaly_detection.inventory import create_inventory, write_manifest
from ecg_anomaly_detection.labels import (
    load_annotation_mapping,
    map_annotations,
    write_mapping_report,
)
from ecg_anomaly_detection.records import (
    load_wfdb_record,
    validate_record,
    write_validation_report,
)
from ecg_anomaly_detection.run_manifest import create_run_manifest, write_run_manifest
from ecg_anomaly_detection.splitting import (
    create_split_manifest,
    load_split_config,
    load_window_metadata,
    write_split_manifest,
)
from ecg_anomaly_detection.windows import (
    extract_windows,
    load_window_config,
    write_window_artifact,
    write_window_report,
)


class PipelineError(ValueError):
    """Raised when orchestration paths or run identity violate their contract."""


@dataclass(frozen=True, slots=True)
class PipelineRunResult:
    """Key output paths and counts for one completed local run."""

    run_id: str
    run_directory: Path
    interim_directory: Path
    acquisition_manifest_path: Path
    inventory_manifest_path: Path
    split_manifest_path: Path
    run_manifest_path: Path
    record_count: int
    window_count: int


def run_pipeline(
    repository_root: Path,
    dataset_config_path: Path,
    mapping_config_path: Path,
    window_config_path: Path,
    split_config_path: Path,
    *,
    fetcher: Fetcher | None = None,
    clock: Callable[[], datetime] | None = None,
    run_id_factory: Callable[[], str] | None = None,
) -> PipelineRunResult:
    """Run acquisition through evidence generation without training a model."""
    root = repository_root.resolve()
    if not (root / "pyproject.toml").is_file():
        raise PipelineError(f"repository root does not contain pyproject.toml: {root}")
    config_paths = tuple(
        _resolve_config(root, path)
        for path in (
            dataset_config_path,
            mapping_config_path,
            window_config_path,
            split_config_path,
        )
    )
    dataset_config = load_dataset_config(config_paths[0])
    mapping_config = load_annotation_mapping(config_paths[1])
    window_config = load_window_config(config_paths[2])
    split_config = load_split_config(config_paths[3])
    run_id = _create_run_id(run_id_factory)
    timestamp = clock or (lambda: datetime.now(UTC))

    raw_data_dir = root / "data" / "raw" / dataset_config.slug / dataset_config.version
    dataset_evidence_dir = (
        root / "artifacts" / "datasets" / dataset_config.slug / dataset_config.version
    )
    dataset_evidence_dir.mkdir(parents=True, exist_ok=True)
    acquisition_manifest_path = dataset_evidence_dir / "acquisition.json"
    acquire_dataset(
        dataset_config,
        root,
        raw_data_dir,
        acquisition_manifest_path,
        fetcher=fetcher,
        clock=timestamp,
    )

    run_directory, interim_directory = _create_run_directories(root, run_id)
    validation_directory = run_directory / "validation"
    mapping_directory = run_directory / "mapping"
    window_report_directory = run_directory / "windows"
    window_artifact_directory = interim_directory / "windows"
    for directory in (
        validation_directory,
        mapping_directory,
        window_report_directory,
        window_artifact_directory,
    ):
        directory.mkdir()

    inventory = create_inventory(dataset_config, raw_data_dir, clock=timestamp)
    inventory_manifest_path = run_directory / "inventory.json"
    write_manifest(inventory, inventory_manifest_path)

    validation_paths: list[Path] = []
    mapping_paths: list[Path] = []
    window_report_paths: list[Path] = []
    window_artifact_paths: list[Path] = []
    total_windows = 0
    for record_id in dataset_config.record_ids:
        loaded = load_wfdb_record(dataset_config, raw_data_dir, record_id)
        validation = validate_record(dataset_config, loaded.signal, loaded.annotations)
        validation_path = validation_directory / f"{record_id}.json"
        write_validation_report(validation, validation_path)
        validation_paths.append(validation_path)

        mapped = map_annotations(mapping_config, loaded.annotations)
        mapping_path = mapping_directory / f"{record_id}.json"
        write_mapping_report(mapped.report, mapping_path)
        mapping_paths.append(mapping_path)

        extracted = extract_windows(window_config, mapping_config, loaded.signal, mapped)
        window_artifact_path = window_artifact_directory / f"{record_id}.npz"
        window_report_path = window_report_directory / f"{record_id}.json"
        write_window_artifact(extracted.window_set, window_artifact_path)
        write_window_report(extracted.report, window_report_path)
        window_artifact_paths.append(window_artifact_path)
        window_report_paths.append(window_report_path)
        total_windows += extracted.report.emitted_window_count

    metadata = load_window_metadata(window_artifact_paths)
    split_manifest = create_split_manifest(split_config, metadata)
    split_manifest_path = run_directory / "split.json"
    write_split_manifest(split_manifest, split_manifest_path)
    if split_manifest.total_window_count != total_windows:
        raise PipelineError("split window count does not match per-record extraction reports")

    run_manifest = create_run_manifest(
        root,
        inventory_manifest_path,
        split_manifest_path,
        config_paths,
        evidence_paths=(
            acquisition_manifest_path,
            *validation_paths,
            *mapping_paths,
            *window_report_paths,
        ),
        artifact_paths=window_artifact_paths,
        clock=timestamp,
        run_id_factory=lambda: run_id,
    )
    run_manifest_path = run_directory / "run-manifest.json"
    write_run_manifest(run_manifest, root, run_manifest_path)
    return PipelineRunResult(
        run_id=run_id,
        run_directory=run_directory,
        interim_directory=interim_directory,
        acquisition_manifest_path=acquisition_manifest_path,
        inventory_manifest_path=inventory_manifest_path,
        split_manifest_path=split_manifest_path,
        run_manifest_path=run_manifest_path,
        record_count=len(dataset_config.record_ids),
        window_count=total_windows,
    )


def _resolve_config(repository_root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else repository_root / path
    if candidate.is_symlink():
        raise PipelineError(f"pipeline configuration must not be a symbolic link: {candidate}")
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(repository_root)
    except ValueError as error:
        raise PipelineError("pipeline configuration must stay within repository root") from error
    if not relative.parts or relative.parts[0] != "configs":
        raise PipelineError("pipeline configuration must be stored under configs/")
    if not resolved.is_file():
        raise PipelineError(f"pipeline configuration must be a regular file: {resolved}")
    return resolved


def _create_run_id(run_id_factory: Callable[[], str] | None) -> str:
    candidate = (run_id_factory or (lambda: str(uuid.uuid4())))()
    try:
        parsed = uuid.UUID(candidate)
    except (AttributeError, TypeError, ValueError) as error:
        raise PipelineError("pipeline run ID must be a valid UUID") from error
    if str(parsed) != candidate:
        raise PipelineError("pipeline run ID must use canonical lowercase UUID formatting")
    return candidate


def _create_run_directories(repository_root: Path, run_id: str) -> tuple[Path, Path]:
    artifact_runs = _prepare_run_parent(repository_root / "artifacts")
    interim_runs = _prepare_run_parent(repository_root / "data" / "interim")
    run_directory = artifact_runs / run_id
    interim_directory = interim_runs / run_id
    try:
        run_directory.mkdir()
        interim_directory.mkdir()
    except FileExistsError as error:
        raise PipelineError(f"pipeline run ID already exists: {run_id}") from error
    except OSError as error:
        raise PipelineError(f"could not create pipeline run directories: {error}") from error
    return run_directory, interim_directory


def _prepare_run_parent(base_directory: Path) -> Path:
    if base_directory.is_symlink() or not base_directory.is_dir():
        raise PipelineError(
            f"pipeline output base must be an existing regular directory: {base_directory}"
        )
    run_parent = base_directory / "runs"
    if run_parent.is_symlink():
        raise PipelineError(f"pipeline run parent must not be a symbolic link: {run_parent}")
    try:
        run_parent.mkdir(exist_ok=True)
    except OSError as error:
        raise PipelineError(
            f"could not create pipeline run parent {run_parent}: {error}"
        ) from error
    return run_parent
