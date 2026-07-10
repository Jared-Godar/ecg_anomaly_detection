"""Configuration-driven orchestration for the supported local data stages."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from ecg_anomaly_detection.acquisition import Fetcher, acquire_dataset
from ecg_anomaly_detection.config import load_dataset_config
from ecg_anomaly_detection.dataset_index import create_dataset_index, write_dataset_index
from ecg_anomaly_detection.evaluation import (
    evaluate_validation_from_index,
    load_evaluation_config,
)
from ecg_anomaly_detection.inventory import create_inventory, write_manifest
from ecg_anomaly_detection.labels import (
    load_annotation_mapping,
    map_annotations,
    write_mapping_report,
)
from ecg_anomaly_detection.progress import ProgressReporter
from ecg_anomaly_detection.records import (
    load_wfdb_record,
    validate_record,
    write_validation_report,
)
from ecg_anomaly_detection.reproducibility import (
    RuntimeStageTimer,
    capture_environment_summary,
    capture_resource_summary,
    create_evidence_manifest,
    write_evidence,
)
from ecg_anomaly_detection.run_manifest import create_run_manifest, write_run_manifest
from ecg_anomaly_detection.splitting import (
    create_split_manifest,
    create_split_quality_summary,
    enforce_split_quality,
    load_split_config,
    load_window_metadata,
    write_split_manifest,
    write_split_quality_summary,
)
from ecg_anomaly_detection.training import load_training_config, train_from_index
from ecg_anomaly_detection.windows import (
    extract_windows,
    load_window_config,
    write_window_artifact,
    write_window_report,
)


class PipelineError(ValueError):
    """Raised when orchestration paths or run identity violate their contract."""


_TOTAL_REPORTED_STAGES = 7


@dataclass(frozen=True, slots=True)
class PipelineRunResult:
    """Key output paths and counts for one completed local run."""

    run_id: str
    run_directory: Path
    interim_directory: Path
    processed_directory: Path
    acquisition_manifest_path: Path
    inventory_manifest_path: Path
    split_manifest_path: Path
    split_quality_summary_path: Path
    dataset_index_path: Path
    model_path: Path
    training_metadata_path: Path
    validation_metrics_path: Path
    environment_summary_path: Path
    runtime_summary_path: Path
    resource_summary_path: Path
    evidence_manifest_path: Path
    run_manifest_path: Path
    record_count: int
    window_count: int


def run_pipeline(
    repository_root: Path,
    dataset_config_path: Path,
    mapping_config_path: Path,
    window_config_path: Path,
    split_config_path: Path,
    training_config_path: Path,
    evaluation_config_path: Path,
    *,
    fetcher: Fetcher | None = None,
    clock: Callable[[], datetime] | None = None,
    run_id_factory: Callable[[], str] | None = None,
    monotonic: Callable[[], float] = perf_counter,
    reporter: ProgressReporter | None = None,
) -> PipelineRunResult:
    """Run acquisition through fitting and validation-only evaluation."""
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
            training_config_path,
            evaluation_config_path,
        )
    )
    dataset_config = load_dataset_config(config_paths[0])
    mapping_config = load_annotation_mapping(config_paths[1])
    window_config = load_window_config(config_paths[2])
    split_config = load_split_config(config_paths[3])
    training_config = load_training_config(config_paths[4])
    evaluation_config = load_evaluation_config(config_paths[5])
    run_id = _create_run_id(run_id_factory)
    timestamp = clock or (lambda: datetime.now(UTC))
    runtime_timer = RuntimeStageTimer(monotonic)
    progress = reporter or ProgressReporter()
    progress.header(f"run {run_id} starting")

    raw_data_dir = root / "data" / "raw" / dataset_config.slug / dataset_config.version
    dataset_evidence_dir = (
        root / "artifacts" / "datasets" / dataset_config.slug / dataset_config.version
    )
    dataset_evidence_dir.mkdir(parents=True, exist_ok=True)
    acquisition_manifest_path = dataset_evidence_dir / "acquisition.json"
    with (
        progress.stage(
            "acquisition",
            1,
            _TOTAL_REPORTED_STAGES,
            detail=f"{len(dataset_config.record_ids)} records, "
            f"{len(dataset_config.expected_files)} files expected",
        ) as stage,
        runtime_timer.stage("acquisition"),
    ):
        acquire_dataset(
            dataset_config,
            root,
            raw_data_dir,
            acquisition_manifest_path,
            fetcher=fetcher,
            clock=timestamp,
        )
        stage.detail(f"manifest written to {acquisition_manifest_path.relative_to(root)}")

    run_directory, interim_directory, processed_directory = _create_run_directories(root, run_id)
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

    with progress.stage("inventory", 2, _TOTAL_REPORTED_STAGES) as stage:
        inventory = create_inventory(dataset_config, raw_data_dir, clock=timestamp)
        inventory_manifest_path = run_directory / "inventory.json"
        write_manifest(inventory, inventory_manifest_path)
        stage.detail(f"{len(inventory.files)} files verified")

    validation_paths: list[Path] = []
    mapping_paths: list[Path] = []
    window_report_paths: list[Path] = []
    window_artifact_paths: list[Path] = []
    total_windows = 0
    record_total = len(dataset_config.record_ids)
    with progress.stage(
        "record_processing", 3, _TOTAL_REPORTED_STAGES, detail=f"{record_total} records"
    ) as stage:
        for record_index, record_id in enumerate(dataset_config.record_ids, start=1):
            loaded = load_wfdb_record(dataset_config, raw_data_dir, record_id)
            with runtime_timer.stage("validation"):
                validation = validate_record(dataset_config, loaded.signal, loaded.annotations)
                validation_path = validation_directory / f"{record_id}.json"
                write_validation_report(validation, validation_path)
            validation_paths.append(validation_path)

            if record_id in window_config.exclude_record_ids:
                progress.note(f"record {record_index}/{record_total} ({record_id}): excluded")
                continue

            with runtime_timer.stage("annotation_mapping"):
                mapped = map_annotations(mapping_config, loaded.annotations)
                mapping_path = mapping_directory / f"{record_id}.json"
                write_mapping_report(mapped.report, mapping_path)
            mapping_paths.append(mapping_path)

            with runtime_timer.stage("window_extraction"):
                extracted = extract_windows(window_config, mapping_config, loaded.signal, mapped)
                window_artifact_path = window_artifact_directory / f"{record_id}.npz"
                window_report_path = window_report_directory / f"{record_id}.json"
                write_window_artifact(extracted.window_set, window_artifact_path)
                write_window_report(extracted.report, window_report_path)
            window_artifact_paths.append(window_artifact_path)
            window_report_paths.append(window_report_path)
            total_windows += extracted.report.emitted_window_count
            progress.note(
                f"record {record_index}/{record_total} ({record_id}): "
                f"{extracted.report.emitted_window_count} windows"
            )
        stage.detail(f"{total_windows} windows across {len(window_artifact_paths)} records")

    with (
        progress.stage("split", 4, _TOTAL_REPORTED_STAGES) as stage,
        runtime_timer.stage("split"),
    ):
        metadata = load_window_metadata(window_artifact_paths)
        split_record_ids = set(metadata.record_shards)
        filtered_split_config = split_config
        if set(split_config.record_subjects) != split_record_ids:
            filtered_split_config = type(split_config)(
                schema_version=split_config.schema_version,
                name=split_config.name,
                version=split_config.version,
                strategy=split_config.strategy,
                seed=split_config.seed,
                train_ratio=split_config.train_ratio,
                validation_ratio=split_config.validation_ratio,
                test_ratio=split_config.test_ratio,
                record_subjects={
                    record_id: subject_id
                    for record_id, subject_id in split_config.record_subjects.items()
                    if record_id in split_record_ids
                },
                quality=split_config.quality,
            )
        split_manifest = create_split_manifest(filtered_split_config, metadata)
        split_manifest_path = run_directory / "split.json"
        write_split_manifest(split_manifest, split_manifest_path)
        stage.detail(
            f"{split_manifest.total_record_count} records, "
            f"manifest {split_manifest_path.relative_to(root)}"
        )
    with (
        progress.stage("split_diagnostics", 5, _TOTAL_REPORTED_STAGES) as stage,
        runtime_timer.stage("split_diagnostics"),
    ):
        split_quality_summary = create_split_quality_summary(split_config, split_manifest, metadata)
        split_quality_summary_path = run_directory / "split_quality_summary.json"
        write_split_quality_summary(split_quality_summary, split_quality_summary_path)
        enforce_split_quality(split_quality_summary)
        stage.detail(f"quality {split_quality_summary.status}")
    if split_manifest.total_window_count != total_windows:
        raise PipelineError("split window count does not match per-record extraction reports")

    dataset_index = create_dataset_index(root, split_manifest_path, window_artifact_paths)
    dataset_index_path = processed_directory / "dataset-index.json"
    write_dataset_index(dataset_index, root, dataset_index_path)

    training_directory = run_directory / "training"
    training_directory.mkdir()
    model_path = training_directory / "model.json"
    training_metadata_path = training_directory / "training-metadata.json"
    with (
        progress.stage("training", 6, _TOTAL_REPORTED_STAGES) as stage,
        runtime_timer.stage("training"),
    ):
        train_from_index(
            root,
            dataset_index_path,
            training_config,
            model_path,
            training_metadata_path,
        )
        stage.detail(f"model written to {model_path.relative_to(root)}")

    evaluation_directory = run_directory / "evaluation"
    evaluation_directory.mkdir()
    validation_metrics_path = evaluation_directory / "validation-metrics.json"
    with (
        progress.stage("validation_evaluation", 7, _TOTAL_REPORTED_STAGES) as stage,
        runtime_timer.stage("validation_evaluation"),
    ):
        evaluate_validation_from_index(
            root,
            dataset_index_path,
            model_path,
            training_metadata_path,
            evaluation_config,
            validation_metrics_path,
        )
        stage.detail(f"metrics written to {validation_metrics_path.relative_to(root)}")

    operational_evidence_paths = (
        acquisition_manifest_path,
        split_quality_summary_path,
        *validation_paths,
        *mapping_paths,
        *window_report_paths,
    )
    artifact_paths = (
        *window_artifact_paths,
        dataset_index_path,
        model_path,
        training_metadata_path,
        validation_metrics_path,
    )
    environment_summary_path = run_directory / "environment_summary.json"
    runtime_summary_path = run_directory / "runtime_summary.json"
    resource_summary_path = run_directory / "resource_summary.json"
    evidence_manifest_path = run_directory / "evidence_manifest.json"
    write_evidence(capture_environment_summary(root), root, environment_summary_path)
    write_evidence(runtime_timer.summary(), root, runtime_summary_path)
    write_evidence(capture_resource_summary(root), root, resource_summary_path)
    reproducibility_evidence_paths = (
        environment_summary_path,
        runtime_summary_path,
        resource_summary_path,
    )
    evidence_manifest = create_evidence_manifest(
        root,
        split_manifest_path,
        config_paths,
        (*operational_evidence_paths, *reproducibility_evidence_paths),
        artifact_paths,
        split_name=split_manifest.split_name,
        split_version=split_manifest.split_version,
        split_strategy=split_manifest.strategy,
        split_seed=split_manifest.seed,
    )
    write_evidence(evidence_manifest, root, evidence_manifest_path)

    run_manifest = create_run_manifest(
        root,
        inventory_manifest_path,
        split_manifest_path,
        config_paths,
        evidence_paths=(
            *operational_evidence_paths,
            *reproducibility_evidence_paths,
            evidence_manifest_path,
        ),
        artifact_paths=artifact_paths,
        clock=timestamp,
        run_id_factory=lambda: run_id,
    )
    run_manifest_path = run_directory / "run-manifest.json"
    write_run_manifest(run_manifest, root, run_manifest_path)
    return PipelineRunResult(
        run_id=run_id,
        run_directory=run_directory,
        interim_directory=interim_directory,
        processed_directory=processed_directory,
        acquisition_manifest_path=acquisition_manifest_path,
        inventory_manifest_path=inventory_manifest_path,
        split_manifest_path=split_manifest_path,
        split_quality_summary_path=split_quality_summary_path,
        dataset_index_path=dataset_index_path,
        model_path=model_path,
        training_metadata_path=training_metadata_path,
        validation_metrics_path=validation_metrics_path,
        environment_summary_path=environment_summary_path,
        runtime_summary_path=runtime_summary_path,
        resource_summary_path=resource_summary_path,
        evidence_manifest_path=evidence_manifest_path,
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


def _create_run_directories(repository_root: Path, run_id: str) -> tuple[Path, Path, Path]:
    artifact_runs = _prepare_run_parent(repository_root / "artifacts")
    interim_runs = _prepare_run_parent(repository_root / "data" / "interim")
    processed_runs = _prepare_run_parent(repository_root / "data" / "processed")
    run_directory = artifact_runs / run_id
    interim_directory = interim_runs / run_id
    processed_directory = processed_runs / run_id
    try:
        run_directory.mkdir()
        interim_directory.mkdir()
        processed_directory.mkdir()
    except FileExistsError as error:
        raise PipelineError(f"pipeline run ID already exists: {run_id}") from error
    except OSError as error:
        raise PipelineError(f"could not create pipeline run directories: {error}") from error
    return run_directory, interim_directory, processed_directory


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
