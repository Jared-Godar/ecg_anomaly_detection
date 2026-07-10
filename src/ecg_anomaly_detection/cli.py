"""Command-line interface for supported local data-pipeline stages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Sequence

from ecg_anomaly_detection.acquisition import AcquisitionError, acquire_dataset
from ecg_anomaly_detection.benchmark_approval import (
    BenchmarkApprovalError,
    record_benchmark_approval,
)
from ecg_anomaly_detection.benchmark_policy import (
    BenchmarkPolicyError,
    load_benchmark_policy,
)
from ecg_anomaly_detection.config import ConfigurationError, load_dataset_config
from ecg_anomaly_detection.dataset_index import (
    DatasetIndexError,
    create_dataset_index,
    write_dataset_index,
)
from ecg_anomaly_detection.evaluation import (
    EvaluationError,
    evaluate_threshold_sweep_from_index,
    load_threshold_sweep_config,
)
from ecg_anomaly_detection.inventory import (
    InventoryError,
    create_inventory,
    read_manifest,
    verify_inventory,
    write_manifest,
)
from ecg_anomaly_detection.labels import (
    AnnotationMappingError,
    load_annotation_mapping,
    map_annotations,
    write_mapping_report,
)
from ecg_anomaly_detection.local_execution import (
    LocalArtifactLifecycleError,
    list_runs,
    purge_run,
)
from ecg_anomaly_detection.pipeline import PipelineError, run_pipeline
from ecg_anomaly_detection.progress import ProgressReporter, format_elapsed_seconds
from ecg_anomaly_detection.records import (
    RecordValidationError,
    load_wfdb_record,
    validate_record,
    write_validation_report,
)
from ecg_anomaly_detection.run_manifest import (
    RunManifestError,
    create_run_manifest,
    write_run_manifest,
)
from ecg_anomaly_detection.splitting import (
    SplitError,
    create_split_manifest,
    create_split_quality_summary,
    enforce_split_quality,
    load_split_config,
    load_window_metadata,
    write_split_manifest,
    write_split_quality_summary,
)
from ecg_anomaly_detection.training import TrainingError
from ecg_anomaly_detection.windows import (
    WindowExtractionError,
    extract_windows,
    load_window_config,
    write_window_artifact,
    write_window_report,
)

_ARTIFACT_GLOB = "*.npz"


class ArtifactDiscoveryError(ValueError):
    """Raised when a --input path cannot be resolved to concrete artifact files."""


def _resolve_input_paths(paths: Sequence[Path]) -> tuple[Path, ...]:
    """Expand directory --input arguments into their contained artifact files.

    Each path must be an existing regular file or an existing, non-symlink
    directory. A directory expands to its immediate `*.npz` children, sorted
    by name for determinism; it is not searched recursively, matching the
    flat layout `run_pipeline()` writes under
    `data/interim/runs/<run-id>/windows/`. File arguments pass through
    unchanged and are not themselves required to end in `.npz`, preserving
    existing behavior exactly. The combined, expanded result must not
    contain the same file twice.
    """
    resolved: list[Path] = []
    for path in paths:
        if path.is_symlink():
            raise ArtifactDiscoveryError(f"input path must not be a symbolic link: {path}")
        if path.is_dir():
            discovered = sorted(path.glob(_ARTIFACT_GLOB))
            if not discovered:
                raise ArtifactDiscoveryError(
                    f"no {_ARTIFACT_GLOB} artifact files found in directory: {path}"
                )
            resolved.extend(discovered)
        elif path.is_file():
            resolved.append(path)
        else:
            raise ArtifactDiscoveryError(f"input path does not exist: {path}")
    seen: dict[Path, Path] = {}
    duplicates: list[Path] = []
    for path in resolved:
        canonical = path.resolve()
        if canonical in seen:
            duplicates.append(path)
        else:
            seen[canonical] = path
    if duplicates:
        raise ArtifactDiscoveryError(
            "duplicate input artifact path(s) after directory expansion: "
            + ", ".join(str(path) for path in duplicates)
        )
    return tuple(resolved)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ecg-data",
        description="Run supported local ECG data-pipeline stages.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    acquire_parser = subparsers.add_parser(
        "acquire", help="retrieve the configured public dataset into the ignored raw-data zone"
    )
    _add_common_arguments(acquire_parser)
    acquire_parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    acquire_parser.add_argument("--output", type=Path, required=True)

    inventory_parser = subparsers.add_parser(
        "inventory", help="verify and hash every required file, then write observed evidence"
    )
    _add_common_arguments(inventory_parser)
    inventory_parser.add_argument("--output", type=Path, required=True)

    verify_parser = subparsers.add_parser(
        "verify", help="verify local files against a previously written manifest"
    )
    _add_common_arguments(verify_parser)
    verify_parser.add_argument("--manifest", type=Path, required=True)

    record_parser = subparsers.add_parser(
        "validate-record", help="load and validate one local WFDB signal and annotation record"
    )
    _add_common_arguments(record_parser)
    record_parser.add_argument("--record-id", required=True)
    record_parser.add_argument("--output", type=Path, required=True)

    mapping_parser = subparsers.add_parser(
        "map-annotations", help="validate a WFDB record and audit its configured annotation mapping"
    )
    _add_common_arguments(mapping_parser)
    mapping_parser.add_argument("--mapping-config", type=Path, required=True)
    mapping_parser.add_argument("--record-id", required=True)
    mapping_parser.add_argument("--output", type=Path, required=True)

    window_parser = subparsers.add_parser(
        "extract-windows",
        help="extract configured beat windows and write an NPZ artifact and report",
    )
    _add_common_arguments(window_parser)
    window_parser.add_argument("--mapping-config", type=Path, required=True)
    window_parser.add_argument("--window-config", type=Path, required=True)
    window_parser.add_argument("--record-id", required=True)
    window_parser.add_argument("--output", type=Path, required=True)
    window_parser.add_argument("--report", type=Path, required=True)

    split_parser = subparsers.add_parser(
        "split-windows",
        help="assign complete subjects to deterministic train, validation, and test partitions",
    )
    split_parser.add_argument("--split-config", type=Path, required=True)
    split_parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="window artifact file, or a directory of *.npz files (repeatable)",
    )
    split_parser.add_argument("--output", type=Path, required=True)
    split_parser.add_argument(
        "--quality-output",
        type=Path,
        help="quality summary path (default: split_quality_summary.json beside --output)",
    )

    run_parser = subparsers.add_parser(
        "create-run-manifest",
        help="hash run evidence and write an auditable operational manifest",
    )
    run_parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    run_parser.add_argument("--inventory-manifest", type=Path, required=True)
    run_parser.add_argument("--split-manifest", type=Path, required=True)
    run_parser.add_argument("--config", type=Path, action="append", required=True)
    run_parser.add_argument("--evidence", type=Path, action="append", default=[])
    run_parser.add_argument("--artifact", type=Path, action="append", default=[])
    run_parser.add_argument("--output", type=Path, required=True)

    pipeline_parser = subparsers.add_parser(
        "run-pipeline",
        help="run supported data stages from acquisition through auditable run evidence",
    )
    pipeline_parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    pipeline_parser.add_argument("--dataset-config", type=Path, required=True)
    pipeline_parser.add_argument("--mapping-config", type=Path, required=True)
    pipeline_parser.add_argument("--window-config", type=Path, required=True)
    pipeline_parser.add_argument("--split-config", type=Path, required=True)
    pipeline_parser.add_argument("--training-config", type=Path, required=True)
    pipeline_parser.add_argument("--evaluation-config", type=Path, required=True)

    index_parser = subparsers.add_parser(
        "index-dataset",
        help="validate record shards and write a model-ready grouped dataset index",
    )
    index_parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    index_parser.add_argument("--split-manifest", type=Path, required=True)
    index_parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="window artifact file, or a directory of *.npz files (repeatable)",
    )
    index_parser.add_argument("--output", type=Path, required=True)

    policy_parser = subparsers.add_parser(
        "validate-benchmark-policy",
        help="validate benchmark governance configuration without accessing benchmark data",
    )
    policy_parser.add_argument("--policy", type=Path, required=True)

    approval_parser = subparsers.add_parser(
        "record-benchmark-approval",
        help=(
            "verify benchmark eligibility, approval, and lineage, and record approval-gate "
            "evidence without accessing the protected test partition"
        ),
    )
    approval_parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    approval_parser.add_argument("--policy", type=Path, required=True)
    approval_parser.add_argument("--run-manifest", type=Path, required=True)
    approval_parser.add_argument("--approval", type=Path, required=True)
    approval_parser.add_argument("--output", type=Path, required=True)

    threshold_sweep_parser = subparsers.add_parser(
        "evaluate-threshold-sweep",
        help=(
            "report coverage and precision/recall/F1 at configured centroid-distance margin "
            "thresholds over only the indexed validation partition"
        ),
    )
    threshold_sweep_parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    threshold_sweep_parser.add_argument("--dataset-index", type=Path, required=True)
    threshold_sweep_parser.add_argument("--model", type=Path, required=True)
    threshold_sweep_parser.add_argument("--training-metadata", type=Path, required=True)
    threshold_sweep_parser.add_argument("--config", type=Path, required=True)
    threshold_sweep_parser.add_argument("--output", type=Path, required=True)

    notebook_parser = subparsers.add_parser(
        "check-local-notebooks",
        help="validate and optionally normalize local notebooks without executing cells",
    )
    notebook_parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    notebook_parser.add_argument(
        "--notebook",
        type=Path,
        action="append",
        default=[],
        help="specific notebook under notebooks/ (repeatable; defaults to notebooks/local/)",
    )
    notebook_parser.add_argument("--format", action="store_true", dest="format_notebooks")
    notebook_parser.add_argument("--strip-outputs", action="store_true")
    notebook_parser.add_argument("--include-narrative", action="store_true")
    notebook_parser.add_argument("--json", action="store_true", dest="json_output")

    list_runs_parser = subparsers.add_parser(
        "list-runs",
        help="list local run_pipeline() output directories with size and manifest status",
    )
    list_runs_parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    list_runs_parser.add_argument("--json", action="store_true", dest="json_output")

    purge_run_parser = subparsers.add_parser(
        "purge-run",
        help="remove one local run's artifact, interim, and processed output directories",
    )
    purge_run_parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    purge_run_parser.add_argument("--run-id", required=True)
    purge_run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be removed without deleting anything",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    options = parser.parse_args(arguments)

    try:
        if options.command == "check-local-notebooks":
            from ecg_anomaly_detection.notebook_quality import (
                NotebookQualityError,
                check_notebooks,
                discover_local_notebooks,
            )

            try:
                notebook_paths = tuple(options.notebook) or discover_local_notebooks(
                    options.repository_root,
                    include_narrative=options.include_narrative,
                )
                summary = check_notebooks(
                    options.repository_root,
                    notebook_paths,
                    format_notebooks=options.format_notebooks,
                    strip_outputs=options.strip_outputs,
                )
            except NotebookQualityError as error:
                print(f"error: {error}", file=sys.stderr)
                return 1
            if options.json_output:
                print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
            else:
                for report in summary.notebooks:
                    status = "valid" if report.valid else "invalid"
                    print(
                        f"{report.path}: {status}; {report.cell_count} cells, "
                        f"{report.output_count} outputs ({report.output_bytes} bytes)"
                    )
                    for issue in report.issues:
                        location = (
                            f" cell {issue.cell_index}" if issue.cell_index is not None else ""
                        )
                        print(
                            f"  {issue.severity}: {issue.code}{location}: {issue.message}",
                            file=sys.stderr,
                        )
                print(
                    f"checked {len(summary.notebooks)} notebooks; changed {summary.changed_count}"
                )
            return 0 if summary.valid else 1
        if options.command == "validate-benchmark-policy":
            policy = load_benchmark_policy(options.policy)
            print(f"validated benchmark policy {policy.policy_id} version {policy.version}")
            return 0
        if options.command == "record-benchmark-approval":
            record = record_benchmark_approval(
                options.repository_root,
                options.policy,
                options.run_manifest,
                options.approval,
                options.output,
            )
            print(
                f"recorded benchmark approval for candidate {record.candidate_run_id} "
                f"in {options.output}"
            )
            return 0
        if options.command == "evaluate-threshold-sweep":
            sweep_config = load_threshold_sweep_config(options.config)
            result = evaluate_threshold_sweep_from_index(
                options.repository_root,
                options.dataset_index,
                options.model,
                options.training_metadata,
                sweep_config,
                options.output,
            )
            print(
                f"swept {len(sweep_config.thresholds)} thresholds over "
                f"{result.window_count} validation windows in {options.output}"
            )
            return 0
        if options.command == "index-dataset":
            index = create_dataset_index(
                options.repository_root,
                options.split_manifest,
                _resolve_input_paths(options.input),
            )
            write_dataset_index(index, options.repository_root, options.output)
            print(
                f"indexed {index.total_record_count} records and "
                f"{index.total_window_count} windows in {options.output}"
            )
            return 0
        if options.command == "run-pipeline":
            started_at = perf_counter()
            result = run_pipeline(
                options.repository_root,
                options.dataset_config,
                options.mapping_config,
                options.window_config,
                options.split_config,
                options.training_config,
                options.evaluation_config,
                reporter=ProgressReporter(stream=sys.stdout),
            )
            elapsed = format_elapsed_seconds(perf_counter() - started_at)
            print(
                f"completed run {result.run_id} in {elapsed}: {result.record_count} records, "
                f"{result.window_count} windows, manifest {result.run_manifest_path}"
            )
            return 0
        if options.command == "create-run-manifest":
            manifest = create_run_manifest(
                options.repository_root,
                options.inventory_manifest,
                options.split_manifest,
                options.config,
                options.evidence,
                options.artifact,
            )
            write_run_manifest(manifest, options.repository_root, options.output)
            print(f"recorded run {manifest.run_id} in {options.output}")
            return 0
        if options.command == "list-runs":
            summaries = list_runs(options.repository_root)
            if options.json_output:
                print(
                    json.dumps(
                        [
                            {
                                "run_id": summary.run_id,
                                "has_run_manifest": summary.has_run_manifest,
                                "total_size_bytes": summary.total_size_bytes,
                                "modified_at_epoch": summary.modified_at_epoch,
                                "directories": [str(path) for path in summary.directories],
                            }
                            for summary in summaries
                        ],
                        indent=2,
                        sort_keys=True,
                    )
                )
            else:
                if not summaries:
                    print("no local runs found")
                for summary in summaries:
                    manifest_note = "manifest" if summary.has_run_manifest else "no manifest"
                    print(
                        f"{summary.run_id}  {summary.total_size_bytes:>12} bytes  "
                        f"{len(summary.directories)} directories  {manifest_note}"
                    )
            return 0
        if options.command == "purge-run":
            result = purge_run(options.repository_root, options.run_id, dry_run=options.dry_run)
            verb = "would remove" if result.dry_run else "removed"
            for directory in result.removed_directories:
                print(f"{verb} {directory}")
            print(f"{verb.capitalize()} {result.freed_bytes} bytes for run {result.run_id}")
            return 0
        if options.command == "split-windows":
            split_config = load_split_config(options.split_config)
            metadata = load_window_metadata(_resolve_input_paths(options.input))
            manifest = create_split_manifest(split_config, metadata)
            write_split_manifest(manifest, options.output)
            quality_path = options.quality_output or options.output.with_name(
                "split_quality_summary.json"
            )
            quality = create_split_quality_summary(split_config, manifest, metadata)
            write_split_quality_summary(quality, quality_path)
            for violation in quality.violations:
                if violation.severity == "warning":
                    print(f"warning: {violation.message}", file=sys.stderr)
            enforce_split_quality(quality)
            print(
                f"assigned {manifest.total_record_count} records "
                f"across 3 partitions in {options.output}; quality {quality.status} in {quality_path}"
            )
            return 0
        config = load_dataset_config(options.config)
        if options.command == "acquire":
            result = acquire_dataset(
                config,
                options.repository_root,
                options.data_dir,
                options.output,
            )
            print(
                f"acquired {result.downloaded_file_count} and reused "
                f"{result.reused_file_count} files in {options.data_dir}"
            )
        elif options.command == "inventory":
            manifest = create_inventory(config, options.data_dir)
            write_manifest(manifest, options.output)
            print(f"inventoried {len(manifest.files)} files in {options.output}")
        elif options.command == "verify":
            manifest = read_manifest(options.manifest)
            verify_inventory(config, options.data_dir, manifest)
            print(f"verified {len(manifest.files)} files against {options.manifest}")
        elif options.command == "validate-record":
            record = load_wfdb_record(config, options.data_dir, options.record_id)
            report = validate_record(config, record.signal, record.annotations)
            write_validation_report(report, options.output)
            print(f"validated record {options.record_id} in {options.output}")
        elif options.command == "map-annotations":
            record = load_wfdb_record(config, options.data_dir, options.record_id)
            validate_record(config, record.signal, record.annotations)
            mapping = load_annotation_mapping(options.mapping_config)
            result = map_annotations(mapping, record.annotations)
            write_mapping_report(result.report, options.output)
            print(f"mapped annotations for record {options.record_id} in {options.output}")
        elif options.command == "extract-windows":
            record = load_wfdb_record(config, options.data_dir, options.record_id)
            validate_record(config, record.signal, record.annotations)
            mapping = load_annotation_mapping(options.mapping_config)
            mapped = map_annotations(mapping, record.annotations)
            window_config = load_window_config(options.window_config)
            extracted = extract_windows(window_config, mapping, record.signal, mapped)
            write_window_artifact(extracted.window_set, options.output)
            write_window_report(extracted.report, options.report)
            print(f"extracted {extracted.report.emitted_window_count} windows in {options.output}")
    except (
        AcquisitionError,
        AnnotationMappingError,
        ArtifactDiscoveryError,
        BenchmarkApprovalError,
        BenchmarkPolicyError,
        ConfigurationError,
        DatasetIndexError,
        EvaluationError,
        InventoryError,
        LocalArtifactLifecycleError,
        PipelineError,
        RecordValidationError,
        RunManifestError,
        SplitError,
        TrainingError,
        WindowExtractionError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
