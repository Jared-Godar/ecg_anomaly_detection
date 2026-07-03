"""Command-line interface for supported local data-pipeline stages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from ecg_anomaly_detection.acquisition import AcquisitionError, acquire_dataset
from ecg_anomaly_detection.config import ConfigurationError, load_dataset_config
from ecg_anomaly_detection.dataset_index import (
    DatasetIndexError,
    create_dataset_index,
    write_dataset_index,
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
from ecg_anomaly_detection.pipeline import PipelineError, run_pipeline
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
    load_split_config,
    load_window_metadata,
    write_split_manifest,
)
from ecg_anomaly_detection.windows import (
    WindowExtractionError,
    extract_windows,
    load_window_config,
    write_window_artifact,
    write_window_report,
)


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
        "inventory", help="hash every required file and write a local baseline manifest"
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
        help="assign complete records to deterministic train, validation, and test partitions",
    )
    split_parser.add_argument("--split-config", type=Path, required=True)
    split_parser.add_argument("--input", type=Path, action="append", required=True)
    split_parser.add_argument("--output", type=Path, required=True)

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

    index_parser = subparsers.add_parser(
        "index-dataset",
        help="validate record shards and write a model-ready grouped dataset index",
    )
    index_parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    index_parser.add_argument("--split-manifest", type=Path, required=True)
    index_parser.add_argument("--input", type=Path, action="append", required=True)
    index_parser.add_argument("--output", type=Path, required=True)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    options = parser.parse_args(arguments)

    try:
        if options.command == "index-dataset":
            index = create_dataset_index(
                options.repository_root,
                options.split_manifest,
                options.input,
            )
            write_dataset_index(index, options.repository_root, options.output)
            print(
                f"indexed {index.total_record_count} records and "
                f"{index.total_window_count} windows in {options.output}"
            )
            return 0
        if options.command == "run-pipeline":
            result = run_pipeline(
                options.repository_root,
                options.dataset_config,
                options.mapping_config,
                options.window_config,
                options.split_config,
            )
            print(
                f"completed run {result.run_id}: {result.record_count} records, "
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
        if options.command == "split-windows":
            split_config = load_split_config(options.split_config)
            metadata = load_window_metadata(options.input)
            manifest = create_split_manifest(split_config, metadata)
            write_split_manifest(manifest, options.output)
            print(
                f"assigned {manifest.total_record_count} records "
                f"across 3 partitions in {options.output}"
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
        ConfigurationError,
        DatasetIndexError,
        InventoryError,
        PipelineError,
        RecordValidationError,
        RunManifestError,
        SplitError,
        WindowExtractionError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
