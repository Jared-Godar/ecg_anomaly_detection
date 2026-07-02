"""Command-line interface for local dataset inventory checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from ecg_anomaly_detection.config import ConfigurationError, load_dataset_config
from ecg_anomaly_detection.inventory import (
    InventoryError,
    create_inventory,
    read_manifest,
    verify_inventory,
    write_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ecg-data",
        description="Inventory and verify ignored local ECG dataset files.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

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
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    options = parser.parse_args(arguments)

    try:
        config = load_dataset_config(options.config)
        if options.command == "inventory":
            manifest = create_inventory(config, options.data_dir)
            write_manifest(manifest, options.output)
            print(f"inventoried {len(manifest.files)} files in {options.output}")
        else:
            manifest = read_manifest(options.manifest)
            verify_inventory(config, options.data_dir, manifest)
            print(f"verified {len(manifest.files)} files against {options.manifest}")
    except (ConfigurationError, InventoryError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
