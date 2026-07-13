#!/usr/bin/env python3
"""Create or restore the private cross-runtime handoff used by Colab notebooks.

This wrapper intentionally depends only on the Python standard library and the
repository source tree.  A fresh Colab VM can therefore restore Step 0 state before
the project's locked third-party notebook dependencies have been installed into the
new kernel.  The implementation and security boundary live in
``ecg_anomaly_detection.notebook_handoff``; this file provides stable CLI arguments
and compact machine-readable output for notebook cells.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

# Make the repository package importable from source before a disposable hosted
# kernel has processed the later editable installation's .pth file.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# Prepend rather than append so an unrelated globally installed distribution cannot
# replace the source revision that owns this wrapper.
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from ecg_anomaly_detection.notebook_handoff import (  # noqa: E402
    HandoffResult,
    create_handoff,
    restore_handoff,
)


def _parser() -> argparse.ArgumentParser:
    """Build the explicit create/restore command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Creation is run only after notebook 00 has independently verified Step 0.
    create = subparsers.add_parser("create", help="create a versioned private handoff")
    create.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
        help="checkout containing completed Step 0 state",
    )
    create.add_argument(
        "--destination-directory",
        type=Path,
        required=True,
        help="persistent private directory, normally mounted Google Drive",
    )

    # Restoration runs in a fresh exact-commit checkout before downstream imports.
    restore = subparsers.add_parser("restore", help="verify and restore one handoff")
    restore.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
        help="exact-commit checkout receiving generated state",
    )
    restore.add_argument(
        "--archive",
        type=Path,
        required=True,
        help="versioned handoff ZIP selected by latest.json",
    )
    return parser


def _result_payload(result: HandoffResult) -> dict[str, object]:
    """Return a compact JSON-safe summary for one successful handoff action.

    Args:
        result: Verified handoff result returned by the package implementation.

    Returns:
        Machine-readable success payload that excludes patient-level content.
    """

    return {
        "handoff_status": result.operation,
        "archive": str(result.archive),
        "run_id": result.run_id,
        "repository_commit": result.repository_commit,
        "file_count": result.file_count,
        "uncompressed_bytes": result.total_bytes,
        "included_partitions": ["train", "validation"],
        "protected_test_shards_included": False,
        "raw_data_included": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one handoff command and print its verified JSON summary.

    Args:
        argv: Optional argument sequence; defaults to process arguments.

    Returns:
        Process exit status.
    """

    arguments = _parser().parse_args(argv)
    # Dispatch only the two subcommands registered by the required parser above.
    if arguments.command == "create":
        result = create_handoff(arguments.repository_root, arguments.destination_directory)
    else:
        result = restore_handoff(arguments.repository_root, arguments.archive)
    print(json.dumps(_result_payload(result), indent=2, sort_keys=True))
    return 0


# Keep imports side-effect free for tests and notebook helpers that reuse this module.
if __name__ == "__main__":
    raise SystemExit(main())
