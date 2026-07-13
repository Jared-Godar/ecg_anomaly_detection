#!/usr/bin/env python3
"""Install or verify the locked stack for one disposable hosted notebook kernel.

Notebook cells call this standard-library wrapper before importing NumPy, SciPy,
scikit-learn, or the editable project.  Exit status 75 means installation succeeded
but the current kernel must restart once; status 0 means the caller is a different,
post-install kernel and may perform its own in-process import verification.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

# Source-tree import support is required before the hosted editable installation has
# been processed by a restarted kernel.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# Prefer the exact checkout containing this script over any globally installed copy.
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from ecg_anomaly_detection.hosted_notebook_runtime import (  # noqa: E402
    RESTART_REQUIRED_EXIT,
    prepare_hosted_environment,
)


def _parser() -> argparse.ArgumentParser:
    """Build the hosted-bootstrap CLI used by curated notebook cells."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
        help="exact checkout whose locked notebook group should be installed",
    )
    parser.add_argument(
        "--kernel-pid",
        type=int,
        required=True,
        help="calling kernel PID used with process start time to prove a restart",
    )
    parser.add_argument(
        "--marker-path",
        type=Path,
        default=Path("/content/.ecg-notebook-bootstrap.json"),
        help="runtime-local installation marker retained across a kernel restart",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path(tempfile.gettempdir()) / "ecg-notebook-bootstrap.log",
        help="runtime-local complete dependency installation log",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare the hosted environment and return readiness as JSON plus exit status.

    Args:
        argv: Optional argument sequence; defaults to process arguments.

    Returns:
        Zero when ready or 75 when the calling kernel must restart once.
    """

    arguments = _parser().parse_args(argv)
    result = prepare_hosted_environment(
        arguments.repository_root,
        kernel_pid=arguments.kernel_pid,
        marker_path=arguments.marker_path,
        log_path=arguments.log_path,
    )
    print(
        json.dumps(
            {
                "hosted_bootstrap_status": result.status,
                "fingerprint": result.fingerprint,
                "log_path": str(result.log_path),
                "installed_by_kernel": result.installed_by_kernel,
            },
            sort_keys=True,
        )
    )
    return RESTART_REQUIRED_EXIT if result.status == "restart_required" else 0


# Importing this wrapper for unit tests must not install packages or inspect a kernel.
if __name__ == "__main__":
    raise SystemExit(main())
