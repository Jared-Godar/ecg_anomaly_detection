"""Prepare a clean, locked dependency stack for a disposable hosted notebook kernel.

Installing a different NumPy/SciPy/scikit-learn stack underneath a live Python
process can leave old extension modules mixed with new Python files.  The resulting
failures are nondeterministic and cannot be repaired safely by mutating
``sys.path``.  This module installs the repository's locked notebook dependencies,
records which kernel performed the install, and requires a kernel restart before
reporting readiness.  It intentionally performs no notebook-state handoff and no
pipeline work.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO

# This code is used only for Colab's disposable system environment.  Local and
# Codespaces execution continues to use the repository .venv through ``uv sync``.
HOSTED_BOOTSTRAP_SCHEMA_VERSION = 1
# Seventy-five is the conventional temporary-failure status and is distinct from
# installation errors; the wrapper uses it to request a deliberate kernel restart.
RESTART_REQUIRED_EXIT = 75


class HostedNotebookBootstrapError(RuntimeError):
    """Report a strict hosted dependency installation or verification failure."""


def _required_executable(name: str) -> Path:
    """Resolve a trusted host executable before constructing a subprocess call.

    Args:
        name: Bare executable name expected from the hosted base image.

    Returns:
        Absolute executable path.
    """

    discovered = shutil.which(name)
    # Missing base-image tools are actionable setup failures; never fall back to a
    # relative command that could resolve differently after the working directory moves.
    if discovered is None:
        raise HostedNotebookBootstrapError(f"Required hosted executable is unavailable: {name}")
    return Path(discovered).resolve()


@dataclass(frozen=True, slots=True)
class HostedBootstrapResult:
    """Describe whether a hosted kernel is ready or must restart once.

    Attributes:
        status: ``ready`` or ``restart_required``.
        fingerprint: Lock/source identity installed into the hosted environment.
        log_path: Runtime-local complete command output for failure diagnosis.
        installed_by_kernel: Process identity that performed the latest installation.
    """

    status: str
    fingerprint: str
    log_path: Path
    installed_by_kernel: str


def _kernel_identity(kernel_pid: int) -> str:
    """Return a kernel identity that remains distinct if Linux reuses a process ID.

    Args:
        kernel_pid: Calling notebook kernel process ID.

    Returns:
        PID plus Linux process start ticks when available.
    """

    stat_path = Path(f"/proc/{kernel_pid}/stat")
    # Colab runs Linux, where field 22 identifies process start time since boot and
    # prevents an immediately restarted kernel with a reused PID from looking stale.
    if stat_path.is_file():
        # Procfs may race with process teardown, so treat read failure as no start-time data.
        try:
            fields = stat_path.read_text(encoding="utf-8").split()
        # A transient process-table read failure falls back to PID without blocking install.
        except OSError:
            fields = []
        # Linux procfs documents starttime as the 22nd whitespace-delimited field.
        if len(fields) >= 22:
            return f"pid:{kernel_pid}:start:{fields[21]}"
    # Non-Linux unit-test and local diagnostic hosts still retain same-PID rerun protection.
    return f"pid:{kernel_pid}"


def _repository_root(path: Path) -> Path:
    """Resolve a checkout containing the lock and project metadata.

    Args:
        path: Candidate repository root.

    Returns:
        Validated absolute checkout path.
    """

    root = path.resolve()
    # Both inputs define the exact hosted dependency environment and must exist
    # before any system package is changed.
    if not (root / "pyproject.toml").is_file() or not (root / "uv.lock").is_file():
        raise HostedNotebookBootstrapError(
            f"Hosted bootstrap requires pyproject.toml and uv.lock at {root}"
        )
    return root


def _git_commit(root: Path) -> str:
    """Return the exact source revision whose editable project will be installed.

    Args:
        root: Validated repository checkout.

    Returns:
        Exact Git object ID.
    """

    git = _required_executable("git")
    # The executable is host-resolved and every argument is a fixed literal; no shell
    # or caller-controlled command text participates in this source-identity probe.
    result = subprocess.run(  # noqa: S603
        [str(git), "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    commit = result.stdout.strip().lower()
    # An exact source identity is required because the editable installation changes
    # as the checkout changes even when the dependency lock does not.
    if result.returncode != 0 or len(commit) != 40:
        raise HostedNotebookBootstrapError("Unable to resolve hosted checkout commit")
    return commit


def environment_fingerprint(repository_root: Path) -> str:
    """Return a digest of dependency declarations, lock, and exact source commit.

    Args:
        repository_root: Checkout whose hosted environment is being prepared.

    Returns:
        SHA-256 environment fingerprint.
    """

    root = _repository_root(repository_root)
    digest = hashlib.sha256()
    # Include separators and relative names so concatenated content cannot produce an
    # ambiguous digest when file boundaries move.
    for relative in (Path("pyproject.toml"), Path("uv.lock")):
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update((root / relative).read_bytes())
        digest.update(b"\0")
    digest.update(_git_commit(root).encode("ascii"))
    return digest.hexdigest()


def _load_marker(path: Path) -> dict[str, object] | None:
    """Load a prior runtime-local bootstrap marker when it is valid JSON.

    Args:
        path: Marker path on the disposable VM filesystem.

    Returns:
        Parsed marker or ``None`` when absent/invalid.
    """

    # An absent marker is the normal first-run path in every fresh Colab VM.
    if not path.is_file():
        return None
    # Parse defensively because a killed runtime may have interrupted an older write.
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    # A partial marker from an interrupted write is stale evidence, not readiness.
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_marker(path: Path, payload: dict[str, object]) -> None:
    """Atomically record which live kernel installed the hosted environment.

    Args:
        path: Runtime-local marker destination.
        payload: JSON-safe marker fields.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    # Write beside the marker so the final replace stays on one filesystem.
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        json.dump(payload, temporary, indent=2, sort_keys=True)
        temporary.write("\n")
    # The marker becomes authoritative only after its complete payload is durable.
    os.replace(temporary_path, path)


def _uv_executable() -> Path:
    """Return an installed uv executable or install it with the official bootstrap.

    Returns:
        Absolute uv executable path.
    """

    existing = shutil.which("uv")
    # Hosted images may already provide uv; reuse it without downloading anything.
    if existing is not None:
        return Path(existing)

    # The repository's documented hosted path uses uv's official installer when the
    # disposable base image does not already expose the tool.
    with tempfile.TemporaryDirectory(prefix="ecg-uv-installer-") as directory:
        installer = Path(directory) / "install.sh"
        curl = _required_executable("curl")
        shell = _required_executable("sh")
        # This explicit trust boundary downloads uv's documented official installer
        # over TLS into a private temporary directory, with no shell interpolation.
        download = subprocess.run(  # noqa: S603
            [str(curl), "-LsSf", "https://astral.sh/uv/install.sh", "-o", str(installer)],
            capture_output=True,
            text=True,
            check=False,
        )
        # A network/bootstrap failure is actionable and must stop before package changes.
        if download.returncode != 0:
            raise HostedNotebookBootstrapError(
                "Unable to download the official uv installer: " + download.stderr[-1000:]
            )
        # Execute only the just-downloaded official installer through the resolved
        # system shell; the temporary path is generated locally rather than supplied
        # by notebook input.
        install = subprocess.run(  # noqa: S603
            [str(shell), str(installer)], capture_output=True, text=True, check=False
        )
        # Do not infer success from the script's presence; require its zero exit status.
        if install.returncode != 0:
            raise HostedNotebookBootstrapError(
                "The official uv installer failed: " + install.stderr[-1000:]
            )
    installed = Path.home() / ".local/bin/uv"
    # Verify the installer's documented destination before returning a command path.
    if not installed.is_file():
        raise HostedNotebookBootstrapError(
            "The uv installer completed but ~/.local/bin/uv was not created"
        )
    return installed


def _run_logged(command: Sequence[str], *, root: Path, log: IO[str], phase: str) -> None:
    """Run one strict hosted bootstrap phase with complete output in one log.

    Args:
        command: Argument vector to execute without a shell.
        root: Checkout used as the command working directory.
        log: Open text log receiving stdout and stderr.
        phase: Human-readable phase name for failures.
    """

    # Every caller passes an internally constructed no-shell argument vector whose
    # first member is an absolute uv or Python executable. Reject future relative
    # commands so a changed PATH cannot redirect this privileged hosted operation.
    if not command or not Path(command[0]).is_absolute():
        raise HostedNotebookBootstrapError(f"{phase} requires an absolute executable path")
    # Combining stdout/stderr preserves command chronology for diagnostics while
    # keeping successful notebook output bounded.
    result = subprocess.run(  # noqa: S603
        list(command),
        cwd=root,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    # Surface the phase and retained log rather than replaying thousands of success lines.
    if result.returncode != 0:
        raise HostedNotebookBootstrapError(
            f"{phase} failed with exit code {result.returncode}; review {log.name}"
        )


def _verify_fresh_process(root: Path, log: IO[str]) -> None:
    """Verify the newly installed stack in a fresh process before requesting restart.

    Args:
        root: Checkout owning the editable project.
        log: Open bootstrap log receiving verification output.
    """

    # A child interpreter imports from the new filesystem state without inheriting
    # the live notebook kernel's already-loaded extension modules.
    verification = (
        "import ecg_anomaly_detection, numpy, scipy, sklearn; "
        "print(ecg_anomaly_detection.__file__); "
        "print(numpy.__version__, scipy.__version__, sklearn.__version__)"
    )
    _run_logged(
        [sys.executable, "-c", verification],
        root=root,
        log=log,
        phase="fresh-process import verification",
    )


def prepare_hosted_environment(
    repository_root: Path,
    *,
    kernel_pid: int,
    marker_path: Path,
    log_path: Path,
) -> HostedBootstrapResult:
    """Install/verify locked hosted dependencies and enforce a one-time restart.

    Args:
        repository_root: Exact checkout to install editably.
        kernel_pid: Current notebook kernel process ID.
        marker_path: Runtime-local install/restart marker.
        log_path: Runtime-local full command log.

    Returns:
        Readiness or restart-required result.
    """

    root = _repository_root(repository_root)
    fingerprint = environment_fingerprint(root)
    marker = _load_marker(marker_path)
    # A matching marker made by a different kernel proves the dependency files have
    # survived a kernel restart; the current kernel may now import them safely.
    current_kernel = _kernel_identity(kernel_pid)
    installed_by_value = marker.get("installed_by_kernel") if marker is not None else None
    # All identity fields must agree before any previous installation can be reused.
    if (
        marker is not None
        and marker.get("schema_version") == HOSTED_BOOTSTRAP_SCHEMA_VERSION
        and marker.get("fingerprint") == fingerprint
        and isinstance(installed_by_value, str)
    ):
        installed_by = installed_by_value
        status = "ready" if installed_by != current_kernel else "restart_required"
        return HostedBootstrapResult(
            status=status,
            fingerprint=fingerprint,
            log_path=log_path,
            installed_by_kernel=installed_by,
        )

    uv = _uv_executable()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    requirements = log_path.with_suffix(".requirements.txt")
    # Truncate the runtime-local log for this exact installation attempt; failures
    # retain every dependency resolver/install diagnostic for review.
    with log_path.open("w", encoding="utf-8") as log:
        _run_logged(
            [
                str(uv),
                "export",
                "--locked",
                "--no-default-groups",
                "--group",
                "notebooks",
                "--no-emit-project",
                "--format",
                "requirements-txt",
                "--output-file",
                str(requirements),
            ],
            root=root,
            log=log,
            phase="locked dependency export",
        )
        _run_logged(
            [
                str(uv),
                "pip",
                "install",
                "--system",
                "--require-hashes",
                "--requirements",
                str(requirements),
            ],
            root=root,
            log=log,
            phase="locked hosted dependency installation",
        )
        _run_logged(
            [str(uv), "pip", "install", "--system", "--no-deps", "--editable", "."],
            root=root,
            log=log,
            phase="editable project installation",
        )
        _verify_fresh_process(root, log)

    _write_marker(
        marker_path,
        {
            "schema_version": HOSTED_BOOTSTRAP_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "installed_by_kernel": current_kernel,
            "repository_commit": _git_commit(root),
        },
    )
    return HostedBootstrapResult(
        status="restart_required",
        fingerprint=fingerprint,
        log_path=log_path,
        installed_by_kernel=current_kernel,
    )
