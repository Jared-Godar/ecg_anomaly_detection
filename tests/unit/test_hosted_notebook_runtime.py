"""Tests for hosted notebook dependency installation and restart enforcement."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import IO

import pytest

from ecg_anomaly_detection import hosted_notebook_runtime
from ecg_anomaly_detection.hosted_notebook_runtime import (
    HostedNotebookBootstrapError,
    environment_fingerprint,
    prepare_hosted_environment,
)

# A deterministic source revision makes environment fingerprints stable in tests.
SOURCE_COMMIT = "c" * 40


def _init_repository(root: Path) -> None:
    """Create the two files required to identify a locked notebook environment.

    Args:
        root: Temporary checkout root.
    """

    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")


def test_environment_fingerprint_changes_with_lock_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dependency lock changes invalidate a prior hosted-runtime marker.

    Args:
        tmp_path: Isolated checkout root.
        monkeypatch: Replaces Git discovery with a deterministic commit.
    """

    _init_repository(tmp_path)
    monkeypatch.setattr(hosted_notebook_runtime, "_git_commit", lambda _root: SOURCE_COMMIT)
    before = environment_fingerprint(tmp_path)

    (tmp_path / "uv.lock").write_text("version = 2\n", encoding="utf-8")

    assert environment_fingerprint(tmp_path) != before


def test_hosted_install_requires_different_kernel_before_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful install remains restart-required until a new kernel PID calls it.

    Args:
        tmp_path: Isolated checkout, marker, and log paths.
        monkeypatch: Replaces network/process work with recorded deterministic calls.
    """

    root = tmp_path / "repository"
    _init_repository(root)
    marker = tmp_path / "runtime/marker.json"
    log = tmp_path / "runtime/bootstrap.log"
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(hosted_notebook_runtime, "_git_commit", lambda _root: SOURCE_COMMIT)
    monkeypatch.setattr(
        hosted_notebook_runtime,
        "_kernel_identity",
        lambda kernel_pid: f"pid:{kernel_pid}:start:test",
    )
    monkeypatch.setattr(hosted_notebook_runtime, "_uv_executable", lambda: Path("/fake/uv"))

    def record_command(command: Sequence[str], *, root: Path, log: IO[str], phase: str) -> None:
        """Record one would-be uv command without changing the test interpreter.

        Args:
            command: Command argument vector.
            root: Repository command working directory.
            log: Open test bootstrap log.
            phase: Human-readable command phase.
        """

        assert root == root.resolve()
        assert phase
        assert log.writable()
        calls.append(tuple(command))

    monkeypatch.setattr(hosted_notebook_runtime, "_run_logged", record_command)
    monkeypatch.setattr(hosted_notebook_runtime, "_verify_fresh_process", lambda _root, _log: None)

    first = prepare_hosted_environment(root, kernel_pid=100, marker_path=marker, log_path=log)
    same_kernel = prepare_hosted_environment(root, kernel_pid=100, marker_path=marker, log_path=log)
    restarted_kernel = prepare_hosted_environment(
        root, kernel_pid=200, marker_path=marker, log_path=log
    )

    assert first.status == "restart_required"
    assert same_kernel.status == "restart_required"
    assert restarted_kernel.status == "ready"
    assert len(calls) == 3
    assert calls[0][:3] == ("/fake/uv", "export", "--locked")
    assert calls[1][:4] == ("/fake/uv", "pip", "install", "--system")
    assert "--editable" in calls[2]
    assert json.loads(marker.read_text())["installed_by_kernel"] == "pid:100:start:test"


def test_fingerprint_change_reinstalls_and_requires_another_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed lock cannot reuse readiness from a different kernel's stale marker.

    Args:
        tmp_path: Isolated checkout, marker, and log paths.
        monkeypatch: Replaces network/process work with no-op test doubles.
    """

    root = tmp_path / "repository"
    _init_repository(root)
    marker = tmp_path / "runtime/marker.json"
    log = tmp_path / "runtime/bootstrap.log"
    monkeypatch.setattr(hosted_notebook_runtime, "_git_commit", lambda _root: SOURCE_COMMIT)
    monkeypatch.setattr(
        hosted_notebook_runtime,
        "_kernel_identity",
        lambda kernel_pid: f"pid:{kernel_pid}:start:test",
    )
    monkeypatch.setattr(hosted_notebook_runtime, "_uv_executable", lambda: Path("/fake/uv"))
    monkeypatch.setattr(
        hosted_notebook_runtime,
        "_run_logged",
        lambda _command, *, root, log, phase: None,
    )
    monkeypatch.setattr(hosted_notebook_runtime, "_verify_fresh_process", lambda _root, _log: None)
    prepare_hosted_environment(root, kernel_pid=100, marker_path=marker, log_path=log)
    (root / "uv.lock").write_text("version = 2\n", encoding="utf-8")

    changed = prepare_hosted_environment(root, kernel_pid=200, marker_path=marker, log_path=log)

    assert changed.status == "restart_required"
    assert json.loads(marker.read_text())["installed_by_kernel"] == "pid:200:start:test"


def test_hosted_bootstrap_rejects_checkout_without_lock(tmp_path: Path) -> None:
    """Missing lock metadata blocks hosted installation before any package changes.

    Args:
        tmp_path: Isolated incomplete checkout.
    """

    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")

    # The public exception keeps the failure at repository validation, before install.
    with pytest.raises(HostedNotebookBootstrapError, match="requires pyproject.toml and uv.lock"):
        environment_fingerprint(tmp_path)
