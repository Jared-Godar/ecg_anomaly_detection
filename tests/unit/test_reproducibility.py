"""Tests for operational reproducibility evidence contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import ecg_anomaly_detection.reproducibility as reproducibility
from ecg_anomaly_detection.reproducibility import (
    RuntimeStageTimer,
    capture_environment_summary,
    capture_git_metadata,
    capture_resource_summary,
    collect_artifact_evidence,
    sha256_file,
)


def test_file_digest_and_artifact_evidence_are_stable(tmp_path: Path) -> None:
    content = b"reproducible\n"
    artifact = tmp_path / "artifact.json"
    artifact.write_bytes(content)

    evidence = collect_artifact_evidence(tmp_path, (artifact,))[0]

    assert sha256_file(artifact) == hashlib.sha256(content).hexdigest()
    assert evidence.path == "artifact.json"
    assert evidence.size_bytes == len(content)
    assert evidence.sha256 == hashlib.sha256(content).hexdigest()


def test_runtime_summary_has_fixed_stages_and_deterministic_json() -> None:
    readings = iter((0.0, 1.0, 1.25, 2.0))
    timer = RuntimeStageTimer(lambda: next(readings))

    with timer.stage("acquisition"):
        pass
    summary = timer.summary()

    parsed = json.loads(summary.to_json())
    assert parsed["schema_version"] == 1
    assert parsed["duration_unit"] == "seconds"
    assert parsed["stage_durations"]["acquisition"] == 0.25
    assert parsed["stage_durations"]["validation_evaluation"] == 0.0
    assert parsed["total_runtime"] == 2.0
    assert summary.to_json() == summary.to_json()
    assert summary.to_json().endswith("\n")


def test_environment_capture_uses_null_fallbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    monkeypatch.setattr(reproducibility, "_run_optional", lambda *_: None)
    monkeypatch.setattr(reproducibility, "_run_optional_allow_empty", lambda *_: None)

    summary = capture_environment_summary(tmp_path)

    assert summary.schema_version == 1
    assert summary.uv_version is None
    assert summary.git.commit is None
    assert summary.git.branch is None
    assert summary.git.dirty is None
    assert summary.dependency_lock.path == "uv.lock"


def test_git_metadata_preserves_clean_status_when_other_values_are_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(command: tuple[str, ...], _: Path) -> str | None:
        return "" if command[1] == "status" else None

    monkeypatch.setattr(reproducibility, "_run_optional", fake_run)
    monkeypatch.setattr(reproducibility, "_run_optional_allow_empty", fake_run)

    metadata = capture_git_metadata(tmp_path)

    assert metadata.commit is None
    assert metadata.branch is None
    assert metadata.dirty is False


def test_resource_capture_uses_null_fallbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable_disk(_: Path) -> None:
        raise OSError("unavailable")

    monkeypatch.setattr(reproducibility.shutil, "disk_usage", unavailable_disk)
    monkeypatch.setattr(reproducibility, "_capture_cpu_model", lambda: None)
    monkeypatch.setattr(reproducibility, "_capture_memory_total", lambda: None)
    monkeypatch.setattr(reproducibility.os, "cpu_count", lambda: None)

    summary = capture_resource_summary(tmp_path)

    assert summary.schema_version == 1
    assert summary.cpu_model is None
    assert summary.logical_core_count is None
    assert summary.memory_total_bytes is None
    assert summary.disk_total_bytes is None
    assert summary.disk_used_bytes is None
    assert summary.disk_free_bytes is None
