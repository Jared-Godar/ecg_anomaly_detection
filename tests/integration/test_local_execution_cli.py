"""Integration tests for the list-runs and purge-run CLI boundary."""

from __future__ import annotations

import json
from pathlib import Path

from ecg_anomaly_detection.cli import main

# A canonical lowercase UUID used as the single run under test in this file.
RUN_ID = "11111111-1111-1111-1111-111111111111"


def _init_repository(root: Path) -> None:
    """Mark root as a repository root by giving it a pyproject.toml, matching production layout.

    Args:
        root: Pytest's per-test isolated temporary directory.
    """

    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")


def _write_run(root: Path) -> None:
    """Create RUN_ID's three companion directories with a manifest and a small blob file.

    Args:
        root: The fixture repository root.
    """

    artifacts_dir = root / "artifacts" / "runs" / RUN_ID
    interim_dir = root / "data" / "interim" / "runs" / RUN_ID
    processed_dir = root / "data" / "processed" / "runs" / RUN_ID
    # Every run has all three companion directories, even if some end up empty.
    for directory in (artifacts_dir, interim_dir, processed_dir):
        directory.mkdir(parents=True)
    (artifacts_dir / "run-manifest.json").write_text("{}", encoding="utf-8")
    (interim_dir / "blob.bin").write_bytes(b"\0" * 5)


def test_list_runs_json_output_reports_the_run(tmp_path: Path, capsys) -> None:
    """`ecg-data list-runs --json` prints a JSON array whose one entry matches the fixture run's
    ID, manifest presence, and total size.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
        capsys: Used to capture and parse main's printed stdout as JSON.
    """

    _init_repository(tmp_path)
    _write_run(tmp_path)

    exit_code = main(["list-runs", "--repository-root", str(tmp_path), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert len(payload) == 1
    assert payload[0]["run_id"] == RUN_ID
    assert payload[0]["has_run_manifest"] is True
    assert payload[0]["total_size_bytes"] == 5 + len(b"{}")


def test_list_runs_text_output_reports_no_local_runs_when_empty(tmp_path: Path, capsys) -> None:
    """With no runs present, the default text-mode `list-runs` prints a "no local runs found" message.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
        capsys: Used to capture main's printed stdout.
    """

    _init_repository(tmp_path)

    exit_code = main(["list-runs", "--repository-root", str(tmp_path)])

    assert exit_code == 0
    assert "no local runs found" in capsys.readouterr().out


def test_purge_run_dry_run_leaves_directories_in_place(tmp_path: Path, capsys) -> None:
    """`purge-run --dry-run` reports what it would remove but leaves the run's directories on disk.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
        capsys: Used to capture main's printed stdout.
    """

    _init_repository(tmp_path)
    _write_run(tmp_path)

    exit_code = main(
        ["purge-run", "--repository-root", str(tmp_path), "--run-id", RUN_ID, "--dry-run"]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "would remove" in output.lower()
    assert (tmp_path / "artifacts" / "runs" / RUN_ID).is_dir()


def test_purge_run_removes_the_run_and_reports_it(tmp_path: Path, capsys) -> None:
    """`purge-run` (no --dry-run) deletes the run's directories, reports success, and a follow-up
    `list-runs` then reports no local runs at all.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
        capsys: Used to capture both commands' printed stdout in turn.
    """

    _init_repository(tmp_path)
    _write_run(tmp_path)

    exit_code = main(["purge-run", "--repository-root", str(tmp_path), "--run-id", RUN_ID])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "removed" in output.lower()
    assert not (tmp_path / "artifacts" / "runs" / RUN_ID).exists()

    follow_up_exit_code = main(["list-runs", "--repository-root", str(tmp_path)])
    assert follow_up_exit_code == 0
    assert "no local runs found" in capsys.readouterr().out


def test_purge_run_unknown_run_id_fails_with_nonzero_exit(tmp_path: Path, capsys) -> None:
    """`purge-run` for a run ID with no matching directory exits 1 and prints an actionable error.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
        capsys: Used to capture main's printed stderr.
    """

    _init_repository(tmp_path)

    exit_code = main(["purge-run", "--repository-root", str(tmp_path), "--run-id", RUN_ID])
    output = capsys.readouterr().err

    assert exit_code == 1
    assert "no local run directories found" in output
