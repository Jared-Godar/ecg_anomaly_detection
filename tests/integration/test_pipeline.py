"""End-to-end integration coverage for the supported local data pipeline."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import wfdb

import ecg_anomaly_detection.acquisition as acquisition
from ecg_anomaly_detection.acquisition import TransferResult
from ecg_anomaly_detection.cli import main


def test_pipeline_command_connects_all_supported_stages_without_network(
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A full `run-pipeline` invocation, run twice, exercises every stage and stays idempotent.

    The most comprehensive test in this repository: seeds three synthetic WFDB records,
    runs the entire acquisition-through-evaluation pipeline via the CLI (with only the
    HTTPS fetcher faked out, so every other stage runs its real, unmodified code path),
    then re-runs the identical command a second time to confirm acquisition's
    verify-and-reuse resume path works end to end (no re-downloads) while still
    producing a fresh, independent run directory and manifest.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory, used as the fake
            repository root.
        tmp_path_factory: Used to create a separate staging directory for the
            synthetic WFDB source records, outside the fake repository.
        monkeypatch: Used to replace the real HTTPS fetcher with a fake serving the
            synthetic records' bytes.
    """

    source_dir = tmp_path_factory.mktemp("wfdb-source")
    record_ids = ("100", "101", "102")
    # Generate one synthetic WFDB record per configured record ID.
    for record_id in record_ids:
        _write_synthetic_record(source_dir, record_id)
    payloads = {
        path.name: path.read_bytes()
        for path in source_dir.iterdir()
        if path.suffix in {".atr", ".dat", ".hea"}
    }
    _initialize_repository(tmp_path, record_ids, payloads)
    calls: list[str] = []

    def fake_fetch(url: str, output: Path, _: float, __: int) -> TransferResult:
        """Serve pre-generated synthetic bytes instead of making a real HTTPS request.

        Records every requested URL in `calls` so the test can assert the second
        `run-pipeline` invocation makes zero fetch calls (proving the resume path
        reused files rather than re-downloading).

        Args:
            url: The requested download URL, recorded into `calls`.
            output: Where to write the served synthetic content.
            _: Timeout, unused by this fake.
            __: Size cap, unused by this fake.

        Returns:
            The digest of the served content.
        """

        content = payloads[output.name]
        output.write_bytes(content)
        calls.append(url)
        return TransferResult(len(content), hashlib.sha256(content).hexdigest())

    monkeypatch.setattr(acquisition, "_fetch_https_file", fake_fetch)
    arguments = [
        "run-pipeline",
        "--repository-root",
        str(tmp_path),
        "--dataset-config",
        "configs/dataset.toml",
        "--mapping-config",
        "configs/mapping.toml",
        "--window-config",
        "configs/window.toml",
        "--split-config",
        "configs/split.toml",
        "--training-config",
        "configs/training.toml",
        "--evaluation-config",
        "configs/evaluation.toml",
    ]

    first_exit_code = main(arguments)
    second_exit_code = main(arguments)

    run_directories = sorted((tmp_path / "artifacts" / "runs").iterdir())
    assert first_exit_code == 0
    assert second_exit_code == 0
    assert len(calls) == 9
    assert len(run_directories) == 2
    # Verify both runs independently: each run gets its own directory, manifest, and
    # complete set of stage outputs, even though the second run reused acquisition's
    # cached files rather than re-downloading them.
    for run_directory in run_directories:
        assert len(tuple((run_directory / "validation").glob("*.json"))) == 3
        assert len(tuple((run_directory / "mapping").glob("*.json"))) == 3
        assert len(tuple((run_directory / "windows").glob("*.json"))) == 3
        split = json.loads((run_directory / "split.json").read_text(encoding="utf-8"))
        split_quality = json.loads(
            (run_directory / "split_quality_summary.json").read_text(encoding="utf-8")
        )
        manifest = json.loads((run_directory / "run-manifest.json").read_text(encoding="utf-8"))
        environment = json.loads(
            (run_directory / "environment_summary.json").read_text(encoding="utf-8")
        )
        runtime = json.loads((run_directory / "runtime_summary.json").read_text(encoding="utf-8"))
        resources = json.loads(
            (run_directory / "resource_summary.json").read_text(encoding="utf-8")
        )
        evidence_manifest = json.loads(
            (run_directory / "evidence_manifest.json").read_text(encoding="utf-8")
        )
        dataset_index = json.loads(
            (
                tmp_path / "data" / "processed" / "runs" / run_directory.name / "dataset-index.json"
            ).read_text(encoding="utf-8")
        )
        assert split["total_record_count"] == 3
        assert split["total_window_count"] == 9
        assert split_quality["status"] == "passed"
        assert manifest["run_id"] == run_directory.name
        assert manifest["git"]["dirty"] is False
        assert manifest["dataset"]["dataset_slug"] == "synthetic"
        assert dataset_index["total_record_count"] == 3
        assert dataset_index["total_window_count"] == 9
        training_metadata = json.loads(
            (run_directory / "training" / "training-metadata.json").read_text(encoding="utf-8")
        )
        assert training_metadata["partition"] == "train"
        assert (run_directory / "training" / "model.json").is_file()
        metrics = json.loads(
            (run_directory / "evaluation" / "validation-metrics.json").read_text(encoding="utf-8")
        )
        assert metrics["partition"] == "validation"
        assert metrics["window_count"] == 3
        assert len(manifest["artifact_files"]) == 7
        assert any(
            item["path"].endswith("training/model.json") for item in manifest["artifact_files"]
        )
        assert len(manifest["evidence_files"]) == 15
        assert any(
            item["path"].endswith("split_quality_summary.json")
            for item in manifest["evidence_files"]
        )
        assert any(
            item["path"].endswith("evaluation/validation-metrics.json")
            for item in manifest["artifact_files"]
        )
        assert environment["schema_version"] == 1
        assert environment["dependency_lock"]["path"] == "uv.lock"
        assert runtime["schema_version"] == 1
        assert set(runtime["stage_durations"]) == {
            "acquisition",
            "validation",
            "annotation_mapping",
            "window_extraction",
            "split",
            "split_diagnostics",
            "training",
            "validation_evaluation",
        }
        assert resources["schema_version"] == 1
        assert evidence_manifest["schema_version"] == 1
        assert evidence_manifest["split"]["name"] == "synthetic-subject-grouped"
        assert len(evidence_manifest["configuration_files"]) == 6
        assert len(evidence_manifest["evidence_files"]) == 14
        assert len(evidence_manifest["artifact_files"]) == 7
        assert all(len(item["sha256"]) == 64 for item in evidence_manifest["artifact_files"])


def test_run_pipeline_command_emits_stage_progress_and_total_elapsed_time(
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`run-pipeline`'s stdout reports every stage's start/completion and a final elapsed time.

    Protects the ProgressReporter integration end to end: every one of the 7 reported
    stages must print both a "starting" and "complete in" line, per-record progress
    notes must appear for each configured record, and the final line must report the
    run's total elapsed time -- confirming CLI users actually see this feedback, not
    just that ProgressReporter's own unit tests pass in isolation.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory, used as the fake
            repository root.
        tmp_path_factory: Used to create a separate staging directory for the
            synthetic WFDB source records, outside the fake repository.
        monkeypatch: Used to replace the real HTTPS fetcher with a fake serving the
            synthetic records' bytes.
        capsys: Used to capture the CLI's stdout for the assertions below.
    """

    source_dir = tmp_path_factory.mktemp("wfdb-source")
    record_ids = ("100", "101", "102")
    # Generate one synthetic WFDB record per configured record ID.
    for record_id in record_ids:
        _write_synthetic_record(source_dir, record_id)
    payloads = {
        path.name: path.read_bytes()
        for path in source_dir.iterdir()
        if path.suffix in {".atr", ".dat", ".hea"}
    }
    _initialize_repository(tmp_path, record_ids, payloads)

    def fake_fetch(url: str, output: Path, _: float, __: int) -> TransferResult:
        """Serve pre-generated synthetic bytes instead of making a real HTTPS request.

        Args:
            url: The requested download URL, unused by this fake (no call recording
                needed in this test, unlike the idempotency test above).
            output: Where to write the served synthetic content.
            _: Timeout, unused by this fake.
            __: Size cap, unused by this fake.

        Returns:
            The digest of the served content.
        """

        content = payloads[output.name]
        output.write_bytes(content)
        return TransferResult(len(content), hashlib.sha256(content).hexdigest())

    monkeypatch.setattr(acquisition, "_fetch_https_file", fake_fetch)
    arguments = [
        "run-pipeline",
        "--repository-root",
        str(tmp_path),
        "--dataset-config",
        "configs/dataset.toml",
        "--mapping-config",
        "configs/mapping.toml",
        "--window-config",
        "configs/window.toml",
        "--split-config",
        "configs/split.toml",
        "--training-config",
        "configs/training.toml",
        "--evaluation-config",
        "configs/evaluation.toml",
    ]

    exit_code = main(arguments)
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "run " in output.splitlines()[0]
    # Confirm every one of the 7 stages reported both a starting and completion line,
    # in the pipeline's own fixed stage order.
    for index, name in enumerate(
        (
            "acquisition",
            "inventory",
            "record_processing",
            "split",
            "split_diagnostics",
            "training",
            "validation_evaluation",
        ),
        start=1,
    ):
        assert f"[{index}/7] {name}: starting" in output
        assert f"[{index}/7] {name}: complete in" in output
    # Confirm per-record progress notes were printed for every configured record.
    for record_id in record_ids:
        assert f"({record_id}): " in output
    assert "completed run " in output.splitlines()[-1]
    assert " in " in output.splitlines()[-1]


def _initialize_repository(
    root: Path, record_ids: tuple[str, ...], payloads: dict[str, bytes]
) -> None:
    """Build a complete fake repository: skeleton dirs, all six pipeline configs, and a Git commit.

    Every config here is a minimal, internally consistent set matching the synthetic
    fixture data (three records, 4 Hz sample rate, tiny window/split geometry) so
    run_pipeline's real, unmodified code can execute against it end to end. A real Git
    commit is required since run_manifest.py's git-state capture shells out to `git
    rev-parse`/`git status`, which would otherwise report UNKNOWN_GIT_REVISION.

    Args:
        root: Repository root used to enforce path and trust boundaries.
        record_ids: The record IDs to declare in the dataset config.
        payloads: Every source file's content, used to compute expected_source_files digests.
    """

    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (root / ".gitignore").write_text(
        "/data/raw/**\n/data/interim/**\n/data/processed/**\n/artifacts/**\n",
        encoding="utf-8",
    )
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "interim").mkdir()
    (root / "data" / "processed").mkdir()
    (root / "artifacts").mkdir()
    configs = root / "configs"
    configs.mkdir()
    records = ", ".join(f'"{record_id}"' for record_id in record_ids)
    expected_files = ",\n".join(
        '  { path = "'
        + name
        + f'", size_bytes = {len(content)}, sha256 = "{hashlib.sha256(content).hexdigest()}" }}'
        for name, content in sorted(payloads.items())
    )
    (configs / "dataset.toml").write_text(
        f"""
schema_version = 1
[dataset]
name = "Synthetic WFDB fixture"
slug = "synthetic"
version = "1.0.0"
source_url = "https://example.test/content/synthetic/1.0.0/"
download_url = "https://example.test/files/synthetic/1.0.0/"
sample_rate_hz = 4
annotation_extension = "atr"
record_ids = [{records}]
required_extensions = ["atr", "dat", "hea"]
expected_source_files = [
{expected_files}
]
""".strip(),
        encoding="utf-8",
    )
    (configs / "mapping.toml").write_text(
        """
schema_version = 1
[mapping]
name = "synthetic-binary"
version = "1.0.0"
unknown_symbol_policy = "error"
[[targets]]
name = "reference_normal"
value = 0
symbols = ["N"]
[[targets]]
name = "selected_other"
value = 1
symbols = ["V"]
[exclusions]
symbols = ["!"]
""".strip(),
        encoding="utf-8",
    )
    (configs / "window.toml").write_text(
        """
schema_version = 1
[window]
name = "synthetic-one-second"
version = "1.0.0"
pre_seconds = 0.5
post_seconds = 0.5
channel_name = "MLII"
boundary_policy = "exclude"
""".strip(),
        encoding="utf-8",
    )
    (configs / "split.toml").write_text(
        """
schema_version = 2
[split]
name = "synthetic-subject-grouped"
version = "2.0.0"
strategy = "seeded-subject-shuffle"
seed = 7
[split.ratios]
train = 0.6
validation = 0.2
test = 0.2
[record_subjects]
100 = "subject-100"
101 = "subject-101"
102 = "subject-102"
""".strip(),
        encoding="utf-8",
    )
    (configs / "training.toml").write_text(
        """
schema_version = 1
[training]
name = "synthetic-baseline"
version = "1.0.0"
estimator = "random-projection-nearest-centroid"
seed = 7
projection_components = 2
""".strip(),
        encoding="utf-8",
    )
    (configs / "evaluation.toml").write_text(
        """
schema_version = 1
[evaluation]
name = "synthetic-validation"
version = "1.0.0"
evaluator = "random-projection-nearest-centroid"
partition = "validation"
zero_division = 0.0
""".strip(),
        encoding="utf-8",
    )
    # every git invocation below is a fixed literal command list, not
    # runtime/user-constructed input.
    subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)  # noqa: S607
    subprocess.run(["git", "add", "."], cwd=root, check=True)  # noqa: S607
    subprocess.run(
        [  # noqa: S607
            "git",
            "-c",
            "user.name=Test User",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
        cwd=root,
        check=True,
    )


def _write_synthetic_record(directory: Path, record_id: str) -> None:
    """Write one 4-second synthetic WFDB record with three well-spaced N/V/N beats.

    Args:
        directory: Where to write the record's WFDB companion files.
        record_id: The record's base filename (matches the dataset config's record_ids).
    """

    sample_axis = np.linspace(0.0, 1.0, 16, endpoint=False)
    signals = np.column_stack((np.sin(2 * np.pi * sample_axis), np.cos(2 * np.pi * sample_axis)))
    wfdb.wrsamp(
        record_id,
        fs=4,
        units=["mV", "mV"],
        sig_name=["MLII", "V5"],
        p_signal=signals,
        write_dir=str(directory),
    )
    wfdb.wrann(
        record_id,
        "atr",
        sample=np.asarray([4, 8, 12]),
        symbol=["N", "V", "N"],
        write_dir=str(directory),
    )
