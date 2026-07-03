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
    source_dir = tmp_path_factory.mktemp("wfdb-source")
    record_ids = ("100", "101", "102")
    for record_id in record_ids:
        _write_synthetic_record(source_dir, record_id)
    payloads = {
        path.name: path.read_bytes()
        for path in source_dir.iterdir()
        if path.suffix in {".atr", ".dat", ".hea"}
    }
    _initialize_repository(tmp_path, record_ids)
    calls: list[str] = []

    def fake_fetch(url: str, output: Path, _: float, __: int) -> TransferResult:
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
    for run_directory in run_directories:
        assert len(tuple((run_directory / "validation").glob("*.json"))) == 3
        assert len(tuple((run_directory / "mapping").glob("*.json"))) == 3
        assert len(tuple((run_directory / "windows").glob("*.json"))) == 3
        split = json.loads((run_directory / "split.json").read_text(encoding="utf-8"))
        manifest = json.loads((run_directory / "run-manifest.json").read_text(encoding="utf-8"))
        dataset_index = json.loads(
            (
                tmp_path / "data" / "processed" / "runs" / run_directory.name / "dataset-index.json"
            ).read_text(encoding="utf-8")
        )
        assert split["total_record_count"] == 3
        assert split["total_window_count"] == 9
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
        assert len(manifest["evidence_files"]) == 10
        assert any(
            item["path"].endswith("evaluation/validation-metrics.json")
            for item in manifest["artifact_files"]
        )


def _initialize_repository(root: Path, record_ids: tuple[str, ...]) -> None:
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
channel_index = 0
boundary_policy = "exclude"
""".strip(),
        encoding="utf-8",
    )
    (configs / "split.toml").write_text(
        """
schema_version = 1
[split]
name = "synthetic-grouped"
version = "1.0.0"
strategy = "seeded-record-shuffle"
seed = 7
[split.ratios]
train = 0.6
validation = 0.2
test = 0.2
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
    subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
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
