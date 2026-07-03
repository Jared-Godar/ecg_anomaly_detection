"""Tests for pipeline orchestration path and identity boundaries."""

from pathlib import Path

import pytest

from ecg_anomaly_detection.pipeline import PipelineError, run_pipeline


def test_pipeline_rejects_configuration_outside_repository(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    external = tmp_path_factory.mktemp("external-config") / "dataset.toml"
    external.write_text("schema_version = 1\n", encoding="utf-8")

    with pytest.raises(PipelineError, match="within repository root"):
        run_pipeline(
            tmp_path,
            external,
            Path("configs/mapping.toml"),
            Path("configs/window.toml"),
            Path("configs/split.toml"),
            Path("configs/training.toml"),
            Path("configs/evaluation.toml"),
        )
