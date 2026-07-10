"""Tests for pipeline orchestration path and identity boundaries."""

from pathlib import Path

import pytest

from ecg_anomaly_detection.pipeline import PipelineError, run_pipeline


def test_pipeline_rejects_configuration_outside_repository(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """A dataset config path pointing outside the repository root is rejected before any stage runs.

    Every config path run_pipeline accepts must resolve inside the
    repository it was given, so a config from an unrelated location (e.g. a
    stray absolute path) can't silently be read.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory, used as
            the fixture repository root.
        tmp_path_factory: Used to create a second, unrelated temporary
            directory for the external config file.
    """

    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    external = tmp_path_factory.mktemp("external-config") / "dataset.toml"
    external.write_text("schema_version = 1\n", encoding="utf-8")

    # external lives under a separate tmp_path_factory directory, outside tmp_path.
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
