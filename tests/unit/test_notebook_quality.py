from __future__ import annotations

import json
from pathlib import Path

import nbformat
import pytest

from ecg_anomaly_detection.notebook_quality import (
    NotebookQualityError,
    check_notebooks,
    discover_local_notebooks,
)


def _write_notebook(path: Path, *, with_output: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cell = nbformat.v4.new_code_cell("value = 1")
    if with_output:
        cell.execution_count = 3
        cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text="one\n")]
    notebook = nbformat.v4.new_notebook(cells=[cell])
    nbformat.write(notebook, path)


def test_valid_notebook_reports_static_hygiene_without_execution(tmp_path: Path) -> None:
    path = tmp_path / "notebooks/local/example.ipynb"
    _write_notebook(path, with_output=True)

    summary = check_notebooks(tmp_path, [path])

    assert summary.valid
    assert summary.changed_count == 0
    report = summary.notebooks[0]
    assert report.cell_count == 1
    assert report.code_cell_count == 1
    assert report.output_count == 1
    assert report.output_bytes > 0
    assert report.execution_count_count == 1
    assert {issue.code for issue in report.issues} == {
        "execution-counts",
        "saved-outputs",
        "untrusted-cells",
    }


def test_malformed_notebook_fails_validation(tmp_path: Path) -> None:
    path = tmp_path / "notebooks/local/broken.ipynb"
    path.parent.mkdir(parents=True)
    path.write_text("{not valid JSON", encoding="utf-8")

    summary = check_notebooks(tmp_path, [path])

    assert not summary.valid
    assert summary.notebooks[0].issues[0].code == "invalid-notebook"


def test_formatting_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "notebooks/local/unformatted.ipynb"
    _write_notebook(path)
    notebook = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(notebook, separators=(",", ":")), encoding="utf-8")

    first = check_notebooks(tmp_path, [path], format_notebooks=True)
    formatted = path.read_bytes()
    second = check_notebooks(tmp_path, [path], format_notebooks=True)

    assert first.changed_count == 1
    assert second.changed_count == 0
    assert path.read_bytes() == formatted


def test_output_stripping_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "notebooks/local/outputs.ipynb"
    _write_notebook(path, with_output=True)

    first = check_notebooks(tmp_path, [path], strip_outputs=True)
    second = check_notebooks(tmp_path, [path], strip_outputs=True)
    stripped = nbformat.read(path, as_version=nbformat.NO_CONVERT)

    assert first.changed_count == 1
    assert second.changed_count == 0
    assert stripped.cells[0].outputs == []
    assert stripped.cells[0].execution_count is None


def test_discovery_excludes_narrative_by_default(tmp_path: Path) -> None:
    local = tmp_path / "notebooks/local/local.ipynb"
    narrative = tmp_path / "notebooks/narrative-walkthrough.ipynb"
    _write_notebook(local)
    _write_notebook(narrative)

    assert discover_local_notebooks(tmp_path) == (local,)
    assert discover_local_notebooks(tmp_path, include_narrative=True) == (local, narrative)


def test_repository_boundary_rejects_non_notebook_path(tmp_path: Path) -> None:
    path = tmp_path / "outside.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(NotebookQualityError, match="under notebooks"):
        check_notebooks(tmp_path, [path])


def test_machine_local_path_and_stale_kernel_are_reported(tmp_path: Path) -> None:
    path = tmp_path / "notebooks/local/paths.ipynb"
    _write_notebook(path)
    notebook = nbformat.read(path, as_version=nbformat.NO_CONVERT)
    notebook.cells[0].source = 'data = Path("/Users/example/private/data.csv")'
    notebook.metadata.kernelspec = {"display_name": "Old", "name": "old-environment"}
    nbformat.write(notebook, path)

    report = check_notebooks(tmp_path, [path]).notebooks[0]

    assert {issue.code for issue in report.issues} >= {
        "machine-local-path",
        "stale-kernel-metadata",
    }
