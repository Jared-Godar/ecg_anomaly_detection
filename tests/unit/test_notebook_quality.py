"""Exercise test notebook quality behavior and its regression contracts."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
import pytest

from ecg_anomaly_detection.notebook_quality import (
    NARRATIVE_NOTEBOOK,
    NotebookQualityError,
    check_notebooks,
    discover_local_notebooks,
)


def _write_notebook(path: Path, *, with_output: bool = False) -> None:
    """Write a minimal one-cell notebook (`value = 1`) to path, optionally with a saved output.

    Args:
        path: Where to write the notebook; parent directories are created as needed.
        with_output: If True, attach an execution_count and a stdout stream
            output to the cell, so tests can exercise output-related checks.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    cell = nbformat.v4.new_code_cell("value = 1")
    # Only attach execution state when the caller wants output-related checks exercised.
    if with_output:
        cell.execution_count = 3
        cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text="one\n")]
    notebook = nbformat.v4.new_notebook(cells=[cell])
    nbformat.write(notebook, path)


def test_valid_notebook_reports_static_hygiene_without_execution(tmp_path: Path) -> None:
    """A notebook with a saved output and execution count is reported with matching cell/output
    counts and the three hygiene issue codes that flag committed execution state.

    check_notebooks never executes cells (see this module's docstring
    guarantee); this test confirms it can still detect and report saved
    outputs, execution counts, and untrusted-cell metadata statically.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

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
    """A file that isn't valid notebook JSON at all fails validation with an "invalid-notebook" issue.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    path = tmp_path / "notebooks/local/broken.ipynb"
    path.parent.mkdir(parents=True)
    path.write_text("{not valid JSON", encoding="utf-8")

    summary = check_notebooks(tmp_path, [path])

    assert not summary.valid
    assert summary.notebooks[0].issues[0].code == "invalid-notebook"


def test_formatting_is_deterministic(tmp_path: Path) -> None:
    """Reformatting a compacted (no-whitespace) notebook changes it once; reformatting again is a no-op.

    Confirms format_notebooks is idempotent: the second call's
    changed_count is 0 and produces byte-identical output to the first
    call's result.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

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
    """Stripping outputs from a notebook with saved output changes it once; stripping again is a no-op.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

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
    """discover_local_notebooks finds only notebooks/local/ by default, and includes the
    curated narrative notebook only when explicitly asked.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    local = tmp_path / "notebooks/local/local.ipynb"
    # Matches NARRATIVE_NOTEBOOK's actual filename so this test would fail if that
    # constant drifted from the real curated notebook again (regression for #152,
    # where a stale constant silently matched nothing without the test noticing).
    narrative = tmp_path / "notebooks/01-narrative-walkthrough.ipynb"
    _write_notebook(local)
    _write_notebook(narrative)

    assert discover_local_notebooks(tmp_path) == (local,)
    assert discover_local_notebooks(tmp_path, include_narrative=True) == (local, narrative)


def test_narrative_notebook_constant_matches_real_repository_file() -> None:
    """NARRATIVE_NOTEBOOK must point at a file that actually exists in this checkout.

    A synthetic tmp_path fixture can't catch the constant drifting from the real
    curated notebook's filename (see #152); this test exercises the real repository
    root instead, so a rename of notebooks/01-narrative-walkthrough.ipynb without a
    matching constant update fails here.
    """

    repository_root = Path(__file__).resolve().parents[2]

    assert (repository_root / NARRATIVE_NOTEBOOK).is_file()


def test_repository_boundary_rejects_non_notebook_path(tmp_path: Path) -> None:
    """A path outside the notebooks/ directory is rejected before check_notebooks attempts to read it.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    path = tmp_path / "outside.json"
    path.write_text("{}", encoding="utf-8")

    # path is outside notebooks/, so it must be rejected regardless of its content.
    with pytest.raises(NotebookQualityError, match="under notebooks"):
        check_notebooks(tmp_path, [path])


def test_machine_local_path_and_stale_kernel_are_reported(tmp_path: Path) -> None:
    """A cell referencing an absolute /Users/... path and stale kernelspec metadata are both flagged.

    Both are portability hazards specific to the ignored notebooks/local/
    sandbox: a hard-coded personal path won't resolve on another machine,
    and stale kernel metadata can silently point a notebook at an
    environment that no longer matches this project's dependencies.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

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
