"""Test the notebook-cell extension of the internal-commentary policy checker."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import nbformat
import pytest

# Standalone repository scripts are not installed as package modules, so load the
# checker through its concrete path exactly as production-adjacent script tests do.
_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "check_notebook_commentary.py"
# Preserve the import specification so the loader and module registration use one identity.
_SPEC = importlib.util.spec_from_file_location("check_notebook_commentary", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
# Register the dynamically created module before execution because the module's own
# sibling-script loading (of check_code_commentary.py) runs as top-level code.
notebook_commentary = importlib.util.module_from_spec(_SPEC)
# Register under sys.modules before exec_module for the same reason the loaded
# module's own sibling-script loading needs it discoverable during import.
sys.modules[_SPEC.name] = notebook_commentary
_SPEC.loader.exec_module(notebook_commentary)

# Bind the public script functions locally so individual tests read like ordinary unit tests.
audit_notebook = notebook_commentary.audit_notebook
# Expose cell extraction separately so its ordering/filtering contract has its own tests.
extract_code_cells = notebook_commentary.extract_code_cells
# Expose discovery separately because its directory-exclusion contract is isolated.
discover_notebooks = notebook_commentary.discover_notebooks
# Expose the command entry point so tests can verify process-status behavior without a subprocess.
main = notebook_commentary.main


def _write_notebook(
    path: Path, cell_sources: list[str], cell_types: list[str] | None = None
) -> None:
    """Write one notebook with the given cell sources (all code cells by default).

    Args:
        path: Where to write the notebook; parent directories are created as needed.
        cell_sources: Source text for each cell, in order.
        cell_types: Matching cell types ("code" or "markdown"); defaults to all-code.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    types = cell_types or ["code"] * len(cell_sources)
    cells = [
        nbformat.v4.new_code_cell(source)
        if kind == "code"
        else nbformat.v4.new_markdown_cell(source)
        for source, kind in zip(cell_sources, types, strict=True)
    ]
    nbformat.write(nbformat.v4.new_notebook(cells=cells), path)


def test_audit_notebook_reports_missing_docstrings_and_comments(tmp_path: Path) -> None:
    """Verify a code cell with an undocumented function and a bare block is flagged.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    path = tmp_path / "notebooks/sample.ipynb"
    _write_notebook(
        path, ["def undocumented(flag):\n    if flag:\n        return 1\n    return 0\n"]
    )

    violations = audit_notebook(path)

    assert [item.message for item in violations] == [
        "function 'undocumented' is missing a docstring",
        "If block is missing a leading explanation comment",
    ]
    # The cell-scoped synthetic path lets a reviewer locate the exact cell.
    assert violations[0].path == Path(f"{path.as_posix()}::cell:0")


def test_audit_notebook_does_not_require_a_module_docstring(tmp_path: Path) -> None:
    """Verify a fully narrated code cell with no module docstring still passes.

    A code cell has no equivalent of a module docstring -- markdown cells carry the
    notebook's own narrative, and are outside this policy's Python-only scope.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    path = tmp_path / "notebooks/sample.ipynb"
    _write_notebook(path, ["# Keep the fixture constant explicit.\nVALUE = 1\n"])

    assert audit_notebook(path) == ()


def test_audit_notebook_skips_cell_magics_without_reporting_syntax_errors(tmp_path: Path) -> None:
    """Verify a %%bash cell is skipped instead of being misreported as invalid Python.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    path = tmp_path / "notebooks/sample.ipynb"
    _write_notebook(
        path,
        [
            "%%bash\nset -euo pipefail\necho hello\n",
            "# Keep the fixture constant explicit.\nVALUE = 1\n",
        ],
    )

    # The %%bash cell contributes no violations of its own (it isn't Python), and the
    # second, fully narrated cell also contributes none.
    assert audit_notebook(path) == ()


def test_audit_notebook_ignores_markdown_cells(tmp_path: Path) -> None:
    """Verify markdown cells never reach the Python audit, even with undocumented-looking prose.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    path = tmp_path / "notebooks/sample.ipynb"
    _write_notebook(
        path,
        ["def undocumented():\n    pass\n", "def also_undocumented():\n    pass\n"],
        cell_types=["markdown", "code"],
    )

    violations = audit_notebook(path)

    # Only the code cell (index 0 among code cells) is audited; the markdown cell
    # preceding it in the notebook is invisible to this checker.
    assert [item.path for item in violations] == [Path(f"{path.as_posix()}::cell:0")]


def test_extract_code_cells_returns_only_code_cells_in_order(tmp_path: Path) -> None:
    """Verify cell extraction indexes only code cells, skipping markdown cells.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    path = tmp_path / "notebooks/sample.ipynb"
    _write_notebook(
        path,
        ["# Narration", "first = 1\n", "# More narration", "second = 2\n"],
        cell_types=["markdown", "code", "markdown", "code"],
    )

    cells = extract_code_cells(path)

    assert [index for index, _ in cells] == [0, 1]
    assert cells[0][1] == "first = 1\n"
    assert cells[1][1] == "second = 2\n"


def test_discover_notebooks_excludes_local_and_archive_directories(tmp_path: Path) -> None:
    """Verify directory discovery skips notebooks/local/ and archive/original_2022/.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    curated = tmp_path / "notebooks/curated.ipynb"
    local = tmp_path / "notebooks/local/scratch.ipynb"
    archived = tmp_path / "archive/original_2022/old.ipynb"
    _write_notebook(curated, ["value = 1\n"])
    _write_notebook(local, ["value = 1\n"])
    _write_notebook(archived, ["value = 1\n"])

    assert discover_notebooks((tmp_path,)) == (curated,)


def test_main_fails_when_no_notebooks_are_discovered(tmp_path: Path) -> None:
    """Verify an empty audit target cannot masquerade as successful coverage.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    # Exit status two distinguishes configuration failure from policy violations.
    assert main([str(tmp_path)]) == 2


def test_main_passes_for_a_fully_narrated_notebook(tmp_path: Path) -> None:
    """Verify a clean notebook produces the passing exit status.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    path = tmp_path / "notebooks/sample.ipynb"
    _write_notebook(path, ["# Keep the fixture constant explicit.\nVALUE = 1\n"])

    assert main([str(path)]) == 0


@pytest.mark.parametrize(
    "notebook_relative_path",
    [
        "notebooks/00-environment-setup-and-artifact-generation.ipynb",
        "notebooks/01-narrative-walkthrough.ipynb",
        "notebooks/02-high-performing-gradient-boosting-validation.ipynb",
    ],
)
def test_real_curated_notebooks_pass_the_commentary_audit(notebook_relative_path: str) -> None:
    """Verify each real curated notebook satisfies the commentary policy (issue #151).

    Exercising the real repository files, not synthetic fixtures, is what proves the
    completed audit -- a passing synthetic test alone couldn't catch a real notebook
    regressing below the standard.
    """

    repository_root = Path(__file__).resolve().parents[2]
    path = repository_root / notebook_relative_path

    assert audit_notebook(path) == ()
