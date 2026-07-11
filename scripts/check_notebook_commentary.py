#!/usr/bin/env python3
"""Extend the exhaustive internal-commentary policy to curated notebook code cells.

`check_code_commentary.py` only discovers `*.py` files, so the three curated,
public-facing notebooks under `notebooks/` were out of scope for issue #149's
repository-wide documentation pass. This script closes that gap for issue #151 by
reusing the same AST/tokenizer audit logic (`check_code_commentary.audit_source`)
against each code cell's source, extracted via `nbformat` without executing any
cell.

Cells using an IPython cell magic (e.g. `%%bash`) are not Python and are skipped:
their content cannot be parsed by `ast.parse`, and this checker's structural rules
(docstrings, comment-guarded blocks) are specific to Python syntax. They remain
part of the notebook's own review, just not this script's.

`notebooks/local/` (disposable experimentation) and `archive/original_2022/`
(immutable historical material) are outside this policy, matching
`check_code_commentary.py`'s own boundaries.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import nbformat

# check_code_commentary.py is a standalone script, not an installed package module;
# load it by path (mirroring tests/scripts/test_check_code_commentary.py's own
# loading pattern) so this script works the same way whether invoked directly,
# through pre-commit, or imported from its own test module.
_CHECK_CODE_COMMENTARY_PATH = Path(__file__).resolve().parent / "check_code_commentary.py"
# Build (rather than execute) the module spec first so a missing/unreadable sibling
# script fails through the explicit check below instead of a less specific AttributeError.
_SPEC = importlib.util.spec_from_file_location("check_code_commentary", _CHECK_CODE_COMMENTARY_PATH)
# A None spec or loader means the sibling script couldn't be found or isn't a loadable
# module; raise here with a specific message rather than continuing toward a fault.
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"could not load sibling script: {_CHECK_CODE_COMMENTARY_PATH}")
# Create the module object before executing it so its own top-level code (including
# the dataclass decorator on CommentaryViolation) can resolve its defining module.
_check_code_commentary = importlib.util.module_from_spec(_SPEC)
# Register under sys.modules before exec_module for the same reason the loaded
# module's own dataclasses need it discoverable during class-decorator evaluation.
sys.modules[_SPEC.name] = _check_code_commentary
_SPEC.loader.exec_module(_check_code_commentary)

# Reuse the exact violation type check_code_commentary.py already defined, so a
# notebook cell violation and a whole-file violation share one identical shape.
CommentaryViolation = _check_code_commentary.CommentaryViolation
# Reuse the exact audit implementation (not a parallel reimplementation) so notebook
# cells and standalone files are held to the identical docstring/comment standard.
audit_source = _check_code_commentary.audit_source

# The three curated, public-facing notebooks this script checks by default.
# notebooks/local/ (disposable) and archive/original_2022/ (immutable) are never
# discovered, matching this policy's repository-wide boundaries.
DEFAULT_NOTEBOOKS = (
    Path("notebooks/00-environment-setup-and-artifact-generation.ipynb"),
    Path("notebooks/01-narrative-walkthrough.ipynb"),
    Path("notebooks/02-high-performing-gradient-boosting-validation.ipynb"),
)


def _is_cell_magic(source: str) -> bool:
    """Return whether a code cell's source is an IPython cell magic, not Python.

    A cell magic (e.g. `%%bash`, `%%time`) occupies the entire cell; everything
    after the `%%` line is interpreted by the magic, not by the Python parser, so
    `ast.parse` cannot audit it and must not be asked to.
    """

    # Skip leading blank lines so a magic preceded by whitespace is still detected.
    first_content_line = next((line for line in source.splitlines() if line.strip()), "")
    return first_content_line.lstrip().startswith("%%")


def extract_code_cells(notebook_path: Path) -> tuple[tuple[int, str], ...]:
    """Return each code cell's zero-based position among code cells and its source.

    Reading through `nbformat.read` (rather than parsing the raw JSON directly)
    normalizes each cell's `source` field to one string regardless of whether it was
    stored on disk as a single string or a list of line fragments.
    """

    notebook = nbformat.read(notebook_path, as_version=nbformat.NO_CONVERT)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    return tuple((index, str(cell.source)) for index, cell in enumerate(code_cells))


def audit_notebook(path: Path) -> tuple[CommentaryViolation, ...]:
    """Audit one curated notebook's Python code cells without executing any cell."""

    # A malformed or unreadable notebook should produce one clear policy diagnostic,
    # matching audit_file's own handling of unparseable Python source.
    try:
        cells = extract_code_cells(path)
    except Exception as error:  # nbformat wraps several underlying JSON/schema errors.
        return (CommentaryViolation(path, 1, f"cannot read notebook: {error}"),)

    violations: list[CommentaryViolation] = []
    # Audit each code cell independently so one cell's violations never mask another's.
    for index, source in cells:
        # Cell magics (%%bash, %%time, ...) are not Python; auditing them as Python
        # would misreport every one as a syntax-error violation.
        if _is_cell_magic(source):
            continue
        cell_path = Path(f"{path.as_posix()}::cell:{index}")
        # require_module_docstring=False: no single code cell is expected to carry
        # the notebook's own narrative the way a standalone module's docstring
        # would -- markdown cells already do that, and are outside this policy.
        violations.extend(audit_source(source, path=cell_path, require_module_docstring=False))
    return tuple(violations)


def discover_notebooks(roots: tuple[Path, ...]) -> tuple[Path, ...]:
    """Return every curated notebook beneath the configured roots, deterministically ordered.

    A directory root's descendants are filtered to exclude `notebooks/local/`
    (disposable sandbox) and `archive/original_2022/` (immutable history), matching
    this policy's repository-wide scope; an explicit file argument is trusted as-is.
    """

    notebooks: set[Path] = set()
    # Resolve each explicit root independently so a missing tree fails visibly below.
    for root in roots:
        # A direct file supports focused tests and one-off local investigation.
        if root.is_file() and root.suffix == ".ipynb":
            notebooks.add(root)
        # A directory contributes descendants outside the excluded sandbox/archive trees.
        elif root.is_dir():
            # Filter descendants individually so a directory root can't smuggle in a
            # sandbox/archive notebook the way an explicit file argument could.
            for candidate in root.rglob("*.ipynb"):
                parts = candidate.parts
                # Exclude notebooks/local/, archive/original_2022/, and checkpoint
                # scratch directories Jupyter creates alongside edited notebooks.
                if "local" in parts or "archive" in parts or ".ipynb_checkpoints" in parts:
                    continue
                notebooks.add(candidate)
    return tuple(sorted(notebooks))


def audit_roots(roots: tuple[Path, ...]) -> tuple[CommentaryViolation, ...]:
    """Audit every discovered notebook and return one stable violation sequence."""

    violations: list[CommentaryViolation] = []
    # Keep notebook audits isolated so every problem is reported in one invocation.
    for path in discover_notebooks(roots):
        violations.extend(audit_notebook(path))
    return tuple(violations)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse optional notebook paths for a focused or default curated-notebook check."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebooks",
        nargs="*",
        type=Path,
        default=list(DEFAULT_NOTEBOOKS),
        help="Notebook files or directories to audit (default: the three curated notebooks)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the notebook commentary audit and return a conventional process exit status."""

    args = parse_args(argv)
    roots = tuple(args.notebooks)
    files = discover_notebooks(roots)
    # Treat an empty discovery set as configuration failure rather than false success.
    if not files:
        print("error: notebook commentary audit found no notebooks", file=sys.stderr)
        return 2

    violations = audit_roots(roots)
    # Emit every violation so one remediation pass can address the complete inventory.
    if violations:
        print(
            f"Notebook commentary audit failed: {len(violations)} violation(s) "
            f"across {len(files)} notebook(s).",
            file=sys.stderr,
        )
        # Preserve deterministic diagnostics for review and automated parsing.
        for violation in violations:
            print(violation.render(), file=sys.stderr)
        return 1

    print(f"Notebook commentary audit passed for {len(files)} notebook(s).")
    return 0


# Keep import behavior side-effect free while providing a normal executable script boundary.
if __name__ == "__main__":
    raise SystemExit(main())
