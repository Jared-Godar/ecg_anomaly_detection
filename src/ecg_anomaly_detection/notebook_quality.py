"""Static quality checks for local notebooks that never execute notebook cells."""

from __future__ import annotations

import json
import re
import warnings
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import nbformat

NARRATIVE_NOTEBOOK = Path("notebooks/narrative-walkthrough.ipynb")
LOCAL_NOTEBOOK_DIRECTORY = Path("notebooks/local")
LOCAL_PATH_PATTERN = re.compile(
    r"(?:^|[\s\"'(=])(?:/Users/[^\s\"')]+|/home/[^\s\"')]+|[A-Za-z]:\\[^\s\"')]+)"
)


class NotebookQualityError(ValueError):
    """Raised when notebook quality arguments violate repository boundaries."""


@dataclass(frozen=True)
class NotebookIssue:
    """One actionable notebook quality finding."""

    code: str
    severity: Literal["error", "warning"]
    message: str
    cell_index: int | None = None


@dataclass(frozen=True)
class NotebookReport:
    """Static validation and hygiene results for one notebook."""

    path: str
    valid: bool
    changed: bool
    cell_count: int
    code_cell_count: int
    output_count: int
    output_bytes: int
    execution_count_count: int
    trusted_cell_count: int
    untrusted_cell_count: int
    issues: tuple[NotebookIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""
        return asdict(self)


@dataclass(frozen=True)
class NotebookQualitySummary:
    """Aggregate results for a deterministic notebook quality run."""

    notebooks: tuple[NotebookReport, ...]

    @property
    def valid(self) -> bool:
        """Return whether every discovered notebook passed structural validation."""
        return all(report.valid for report in self.notebooks)

    @property
    def changed_count(self) -> int:
        """Return the number of notebooks rewritten by an opt-in operation."""
        return sum(report.changed for report in self.notebooks)

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic machine-readable output."""
        return {
            "schema_version": 1,
            "valid": self.valid,
            "notebook_count": len(self.notebooks),
            "changed_count": self.changed_count,
            "notebooks": [report.to_dict() for report in self.notebooks],
        }


def discover_local_notebooks(
    repository_root: Path, *, include_narrative: bool = False
) -> tuple[Path, ...]:
    """Discover local notebooks and optionally the package-backed narrative notebook."""
    root = repository_root.resolve()
    local_directory = root / LOCAL_NOTEBOOK_DIRECTORY
    notebooks = (
        sorted(
            path
            for path in local_directory.rglob("*.ipynb")
            if ".ipynb_checkpoints" not in path.parts
        )
        if local_directory.is_dir()
        else []
    )
    if include_narrative:
        narrative = root / NARRATIVE_NOTEBOOK
        if narrative.is_file():
            notebooks.append(narrative)
    return tuple(notebooks)


def check_notebooks(
    repository_root: Path,
    notebook_paths: Sequence[Path],
    *,
    format_notebooks: bool = False,
    strip_outputs: bool = False,
) -> NotebookQualitySummary:
    """Validate and optionally normalize notebooks without executing any cell."""
    root = repository_root.resolve()
    reports = tuple(
        _check_notebook(
            root,
            _resolve_notebook_path(root, path),
            format_notebook=format_notebooks,
            strip_outputs=strip_outputs,
        )
        for path in sorted(notebook_paths, key=lambda candidate: str(candidate))
    )
    return NotebookQualitySummary(notebooks=reports)


def _resolve_notebook_path(repository_root: Path, path: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (repository_root / path).resolve()
    try:
        relative = resolved.relative_to(repository_root)
    except ValueError as error:
        raise NotebookQualityError(f"notebook is outside the repository: {path}") from error
    if resolved.suffix != ".ipynb" or relative.parts[:1] != ("notebooks",):
        raise NotebookQualityError(f"notebook must be an .ipynb file under notebooks/: {path}")
    return resolved


def _check_notebook(
    repository_root: Path,
    path: Path,
    *,
    format_notebook: bool,
    strip_outputs: bool,
) -> NotebookReport:
    relative_path = path.relative_to(repository_root).as_posix()
    try:
        with warnings.catch_warnings(record=True) as validation_warnings:
            warnings.simplefilter("always")
            notebook = nbformat.read(path, as_version=nbformat.NO_CONVERT)
            nbformat.validate(notebook)
    except Exception as error:  # nbformat wraps JSON and schema errors from several libraries.
        return NotebookReport(
            path=relative_path,
            valid=False,
            changed=False,
            cell_count=0,
            code_cell_count=0,
            output_count=0,
            output_bytes=0,
            execution_count_count=0,
            trusted_cell_count=0,
            untrusted_cell_count=0,
            issues=(NotebookIssue("invalid-notebook", "error", str(error)),),
        )

    issues: list[NotebookIssue] = []
    warning_counts = Counter(
        (warning.category.__name__, str(warning.message)) for warning in validation_warnings
    )
    for (category, message), count in warning_counts.items():
        if category == "MissingIDFieldWarning":
            issues.append(
                NotebookIssue(
                    "missing-cell-ids",
                    "warning",
                    f"{count} cells are missing stable IDs; --format adds them",
                )
            )
        else:
            issues.append(
                NotebookIssue(
                    "nbformat-warning",
                    "warning",
                    f"{message} ({count} occurrences)",
                )
            )
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    outputs = [output for cell in code_cells for output in cell.get("outputs", [])]
    execution_count_count = sum(cell.get("execution_count") is not None for cell in code_cells)
    trusted_cell_count = sum(cell.get("metadata", {}).get("trusted") is True for cell in code_cells)
    untrusted_cell_count = len(code_cells) - trusted_cell_count
    output_bytes = sum(
        len(json.dumps(output, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        for output in outputs
    )

    if outputs:
        issues.append(
            NotebookIssue("saved-outputs", "warning", f"contains {len(outputs)} saved outputs")
        )
    if execution_count_count:
        issues.append(
            NotebookIssue(
                "execution-counts",
                "warning",
                f"contains {execution_count_count} saved execution counts",
            )
        )
    if untrusted_cell_count:
        issues.append(
            NotebookIssue(
                "untrusted-cells",
                "warning",
                f"contains {untrusted_cell_count} code cells without trusted metadata",
            )
        )

    kernel_name = str(notebook.metadata.get("kernelspec", {}).get("name", ""))
    if kernel_name and kernel_name not in {"python3", "ecg-anomaly-detection"}:
        issues.append(
            NotebookIssue(
                "stale-kernel-metadata",
                "warning",
                f"kernel name {kernel_name!r} is not a recognized repository kernel",
            )
        )

    for index, cell in enumerate(notebook.cells):
        source = str(cell.get("source", ""))
        if LOCAL_PATH_PATTERN.search(source):
            issues.append(
                NotebookIssue(
                    "machine-local-path",
                    "warning",
                    "cell contains an absolute machine-local path",
                    cell_index=index,
                )
            )

    original = path.read_text(encoding="utf-8")
    if strip_outputs:
        for cell in code_cells:
            cell["outputs"] = []
            cell["execution_count"] = None
    serialized = nbformat.writes(notebook, version=nbformat.NO_CONVERT) + "\n"
    changed = (format_notebook or strip_outputs) and serialized != original
    if changed:
        path.write_text(serialized, encoding="utf-8")

    return NotebookReport(
        path=relative_path,
        valid=True,
        changed=changed,
        cell_count=len(notebook.cells),
        code_cell_count=len(code_cells),
        output_count=len(outputs),
        output_bytes=output_bytes,
        execution_count_count=execution_count_count,
        trusted_cell_count=trusted_cell_count,
        untrusted_cell_count=untrusted_cell_count,
        issues=tuple(issues),
    )
