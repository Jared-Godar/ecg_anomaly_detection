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

# The one package-backed notebook outside notebooks/local/ this module will check when
# explicitly asked (include_narrative=True); it's excluded by default because it's a
# curated, reviewed artifact rather than disposable local experimentation.
NARRATIVE_NOTEBOOK = Path("notebooks/narrative-walkthrough.ipynb")
# The gitignored sandbox directory this module discovers notebooks from by default.
LOCAL_NOTEBOOK_DIRECTORY = Path("notebooks/local")
# Matches an absolute filesystem path (macOS/Linux /Users or /home, or a Windows drive
# letter) appearing inside a notebook cell's source, so a notebook can't be committed
# with a machine-specific path baked into it -- the leading lookahead group requires
# the path to start at a word boundary (start-of-string, whitespace, quote, paren, or
# `=`) so it doesn't match a path fragment embedded inside a longer, unrelated token.
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
    # The narrative notebook is opt-in since it's curated package content, not part of
    # the disposable local/ sandbox this function discovers by default.
    if include_narrative:
        narrative = root / NARRATIVE_NOTEBOOK
        # The narrative notebook may not exist in every checkout (e.g. removed or
        # renamed); treat its absence as "nothing extra to check" rather than an error.
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
    """Resolve a notebook path and enforce it's an .ipynb file under notebooks/.

    Args:
        repository_root: Repository root used to enforce path and trust boundaries.
        path: The candidate notebook path, absolute or relative to repository_root.

    Returns:
        The resolved, validated absolute path.
    """

    resolved = path.resolve() if path.is_absolute() else (repository_root / path).resolve()
    # relative_to raises ValueError when resolved escapes repository_root (e.g. via
    # `..` segments); translate that into this module's own exception type.
    try:
        relative = resolved.relative_to(repository_root)
    except ValueError as error:
        raise NotebookQualityError(f"notebook is outside the repository: {path}") from error
    # This module is scoped exclusively to notebooks/ (see NARRATIVE_NOTEBOOK and
    # LOCAL_NOTEBOOK_DIRECTORY above); any other .ipynb elsewhere isn't a supported input.
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
    """Load, validate, and optionally normalize one notebook without executing any cell.

    Structural validation (parse + nbformat schema check) always runs; format_notebook
    and strip_outputs are separate opt-in normalization passes applied only after
    validation succeeds, and only actually rewrite the file if the serialized content
    changed.

    Args:
        repository_root: Repository root, used to compute the report's relative path.
        path: The already-resolved, validated notebook path to check.
        format_notebook: Whether to let nbformat add stable cell IDs where missing.
        strip_outputs: Whether to clear every code cell's saved outputs and execution count.

    Returns:
        The notebook's structural validity, hygiene issues, and change status.
    """

    relative_path = path.relative_to(repository_root).as_posix()
    # nbformat wraps a wide variety of underlying JSON/schema-library errors under one
    # umbrella; catching bare Exception here (rather than a specific type) is
    # deliberate, since a malformed notebook should always produce a structured,
    # reportable NotebookIssue rather than propagate an uncaught exception and abort
    # the whole check run over every other notebook.
    try:
        # record=True captures nbformat's own validation warnings (e.g. missing cell
        # IDs) instead of letting them print to stderr; simplefilter("always") ensures
        # every occurrence is captured, not just the first (Python's default warning
        # filter deduplicates by default).
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
    # Group captured warnings by (category, message) so repeated identical warnings
    # (e.g. every cell missing an ID) are reported once with an occurrence count,
    # rather than as one NotebookIssue per warning instance.
    for (category, message), count in warning_counts.items():
        # Missing cell IDs get a specific, actionable message (mentioning --format);
        # every other nbformat warning falls through to a generic report below.
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

    # Saved outputs bloat diffs and can leak local/sensitive data into version control;
    # flag their presence (this module never executes cells, so it can't verify outputs
    # are still accurate anyway).
    if outputs:
        issues.append(
            NotebookIssue("saved-outputs", "warning", f"contains {len(outputs)} saved outputs")
        )
    # Saved execution counts imply a specific run order that isn't guaranteed to still
    # be accurate or reproducible once committed.
    if execution_count_count:
        issues.append(
            NotebookIssue(
                "execution-counts",
                "warning",
                f"contains {execution_count_count} saved execution counts",
            )
        )
    # An untrusted code cell's outputs won't render in some notebook viewers/servers
    # without the user manually re-trusting the file.
    if untrusted_cell_count:
        issues.append(
            NotebookIssue(
                "untrusted-cells",
                "warning",
                f"contains {untrusted_cell_count} code cells without trusted metadata",
            )
        )

    kernel_name = str(notebook.metadata.get("kernelspec", {}).get("name", ""))
    # An empty kernel_name means no kernelspec was recorded at all, which is a
    # separate (unflagged) state from one naming an unrecognized kernel -- only the
    # latter is actionable here, since it suggests the notebook was authored under a
    # different environment than this repository's own.
    if kernel_name and kernel_name not in {"python3", "ecg-anomaly-detection"}:
        issues.append(
            NotebookIssue(
                "stale-kernel-metadata",
                "warning",
                f"kernel name {kernel_name!r} is not a recognized repository kernel",
            )
        )

    # Scan every cell's source (not just code cells) for a machine-local path, since
    # markdown cells can just as easily embed one in prose or a code-fenced example.
    for index, cell in enumerate(notebook.cells):
        source = str(cell.get("source", ""))
        # See LOCAL_PATH_PATTERN's own comment for exactly what this matches.
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
    # strip_outputs is a separate opt-in from format_notebook (see check_notebooks'
    # own parameters); only clear outputs/execution counts when explicitly requested.
    if strip_outputs:
        # Clear every code cell's outputs and execution count in place, ahead of the
        # re-serialization below.
        for cell in code_cells:
            cell["outputs"] = []
            cell["execution_count"] = None
    serialized = nbformat.writes(notebook, version=nbformat.NO_CONVERT) + "\n"
    changed = (format_notebook or strip_outputs) and serialized != original
    # Only rewrite the file if a normalization pass was actually requested AND it
    # produced different content -- an already-clean notebook is left untouched even
    # with --format/--strip-outputs passed, so re-running this check doesn't spuriously
    # dirty the working tree.
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
