"""Lifecycle helpers for locally generated, run-scoped pipeline output.

These helpers list and remove entire `run_pipeline()` output directories by run ID.
They never touch `data/raw/` or `artifacts/datasets/`, which hold the shared,
create-only dataset acquisition baseline rather than run-scoped output, and they
never partially modify a run directory's contents — a run is either present in
full or removed in full.
"""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path


class LocalArtifactLifecycleError(ValueError):
    """Raised when a local run listing or purge request violates its contract."""


@dataclass(frozen=True, slots=True)
class RunLocation:
    """The three companion directories one pipeline run may have written."""

    artifacts_directory: Path
    interim_directory: Path
    processed_directory: Path

    def existing_directories(self) -> tuple[Path, ...]:
        """Directories that exist, following symlinks; used for read-only listing."""
        return tuple(
            directory
            for directory in (
                self.artifacts_directory,
                self.interim_directory,
                self.processed_directory,
            )
            if directory.is_dir()
        )

    def removable_directories(self) -> tuple[Path, ...]:
        """Directories safe to delete: present and not a symlink at this exact path."""
        directories = []
        # Iterate over `(self.artifacts_directory, self.interim_directory,
        # self.processed_directory)` one item at a time so ordering, validation, and failure
        # attribution remain explicit.
        for directory in (
            self.artifacts_directory,
            self.interim_directory,
            self.processed_directory,
        ):
            # Evaluate `directory.is_symlink()` explicitly so invalid or alternate states follow the
            # documented contract.
            if directory.is_symlink():
                raise LocalArtifactLifecycleError(
                    f"run directory must not be a symbolic link: {directory}"
                )
            # Evaluate `directory.is_dir()` explicitly so invalid or alternate states follow the
            # documented contract.
            if directory.is_dir():
                directories.append(directory)
        return tuple(directories)


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Local disk state for one pipeline run, for listing and purge decisions."""

    run_id: str
    has_run_manifest: bool
    total_size_bytes: int
    modified_at_epoch: float
    directories: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class PurgeResult:
    """What a purge removed or, for a dry run, would remove."""

    run_id: str
    dry_run: bool
    removed_directories: tuple[Path, ...]
    freed_bytes: int


def list_runs(repository_root: Path) -> tuple[RunSummary, ...]:
    """Return a summary for every local run, newest first by modification time."""
    root = _resolve_repository_root(repository_root)
    artifact_runs = root / "artifacts" / "runs"
    # Evaluate `artifact_runs.is_symlink()` explicitly so invalid or alternate states follow the
    # documented contract.
    if artifact_runs.is_symlink():
        raise LocalArtifactLifecycleError(
            f"run parent must not be a symbolic link: {artifact_runs}"
        )
    # Evaluate `not artifact_runs.is_dir()` explicitly so invalid or alternate states follow the
    # documented contract.
    if not artifact_runs.is_dir():
        return ()
    summaries = []
    # Iterate over `sorted(artifact_runs.iterdir())` one item at a time so ordering, validation, and
    # failure attribution remain explicit.
    for run_directory in sorted(artifact_runs.iterdir()):
        # Evaluate `not run_directory.is_dir() or not _is_canonical_run_id(run_directory.name)`
        # explicitly so invalid or alternate states follow the documented contract.
        if not run_directory.is_dir() or not _is_canonical_run_id(run_directory.name):
            continue
        run_id = run_directory.name
        location = _run_location(root, run_id)
        directories = location.existing_directories()
        summaries.append(
            RunSummary(
                run_id=run_id,
                has_run_manifest=(run_directory / "run-manifest.json").is_file(),
                total_size_bytes=sum(_directory_size(directory) for directory in directories),
                modified_at_epoch=run_directory.stat().st_mtime,
                directories=directories,
            )
        )
    return tuple(sorted(summaries, key=lambda summary: summary.modified_at_epoch, reverse=True))


def purge_run(
    repository_root: Path,
    run_id: str,
    *,
    dry_run: bool = False,
) -> PurgeResult:
    """Remove (or preview removing) one run's artifact, interim, and processed output.

    Only the three directories a matching `run_pipeline()` call would have created
    for this exact run ID are ever touched. Raw source data and the dataset
    acquisition baseline are never in scope, regardless of run ID.
    """
    root = _resolve_repository_root(repository_root)
    # Evaluate `not _is_canonical_run_id(run_id)` explicitly so invalid or alternate states follow
    # the documented contract.
    if not _is_canonical_run_id(run_id):
        raise LocalArtifactLifecycleError(f"run ID must be a canonical lowercase UUID: {run_id}")
    location = _run_location(root, run_id)
    directories = location.removable_directories()
    # Evaluate `not directories` explicitly so invalid or alternate states follow the documented
    # contract.
    if not directories:
        raise LocalArtifactLifecycleError(f"no local run directories found for run ID: {run_id}")
    freed_bytes = sum(_directory_size(directory) for directory in directories)
    # Evaluate `not dry_run` explicitly so invalid or alternate states follow the documented
    # contract.
    if not dry_run:
        # Iterate over `directories` one item at a time so ordering, validation, and failure
        # attribution remain explicit.
        for directory in directories:
            shutil.rmtree(directory)
    return PurgeResult(
        run_id=run_id,
        dry_run=dry_run,
        removed_directories=directories,
        freed_bytes=freed_bytes,
    )


def _run_location(repository_root: Path, run_id: str) -> RunLocation:
    """Compute and return run location for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        repository_root: Repository root used to enforce path and trust boundaries.
        run_id: The run id value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    return RunLocation(
        artifacts_directory=repository_root / "artifacts" / "runs" / run_id,
        interim_directory=repository_root / "data" / "interim" / "runs" / run_id,
        processed_directory=repository_root / "data" / "processed" / "runs" / run_id,
    )


def _resolve_repository_root(repository_root: Path) -> Path:
    """Resolve repository root according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        repository_root: Repository root used to enforce path and trust boundaries.

    Returns:
        The value produced by the documented operation.
    """

    root = repository_root.resolve()
    # Evaluate `not (root / 'pyproject.toml').is_file()` explicitly so invalid or alternate states
    # follow the documented contract.
    if not (root / "pyproject.toml").is_file():
        raise LocalArtifactLifecycleError(
            f"repository root does not contain pyproject.toml: {root}"
        )
    return root


def _is_canonical_run_id(candidate: str) -> bool:
    """Return whether is canonical run id under the documented validation contract.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        candidate: The candidate value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    # Attempt this boundary operation here so (AttributeError, TypeError, ValueError) can be
    # translated or cleaned up under the repository contract.
    try:
        parsed = uuid.UUID(candidate)
    except (AttributeError, TypeError, ValueError):
        return False
    return str(parsed) == candidate


def _directory_size(directory: Path) -> int:
    """Compute and return directory size for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        directory: The directory value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
