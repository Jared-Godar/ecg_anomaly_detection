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
        # Check all three companion directories for this run, even if some are
        # missing (a run may not have reached every stage before being interrupted).
        for directory in (
            self.artifacts_directory,
            self.interim_directory,
            self.processed_directory,
        ):
            # A symlink at this exact path would mean shutil.rmtree (called by
            # purge_run) could delete through the link to an unrelated location; refuse
            # to consider it removable rather than silently following it.
            if directory.is_symlink():
                raise LocalArtifactLifecycleError(
                    f"run directory must not be a symbolic link: {directory}"
                )
            # Only an existing directory is actually removable; a run that never
            # reached this stage simply has no directory here to include.
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
    # Reject a symlinked runs parent rather than following it into an unrelated
    # location and listing whatever run-like directories happen to live there.
    if artifact_runs.is_symlink():
        raise LocalArtifactLifecycleError(
            f"run parent must not be a symbolic link: {artifact_runs}"
        )
    # No runs directory yet is a valid, common state (e.g. before the first
    # run_pipeline() invocation), not an error -- return an empty result.
    if not artifact_runs.is_dir():
        return ()
    summaries = []
    # Iterate in sorted (name) order for deterministic processing before the final
    # reverse-chronological sort below.
    for run_directory in sorted(artifact_runs.iterdir()):
        # Skip anything that isn't a directory, and anything whose name isn't a
        # canonical run ID -- e.g. stray files or leftover non-UUID directories that
        # this module never created and shouldn't treat as a run.
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
    # A malformed run ID can't correspond to any real run_pipeline() output, since
    # run_pipeline itself only ever creates canonical-UUID-named directories.
    if not _is_canonical_run_id(run_id):
        raise LocalArtifactLifecycleError(f"run ID must be a canonical lowercase UUID: {run_id}")
    location = _run_location(root, run_id)
    directories = location.removable_directories()
    # A run ID with no matching directories anywhere means there's nothing to purge --
    # likely a typo'd or already-purged run ID, worth reporting rather than silently
    # succeeding at removing zero directories.
    if not directories:
        raise LocalArtifactLifecycleError(f"no local run directories found for run ID: {run_id}")
    freed_bytes = sum(_directory_size(directory) for directory in directories)
    # dry_run reports what would be removed (directories and freed_bytes are always
    # computed above) without actually deleting anything.
    if not dry_run:
        # Remove every companion directory found for this run; a run is either fully
        # present or fully removed, matching this module's documented all-or-nothing
        # contract.
        for directory in directories:
            shutil.rmtree(directory)
    return PurgeResult(
        run_id=run_id,
        dry_run=dry_run,
        removed_directories=directories,
        freed_bytes=freed_bytes,
    )


def _run_location(repository_root: Path, run_id: str) -> RunLocation:
    """Build the three companion paths run_pipeline() would create for one run ID.

    Purely a path-construction helper; it doesn't check whether any of these
    directories actually exist (see RunLocation.existing_directories/removable_directories
    for that).

    Args:
        repository_root: Repository root used to enforce path and trust boundaries.
        run_id: The canonical run ID to build paths for.

    Returns:
        The run's artifacts/interim/processed directory paths.
    """

    return RunLocation(
        artifacts_directory=repository_root / "artifacts" / "runs" / run_id,
        interim_directory=repository_root / "data" / "interim" / "runs" / run_id,
        processed_directory=repository_root / "data" / "processed" / "runs" / run_id,
    )


def _resolve_repository_root(repository_root: Path) -> Path:
    """Resolve and validate the repository root shared by list_runs and purge_run.

    Args:
        repository_root: The candidate repository root.

    Returns:
        The resolved, validated repository root.
    """

    root = repository_root.resolve()
    # A pyproject.toml at the root is the cheapest available signal that this is
    # really the repository root, before every run-directory path below trusts it as
    # the containment root.
    if not (root / "pyproject.toml").is_file():
        raise LocalArtifactLifecycleError(
            f"repository root does not contain pyproject.toml: {root}"
        )
    return root


def _is_canonical_run_id(candidate: str) -> bool:
    """Return whether a string is a canonical (lowercase, hyphenated) UUID.

    Matches the exact formatting pipeline.py's _create_run_id enforces when a run is
    created, so this function only ever recognizes directories that a real
    run_pipeline() invocation could have produced.

    Args:
        candidate: The string to check (typically a directory name).

    Returns:
        True if candidate is a canonical UUID string.
    """

    # uuid.UUID accepts multiple textual representations of the same UUID (different
    # case, with/without hyphens); a non-UUID string raises one of these three
    # exception types depending on exactly how it's malformed.
    try:
        parsed = uuid.UUID(candidate)
    except (AttributeError, TypeError, ValueError):
        return False
    # Comparing str(parsed) back to candidate specifically rejects non-canonical but
    # otherwise valid UUID spellings (different case, missing hyphens), matching
    # pipeline.py's own canonical-formatting requirement.
    return str(parsed) == candidate


def _directory_size(directory: Path) -> int:
    """Sum the size in bytes of every regular file under a directory, recursively.

    Args:
        directory: The directory to measure.

    Returns:
        Total size in bytes of all regular files found.
    """

    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
