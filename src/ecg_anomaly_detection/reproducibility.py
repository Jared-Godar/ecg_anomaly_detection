"""Operational reproducibility evidence for supported pipeline runs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

# Chunk size for streaming file reads during digest computation. 1 MiB balances syscall
# overhead against peak memory for evidence files that can be large.
BUFFER_SIZE = 1024 * 1024
# The fixed, closed set of pipeline stages RuntimeStageTimer accepts. Kept as an
# explicit tuple (rather than discovered dynamically) so a typo'd stage name in calling
# code fails immediately in RuntimeStageTimer.stage rather than silently creating an
# unexpected new key in the runtime summary.
STAGE_NAMES = (
    "acquisition",
    "validation",
    "annotation_mapping",
    "window_extraction",
    "split",
    "split_diagnostics",
    "training",
    "validation_evaluation",
)


class ReproducibilityEvidenceError(ValueError):
    """Raised when required reproducibility evidence violates its contract."""


@dataclass(frozen=True, slots=True)
class ArtifactEvidence:
    """Repository-relative identity for one input, output, or evidence file."""

    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class GitMetadata:
    """Git metadata, nullable when it cannot be captured on the current host."""

    commit: str | None
    branch: str | None
    dirty: bool | None


@dataclass(frozen=True, slots=True)
class EnvironmentSummary:
    """Host, interpreter, dependency, and source identity for a run."""

    schema_version: int
    os_name: str
    os_release: str
    architecture: str
    python_version: str
    python_implementation: str
    uv_version: str | None
    dependency_lock: ArtifactEvidence
    git: GitMetadata

    def to_json(self) -> str:
        """Serialize this structured record as deterministic JSON.

        Returns:
            The summary as a JSON string with sorted, deterministic key ordering.
        """

        return _to_json(self)


@dataclass(frozen=True, slots=True)
class RuntimeSummary:
    """Monotonic elapsed durations for supported pipeline stages."""

    schema_version: int
    duration_unit: str
    stage_durations: dict[str, float]
    total_runtime: float

    def to_json(self) -> str:
        """Serialize this structured record as deterministic JSON.

        Returns:
            The summary as a JSON string with sorted, deterministic key ordering.
        """

        return _to_json(self)


@dataclass(frozen=True, slots=True)
class ResourceSummary:
    """Best-effort host capacity and repository-filesystem utilization."""

    schema_version: int
    cpu_model: str | None
    logical_core_count: int | None
    memory_total_bytes: int | None
    disk_total_bytes: int | None
    disk_used_bytes: int | None
    disk_free_bytes: int | None

    def to_json(self) -> str:
        """Serialize this structured record as deterministic JSON.

        Returns:
            The summary as a JSON string with sorted, deterministic key ordering.
        """

        return _to_json(self)


@dataclass(frozen=True, slots=True)
class SplitIdentity:
    """Identity copied from the validated split manifest."""

    name: str
    version: str
    strategy: str
    seed: int
    manifest: ArtifactEvidence


@dataclass(frozen=True, slots=True)
class EvidenceManifest:
    """Digest index connecting run configuration, split, artifacts, and evidence."""

    schema_version: int
    split: SplitIdentity
    configuration_files: tuple[ArtifactEvidence, ...]
    evidence_files: tuple[ArtifactEvidence, ...]
    artifact_files: tuple[ArtifactEvidence, ...]

    def to_json(self) -> str:
        """Serialize this structured record as deterministic JSON.

        Returns:
            The manifest as a JSON string with sorted, deterministic key ordering.
        """

        return _to_json(self)


class RuntimeStageTimer:
    """Accumulate monotonic time by a fixed set of pipeline stages."""

    def __init__(self, timer: Callable[[], float] = perf_counter) -> None:
        """Start the timer and zero every stage's accumulated duration.

        Args:
            timer: A monotonic clock function; overridable in tests to control elapsed
                time deterministically instead of depending on real wall-clock time.
        """

        self._timer = timer
        self._started_at = timer()
        self._durations = dict.fromkeys(STAGE_NAMES, 0.0)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Measure and accumulate elapsed time for one named pipeline stage.

        Reentrant-safe across separate calls for the same stage name: durations
        accumulate rather than overwrite, so a stage entered multiple times (e.g. once
        per record in a loop) reports its total time across all invocations.

        Args:
            name: Which stage this timing block belongs to; must be one of STAGE_NAMES.

        Returns:
            A context manager; the enclosed block's wall-clock time is added to this
            stage's running total when the block exits.
        """

        # An unrecognized stage name would otherwise silently create a new dict key
        # rather than accumulating into one of the fixed, known stages.
        if name not in self._durations:
            raise ReproducibilityEvidenceError(f"unsupported runtime stage: {name}")
        started_at = self._timer()
        # `finally` ensures the elapsed time is recorded even if the enclosed block
        # raises, so a failed stage's partial duration still contributes to the total.
        try:
            yield
        finally:
            elapsed = self._timer() - started_at
            # A monotonic clock should never move backwards; if it does, the timer
            # implementation itself (or the injected fake in a test) is broken, and
            # accumulating a negative duration would produce a nonsensical summary.
            if elapsed < 0:
                raise ReproducibilityEvidenceError("monotonic timer moved backwards")
            self._durations[name] += elapsed

    def summary(self) -> RuntimeSummary:
        """Finalize accumulated stage durations into a serializable summary.

        Returns:
            Per-stage durations and the total elapsed time since construction, in seconds.
        """

        total = self._timer() - self._started_at
        # Same monotonic-clock sanity check as in stage() above, applied to the
        # timer's total elapsed time since construction.
        if total < 0:
            raise ReproducibilityEvidenceError("monotonic timer moved backwards")
        return RuntimeSummary(
            schema_version=1,
            duration_unit="seconds",
            stage_durations={name: round(value, 9) for name, value in self._durations.items()},
            total_runtime=round(total, 9),
        )


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a regular file."""
    # Reject a symlink as well as a missing/non-regular file: resolving through a link
    # could hash a file other than the one this evidence claims to describe.
    if path.is_symlink() or not path.is_file():
        raise ReproducibilityEvidenceError(f"evidence path must be a regular file: {path}")
    digest = hashlib.sha256()
    # Read in fixed-size chunks rather than the whole file at once, since evidence
    # files (e.g. uv.lock, large artifacts) can be sizable.
    with path.open("rb") as source:
        # The walrus operator lets both the read and the loop's termination condition
        # (an empty final chunk) live in one line without a separate `break`.
        while chunk := source.read(BUFFER_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def collect_artifact_evidence(
    repository_root: Path, paths: Sequence[Path]
) -> tuple[ArtifactEvidence, ...]:
    """Collect stable, repository-relative file evidence in caller-supplied order."""
    root = repository_root.resolve()
    evidence: list[ArtifactEvidence] = []
    seen: set[Path] = set()
    # Preserve the caller's own ordering (rather than sorting) since callers pass
    # semantically ordered lists (e.g. configs in the order they were loaded).
    for path in paths:
        candidate = path if path.is_absolute() else root / path
        # Reject a symlink before resolving it, so a link that points outside the
        # repository can't be validated against a resolved target it doesn't actually name.
        if candidate.is_symlink():
            raise ReproducibilityEvidenceError(f"evidence path must not be a symlink: {candidate}")
        resolved = candidate.resolve()
        # relative_to raises ValueError when resolved escapes root (e.g. via `..`
        # segments); translate that into this module's own exception type.
        try:
            relative = resolved.relative_to(root)
        except ValueError as error:
            raise ReproducibilityEvidenceError(
                f"evidence path must stay within repository root: {resolved}"
            ) from error
        # A duplicated path would otherwise contribute the same file's evidence twice
        # to the resulting tuple.
        if resolved in seen:
            raise ReproducibilityEvidenceError(f"duplicate evidence path: {relative.as_posix()}")
        seen.add(resolved)
        # A directory or special file can't be hashed as file content below.
        if not resolved.is_file():
            raise ReproducibilityEvidenceError(f"evidence path must be a regular file: {resolved}")
        evidence.append(
            ArtifactEvidence(
                path=relative.as_posix(),
                size_bytes=resolved.stat().st_size,
                sha256=sha256_file(resolved),
            )
        )
    return tuple(evidence)


def capture_git_metadata(repository_root: Path) -> GitMetadata:
    """Capture Git metadata without failing evidence generation when unavailable."""
    commit = _run_optional(("git", "rev-parse", "HEAD"), repository_root)
    branch = _run_optional(("git", "branch", "--show-current"), repository_root)
    status = _run_optional_allow_empty(
        ("git", "status", "--porcelain", "--untracked-files=normal"), repository_root
    )
    valid_commit = commit if commit and len(commit) == 40 else None
    return GitMetadata(
        commit=valid_commit,
        branch=branch or None,
        dirty=None if status is None else bool(status),
    )


def capture_environment_summary(repository_root: Path) -> EnvironmentSummary:
    """Capture required runtime identity and best-effort tool/source metadata."""
    root = repository_root.resolve()
    lock = collect_artifact_evidence(root, (root / "uv.lock",))[0]
    uv_output = _run_optional(("uv", "--version"), root)
    return EnvironmentSummary(
        schema_version=1,
        os_name=platform.system() or sys.platform,
        os_release=platform.release(),
        architecture=platform.machine(),
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        uv_version=uv_output,
        dependency_lock=lock,
        git=capture_git_metadata(root),
    )


def capture_resource_summary(repository_root: Path) -> ResourceSummary:
    """Capture optional host resources, returning null for unavailable metrics."""
    disk_total: int | None = None
    disk_used: int | None = None
    disk_free: int | None = None
    # Disk usage can fail (e.g. an unreadable filesystem); treat it as best-effort like
    # every other field in ResourceSummary rather than failing the whole capture.
    try:
        usage = shutil.disk_usage(repository_root)
        disk_total, disk_used, disk_free = usage.total, usage.used, usage.free
    except OSError:
        pass
    return ResourceSummary(
        schema_version=1,
        cpu_model=_capture_cpu_model(),
        logical_core_count=os.cpu_count(),
        memory_total_bytes=_capture_memory_total(),
        disk_total_bytes=disk_total,
        disk_used_bytes=disk_used,
        disk_free_bytes=disk_free,
    )


def create_evidence_manifest(
    repository_root: Path,
    split_manifest_path: Path,
    config_paths: Sequence[Path],
    evidence_paths: Sequence[Path],
    artifact_paths: Sequence[Path],
    *,
    split_name: str,
    split_version: str,
    split_strategy: str,
    split_seed: int,
) -> EvidenceManifest:
    """Create the digest index after all referenced files have been written."""
    root = repository_root.resolve()
    split_file = collect_artifact_evidence(root, (split_manifest_path,))[0]
    return EvidenceManifest(
        schema_version=1,
        split=SplitIdentity(
            name=split_name,
            version=split_version,
            strategy=split_strategy,
            seed=split_seed,
            manifest=split_file,
        ),
        configuration_files=collect_artifact_evidence(root, config_paths),
        evidence_files=collect_artifact_evidence(root, evidence_paths),
        artifact_files=collect_artifact_evidence(root, artifact_paths),
    )


def write_evidence(
    document: EnvironmentSummary | RuntimeSummary | ResourceSummary | EvidenceManifest,
    repository_root: Path,
    output_path: Path,
) -> None:
    """Write one JSON evidence document under the ignored artifacts directory."""
    root = repository_root.resolve()
    candidate = output_path if output_path.is_absolute() else root / output_path
    # Reject a symlink before resolving it: resolving would silently follow the link and
    # write to wherever it points, defeating the repository-root containment check below.
    if candidate.is_symlink():
        raise ReproducibilityEvidenceError("evidence output must not be a symbolic link")
    resolved = candidate.resolve()
    # relative_to raises ValueError when resolved escapes root.
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ReproducibilityEvidenceError(
            "evidence output must stay within repository root"
        ) from error
    # Reproducibility evidence is pipeline-generated, matching this repository's
    # directory contract for artifacts.
    if not relative.parts or relative.parts[0] != "artifacts":
        raise ReproducibilityEvidenceError("evidence output must be written under artifacts/")
    # Combine the extension and parent-directory checks since both must hold before
    # the write below can succeed, and either failing means the same "not ready to
    # write" state.
    if resolved.suffix != ".json" or not resolved.parent.is_dir():
        raise ReproducibilityEvidenceError(
            "evidence output must be a JSON file with an existing parent directory"
        )
    resolved.write_text(document.to_json(), encoding="utf-8")


def _to_json(document: Any) -> str:
    """Serialize any of this module's frozen dataclasses to deterministic JSON.

    Shared by every to_json() method above so the same indent/sort_keys formatting is
    guaranteed identical across all four document types.

    Args:
        document: One of EnvironmentSummary, RuntimeSummary, ResourceSummary, or
            EvidenceManifest.

    Returns:
        The document as a JSON string with sorted, deterministic key ordering.
    """

    return json.dumps(asdict(document), indent=2, sort_keys=True) + "\n"


def _run_optional(command: tuple[str, ...], cwd: Path) -> str | None:
    """Run a command and return its stripped stdout, or None if it failed or was empty.

    Used for evidence that's genuinely optional (git commit/branch, uv version): a
    missing tool, non-Git checkout, or any other failure degrades to None rather than
    aborting evidence capture entirely, since this module's whole purpose is
    best-effort reproducibility evidence, not a hard requirement on host tooling.

    Args:
        command: The argv to execute.
        cwd: Working directory for the subprocess.

    Returns:
        Stripped stdout, or None if the command failed or produced no output.
    """

    # Collapse "tool not installed" (OSError) and "command exited non-zero"
    # (CalledProcessError, since check=True) into the same None result.
    try:
        # every call site below passes a fixed literal command tuple, not
        # runtime/user-constructed input.
        result = subprocess.run(  # noqa: S603
            command, cwd=cwd, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _run_optional_allow_empty(command: tuple[str, ...], cwd: Path) -> str | None:
    """Run a command and return its stripped stdout, treating empty output as valid.

    Same failure handling as _run_optional, but used for `git status --porcelain`,
    where empty output is a meaningful, valid result (a clean working tree) rather
    than "no data" -- _run_optional's `or None` would otherwise indistinguishably
    collapse "clean" and "command failed" into the same None value.

    Args:
        command: The argv to execute.
        cwd: Working directory for the subprocess.

    Returns:
        Stripped stdout (possibly empty), or None if the command itself failed.
    """

    # Same OSError/CalledProcessError collapsing as _run_optional above.
    try:
        # every call site below passes a fixed literal command tuple, not
        # runtime/user-constructed input.
        result = subprocess.run(  # noqa: S603
            command, cwd=cwd, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _capture_cpu_model() -> str | None:
    """Best-effort CPU model string, using a platform-specific lookup where available.

    Falls back through three tiers: a macOS sysctl call, a Linux /proc/cpuinfo read,
    then Python's own platform.processor() -- each tier is tried only if the platform
    matches, and any failure at any tier degrades to None rather than raising, matching
    every other field in ResourceSummary.

    Returns:
        The CPU model string, or None if it couldn't be determined.
    """

    # macOS doesn't expose a CPU model string via /proc; sysctl is the standard tool.
    if sys.platform == "darwin":
        return _run_optional(("sysctl", "-n", "machdep.cpu.brand_string"), Path.cwd())
    # Linux exposes CPU info via the /proc/cpuinfo pseudo-file instead of a tool call.
    if sys.platform.startswith("linux"):
        # /proc/cpuinfo may not exist or be readable in some sandboxed/containerized
        # environments; degrade to None rather than raising.
        try:
            # Scan every line for the first "model name" field; /proc/cpuinfo repeats
            # this per logical core, so the first occurrence is sufficient.
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                # Case-insensitive since the exact casing isn't guaranteed by /proc.
                if line.lower().startswith("model name"):
                    return line.partition(":")[2].strip() or None
        except OSError:
            return None
    return platform.processor() or None


def _capture_memory_total() -> int | None:
    """Best-effort total physical memory in bytes, using a platform-specific lookup.

    Same platform-tiered fallback strategy as _capture_cpu_model: macOS via sysctl,
    Linux via /proc/meminfo, otherwise None.

    Returns:
        Total memory in bytes, or None if it couldn't be determined.
    """

    # macOS reports memory size via sysctl's hw.memsize key, in bytes already.
    if sys.platform == "darwin":
        value = _run_optional(("sysctl", "-n", "hw.memsize"), Path.cwd())
        # sysctl output should be a plain integer; a malformed value degrades to None.
        try:
            return int(value) if value else None
        except ValueError:
            return None
    # Linux exposes memory info via /proc/meminfo, in kibibytes; convert to bytes below.
    if sys.platform.startswith("linux"):
        # /proc/meminfo may not exist, or its MemTotal line may be malformed, in some
        # sandboxed/containerized environments; degrade to None rather than raising.
        try:
            # Scan for the MemTotal line; /proc/meminfo lists many other fields this
            # function doesn't need.
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                # MemTotal is reported in kibibytes; convert to bytes on return above.
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            return None
    return None
