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

BUFFER_SIZE = 1024 * 1024
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
        return _to_json(self)


@dataclass(frozen=True, slots=True)
class RuntimeSummary:
    """Monotonic elapsed durations for supported pipeline stages."""

    schema_version: int
    duration_unit: str
    stage_durations: dict[str, float]
    total_runtime: float

    def to_json(self) -> str:
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
        return _to_json(self)


class RuntimeStageTimer:
    """Accumulate monotonic time by a fixed set of pipeline stages."""

    def __init__(self, timer: Callable[[], float] = perf_counter) -> None:
        self._timer = timer
        self._started_at = timer()
        self._durations = dict.fromkeys(STAGE_NAMES, 0.0)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if name not in self._durations:
            raise ReproducibilityEvidenceError(f"unsupported runtime stage: {name}")
        started_at = self._timer()
        try:
            yield
        finally:
            elapsed = self._timer() - started_at
            if elapsed < 0:
                raise ReproducibilityEvidenceError("monotonic timer moved backwards")
            self._durations[name] += elapsed

    def summary(self) -> RuntimeSummary:
        total = self._timer() - self._started_at
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
    if path.is_symlink() or not path.is_file():
        raise ReproducibilityEvidenceError(f"evidence path must be a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
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
    for path in paths:
        candidate = path if path.is_absolute() else root / path
        if candidate.is_symlink():
            raise ReproducibilityEvidenceError(f"evidence path must not be a symlink: {candidate}")
        resolved = candidate.resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as error:
            raise ReproducibilityEvidenceError(
                f"evidence path must stay within repository root: {resolved}"
            ) from error
        if resolved in seen:
            raise ReproducibilityEvidenceError(f"duplicate evidence path: {relative.as_posix()}")
        seen.add(resolved)
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
    if candidate.is_symlink():
        raise ReproducibilityEvidenceError("evidence output must not be a symbolic link")
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ReproducibilityEvidenceError(
            "evidence output must stay within repository root"
        ) from error
    if not relative.parts or relative.parts[0] != "artifacts":
        raise ReproducibilityEvidenceError("evidence output must be written under artifacts/")
    if resolved.suffix != ".json" or not resolved.parent.is_dir():
        raise ReproducibilityEvidenceError(
            "evidence output must be a JSON file with an existing parent directory"
        )
    resolved.write_text(document.to_json(), encoding="utf-8")


def _to_json(document: Any) -> str:
    return json.dumps(asdict(document), indent=2, sort_keys=True) + "\n"


def _run_optional(command: tuple[str, ...], cwd: Path) -> str | None:
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
    if sys.platform == "darwin":
        return _run_optional(("sysctl", "-n", "machdep.cpu.brand_string"), Path.cwd())
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    return line.partition(":")[2].strip() or None
        except OSError:
            return None
    return platform.processor() or None


def _capture_memory_total() -> int | None:
    if sys.platform == "darwin":
        value = _run_optional(("sysctl", "-n", "hw.memsize"), Path.cwd())
        try:
            return int(value) if value else None
        except ValueError:
            return None
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            return None
    return None
