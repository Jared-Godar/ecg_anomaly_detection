"""Create and restore a bounded Colab handoff for the public notebook workflow.

Google Colab may attach each opened notebook to a different disposable virtual
machine.  The supported three-notebook walkthrough therefore cannot treat an
in-memory variable or ``/content`` as durable cross-notebook state.  This module
packages only the completed Step 0 evidence and train/validation waveform shards
needed by notebooks 01 and 02, records their digests and exact repository commit,
and restores them into a fresh checkout after verifying the complete archive.

Raw acquisition files and protected-test shards are deliberately excluded.  The
handoff is a private convenience artifact in the reviewer's own Google Drive; it
is not a repository artifact, benchmark artifact, or distribution mechanism.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, TypedDict

# This schema is intentionally independent of run-manifest and dataset-index
# schemas: it governs only the temporary cross-runtime notebook transport.
HANDOFF_SCHEMA_VERSION = 1
# Keeping the manifest at a fixed archive member makes it readable before a fresh
# Colab runtime has cloned or installed the repository package.
HANDOFF_MANIFEST_MEMBER = "notebook-handoff-manifest.json"
# A small pointer avoids requiring downstream notebooks to guess which UUID-scoped
# handoff is newest in a reviewer's private Drive directory.
HANDOFF_POINTER_NAME = "latest.json"
# Only development partitions used by the public validation notebook may cross the
# handoff boundary.  The protected test partition stays indexed but never copied.
INCLUDED_PARTITIONS = ("train", "validation")
# Stream hashing and copy operations so the full waveform collection is never held
# in memory merely to create or verify a transport archive.
COPY_CHUNK_BYTES = 1024 * 1024


class NotebookHandoffError(RuntimeError):
    """Report a fail-closed handoff creation or restoration contract violation."""


class _FileEntry(TypedDict):
    """Describe one digest-verified repository file in the handoff manifest."""

    role: str
    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class HandoffResult:
    """Summarize one created or restored handoff without exposing patient-level data.

    Attributes:
        operation: ``created``, ``verified_existing``, or ``restored``.
        archive: Absolute path to the verified handoff archive.
        run_id: Governed pipeline run identifier transported by the archive.
        repository_commit: Exact Git commit required by the handoff.
        file_count: Number of repository files included, excluding the manifest.
        total_bytes: Sum of uncompressed included repository file sizes.
    """

    operation: str
    archive: Path
    run_id: str
    repository_commit: str
    file_count: int
    total_bytes: int


def _load_json_object(path: Path, *, description: str) -> dict[str, Any]:
    """Load one required JSON object with a handoff-specific failure message.

    Args:
        path: JSON file to read.
        description: Human-readable role used in raised diagnostics.

    Returns:
        Parsed JSON object.
    """

    # Missing prerequisite evidence means Step 0 did not produce a complete source
    # state; creating a partial handoff would only defer the failure to another VM.
    if not path.is_file():
        raise NotebookHandoffError(f"Missing {description}: {path}")
    # Convert parser details into one stable path-specific contract failure.
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise NotebookHandoffError(f"Invalid JSON in {description}: {path}") from exc
    # All handoff inputs and manifests are named-field objects, never lists/scalars.
    if not isinstance(value, dict):
        raise NotebookHandoffError(f"Expected a JSON object for {description}: {path}")
    return value


def _repository_root(path: Path) -> Path:
    """Resolve and validate the repository root used for handoff paths.

    Args:
        path: Candidate checkout root.

    Returns:
        Resolved repository root.
    """

    root = path.resolve()
    # Requiring both metadata and source prevents extraction into an arbitrary
    # caller-controlled directory that merely happens to exist.
    if not (root / "pyproject.toml").is_file() or not (root / "src").is_dir():
        raise NotebookHandoffError(f"Not a repository root: {root}")
    return root


def _relative_existing_file(root: Path, value: object, *, description: str) -> Path:
    """Resolve one repository-relative file while rejecting escapes and omissions.

    Args:
        root: Validated repository root.
        value: Candidate repository-relative path from generated JSON.
        description: Human-readable file role for diagnostics.

    Returns:
        Normalized repository-relative path.
    """

    # Generated contracts must name paths explicitly; accepting null or a non-string
    # would make the selected handoff contents ambiguous.
    if not isinstance(value, str) or not value:
        raise NotebookHandoffError(f"{description} does not name a file path")
    relative = Path(value)
    # Absolute paths and parent traversal could copy arbitrary host files into Drive.
    if relative.is_absolute() or ".." in relative.parts:
        raise NotebookHandoffError(f"Unsafe {description} path: {value}")
    resolved = (root / relative).resolve()
    # Resolve symlinks before the containment check so a tracked-looking path cannot
    # escape through a symlink into unrelated host data.
    try:
        normalized = resolved.relative_to(root)
    except ValueError as exc:
        raise NotebookHandoffError(f"{description} escapes repository root: {value}") from exc
    # Every manifest member is verified before archive creation begins.
    if not resolved.is_file():
        raise NotebookHandoffError(f"Missing {description}: {normalized.as_posix()}")
    return normalized


def _sha256_stream(source: BinaryIO) -> tuple[str, int]:
    """Return the SHA-256 digest and byte count for one open binary stream.

    Args:
        source: Binary stream positioned at the start of the content.

    Returns:
        Hex digest and observed byte count.
    """

    digest = hashlib.sha256()
    size_bytes = 0
    # Chunked reads bound memory while still detecting truncation independently of
    # archive metadata's advertised uncompressed size.
    for chunk in iter(lambda: source.read(COPY_CHUNK_BYTES), b""):
        digest.update(chunk)
        size_bytes += len(chunk)
    return digest.hexdigest(), size_bytes


def _sha256_file(path: Path) -> tuple[str, int]:
    """Return the SHA-256 digest and byte count for one filesystem file.

    Args:
        path: File to hash.

    Returns:
        Hex digest and observed byte count.
    """

    # Open once and reuse the streaming implementation shared with archive members.
    with path.open("rb") as source:
        return _sha256_stream(source)


def _git_commit(root: Path) -> str:
    """Return the checkout's exact commit or fail before producing a handoff.

    Args:
        root: Validated repository root.

    Returns:
        Forty-character lowercase Git object ID.
    """

    # The exact source revision is part of downstream reproducibility; a source
    # archive without Git metadata cannot produce an attributable Colab handoff.
    git = shutil.which("git")
    # A source checkout without Git cannot establish the exact commit required by
    # downstream restoration; do not rely on a relative executable lookup.
    if git is None:
        raise NotebookHandoffError("Git is unavailable; cannot determine repository commit")
    # The resolved executable and literal arguments form a fixed no-shell probe;
    # caller-controlled data is limited to the already-validated working directory.
    result = subprocess.run(  # noqa: S603
        [str(Path(git).resolve()), "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    commit = result.stdout.strip().lower()
    # Fail closed on command errors or unexpected output rather than recording an
    # empty/abbreviated identity that a fresh runtime could not check out exactly.
    if (
        result.returncode != 0
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise NotebookHandoffError("Unable to determine an exact repository commit")
    return commit


def _collect_handoff_files(root: Path) -> tuple[str, dict[str, Path]]:
    """Select the completed Step 0 files allowed in a Colab handoff.

    Args:
        root: Validated repository root.

    Returns:
        Run identifier and mapping of semantic role to relative file path.
    """

    status_relative = Path("notebooks/local/step0-pipeline-status.json")
    status = _load_json_object(root / status_relative, description="Step 0 status")
    # A blocked/failed run must never be made portable as if it were downstream-ready.
    if status.get("status") != "complete":
        raise NotebookHandoffError(f"Step 0 status is not complete: {status.get('status')!r}")
    artifacts = status.get("artifacts")
    # Notebook 00's strict status contract records the four downstream evidence paths
    # in one object; do not rediscover a potentially different newest run.
    if not isinstance(artifacts, dict):
        raise NotebookHandoffError("Step 0 status is missing its artifacts object")

    files: dict[str, Path] = {
        "step0_status": status_relative,
        "dataset_index": _relative_existing_file(
            root, artifacts.get("dataset_index"), description="dataset index"
        ),
        "split_manifest": _relative_existing_file(
            root, artifacts.get("split_manifest"), description="split manifest"
        ),
        "split_quality": _relative_existing_file(
            root, artifacts.get("split_quality"), description="split quality summary"
        ),
        "run_manifest": _relative_existing_file(
            root, artifacts.get("run_manifest"), description="run manifest"
        ),
    }
    run_id = files["dataset_index"].parent.name
    # Cross-check every run-scoped evidence path so a handoff cannot silently combine
    # artifacts from two local pipeline runs.
    expected_artifact_parent = Path("artifacts/runs") / run_id
    # Check the three required run artifacts independently for clear diagnostics.
    for role in ("split_manifest", "split_quality", "run_manifest"):
        # Each evidence file may live in a nested subdirectory only when explicitly
        # selected below (the required three all sit directly under the run root).
        if files[role].parent != expected_artifact_parent:
            raise NotebookHandoffError(
                f"{role} does not belong to dataset-index run {run_id}: {files[role]}"
            )

    index = _load_json_object(root / files["dataset_index"], description="dataset index")
    partitions = index.get("partitions")
    # Preserve the index's full lineage metadata while selecting file content only
    # from train/validation; all three partition descriptors must still be present.
    if not isinstance(partitions, dict) or set(partitions) != {
        "train",
        "validation",
        "test",
    }:
        raise NotebookHandoffError(
            "Dataset index must expose train, validation, and protected test metadata"
        )
    # Add every development shard under a role that preserves partition and record
    # identity without reading any protected-test shard bytes.
    for partition_name in INCLUDED_PARTITIONS:
        partition = partitions.get(partition_name)
        # Each transported development partition must expose an explicit shard list.
        if not isinstance(partition, dict) or not isinstance(partition.get("shards"), list):
            raise NotebookHandoffError(
                f"Dataset index partition {partition_name!r} has no shard list"
            )
        # Preserve one role per record so duplicates remain detectable in the manifest.
        for shard in partition["shards"]:
            # Shard descriptors are JSON objects in the dataset-index schema.
            if not isinstance(shard, dict):
                raise NotebookHandoffError(
                    f"Dataset index partition {partition_name!r} has an invalid shard"
                )
            file_metadata = shard.get("file")
            record_id = shard.get("record_id")
            # Both the nested file object and record identity are required for selection.
            if not isinstance(file_metadata, dict) or not isinstance(record_id, str):
                raise NotebookHandoffError(
                    f"Dataset index partition {partition_name!r} has incomplete shard metadata"
                )
            role = f"{partition_name}_shard:{record_id}"
            files[role] = _relative_existing_file(
                root,
                file_metadata.get("path"),
                description=f"{partition_name} shard {record_id}",
            )

    # The comparison table in notebook 02 is optional, so include its validation-only
    # baseline when present without turning its absence into a handoff failure.
    baseline = expected_artifact_parent / "evaluation/validation-metrics.json"
    # Optional baseline comparison evidence remains validation-only when available.
    if (root / baseline).is_file():
        files["validation_baseline_metrics"] = baseline
    return run_id, files


def _manifest_entry(root: Path, *, role: str, relative: Path) -> _FileEntry:
    """Build one digest-bearing manifest entry for a selected repository file.

    Args:
        root: Validated repository root.
        role: Stable semantic role assigned during collection.
        relative: Repository-relative file path.

    Returns:
        JSON-serializable manifest entry.
    """

    digest, size_bytes = _sha256_file(root / relative)
    return {
        "role": role,
        "path": relative.as_posix(),
        "sha256": digest,
        "size_bytes": size_bytes,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Write one JSON object atomically beside its final destination.

    Args:
        path: Destination JSON path.
        payload: JSON-serializable object to write.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    # NamedTemporaryFile on the same filesystem lets os.replace provide one atomic
    # pointer update even when Google Drive is the destination mount.
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        json.dump(payload, temporary, indent=2, sort_keys=True)
        temporary.write("\n")
    # Replace only after the complete JSON payload has been flushed and closed.
    os.replace(temporary_path, path)


def create_handoff(repository_root: Path, destination_directory: Path) -> HandoffResult:
    """Create a verified, versioned Colab handoff and update ``latest.json``.

    Args:
        repository_root: Checkout containing completed Step 0 generated state.
        destination_directory: Private persistent directory, normally in Google Drive.

    Returns:
        Summary of the newly created archive.
    """

    root = _repository_root(repository_root)
    run_id, selected_files = _collect_handoff_files(root)
    repository_commit = _git_commit(root)
    # Sort by repository path for stable review output and archive member order.
    entries = sorted(
        (
            _manifest_entry(root, role=role, relative=relative)
            for role, relative in selected_files.items()
        ),
        key=lambda entry: str(entry["path"]),
    )
    total_bytes = sum(entry["size_bytes"] for entry in entries)
    manifest: dict[str, object] = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "execution_profile": "colab",
        "repository_commit": repository_commit,
        "run_id": run_id,
        "included_partitions": list(INCLUDED_PARTITIONS),
        "excluded_content": [
            "raw acquisition files",
            "interim protected-test shards",
            "protected-test metrics or predictions",
        ],
        "claim_boundary": (
            "private notebook transport for research/educational validation-only workflow; "
            "not benchmark, clinical, diagnostic, monitoring, treatment, or production evidence"
        ),
        "files": entries,
    }

    destination = destination_directory.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f"step0-{run_id}-{repository_commit[:12]}.zip"
    # A harmless rerun must be idempotent.  Verify an existing same-run/same-commit
    # archive against the current source files instead of rewriting or trusting it.
    if archive.exists():
        existing = restore_handoff(root, archive)
        archive_digest, archive_size = _sha256_file(archive)
        pointer = {
            "schema_version": HANDOFF_SCHEMA_VERSION,
            "archive_name": archive.name,
            "archive_sha256": archive_digest,
            "archive_size_bytes": archive_size,
            "repository_commit": repository_commit,
            "run_id": run_id,
        }
        _write_json_atomic(destination / HANDOFF_POINTER_NAME, pointer)
        return HandoffResult(
            operation="verified_existing",
            archive=existing.archive,
            run_id=existing.run_id,
            repository_commit=existing.repository_commit,
            file_count=existing.file_count,
            total_bytes=existing.total_bytes,
        )
    # Reserve a same-filesystem temporary filename before opening the ZIP writer.
    with tempfile.NamedTemporaryFile(
        dir=destination,
        prefix=f".{archive.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_archive = Path(temporary.name)
    # Build the complete ZIP away from its final name so a disconnected Drive mount
    # never leaves a partial archive looking ready.
    try:
        # Keep the archive open only while writing its fixed manifest and selected files.
        with zipfile.ZipFile(
            temporary_archive,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as bundle:
            bundle.writestr(
                HANDOFF_MANIFEST_MEMBER,
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            )
            # NPZ shards are already compressed; storing all members avoids expensive
            # redundant compression while Drive provides the persistent transport.
            for entry in entries:
                bundle.write(root / str(entry["path"]), arcname=str(entry["path"]))
        archive_digest, archive_size = _sha256_file(temporary_archive)
        os.replace(temporary_archive, archive)
    # Remove an incomplete temporary bundle without touching any prior versioned handoff.
    except Exception:
        temporary_archive.unlink(missing_ok=True)
        raise

    pointer = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "archive_name": archive.name,
        "archive_sha256": archive_digest,
        "archive_size_bytes": archive_size,
        "repository_commit": repository_commit,
        "run_id": run_id,
    }
    _write_json_atomic(destination / HANDOFF_POINTER_NAME, pointer)
    return HandoffResult(
        operation="created",
        archive=archive,
        run_id=run_id,
        repository_commit=repository_commit,
        file_count=len(entries),
        total_bytes=total_bytes,
    )


def _parse_manifest(bundle: zipfile.ZipFile) -> dict[str, Any]:
    """Load and validate the fixed manifest member from an open handoff archive.

    Args:
        bundle: Open ZIP handoff.

    Returns:
        Parsed manifest object.
    """

    # Duplicate names make ZIP extraction semantics ambiguous and can hide a second
    # payload behind the manifest's one expected member path.
    names = bundle.namelist()
    # Any duplicate ZIP member makes later path-based lookups ambiguous.
    if len(names) != len(set(names)):
        raise NotebookHandoffError("Handoff archive contains duplicate member names")
    # The fixed manifest is required before other members can be interpreted safely.
    if HANDOFF_MANIFEST_MEMBER not in names:
        raise NotebookHandoffError("Handoff archive is missing its manifest")
    # A bounded JSON manifest is small; decode it directly while keeping waveform
    # members on the streaming path below.
    # Decode and parse together so malformed text and JSON share one stable error.
    try:
        manifest = json.loads(bundle.read(HANDOFF_MANIFEST_MEMBER).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NotebookHandoffError("Handoff manifest is not valid UTF-8 JSON") from exc
    # A named-field object is the only supported manifest top-level shape.
    if not isinstance(manifest, dict):
        raise NotebookHandoffError("Handoff manifest must contain a JSON object")
    # Reject future/older schemas rather than assuming compatible security semantics.
    if manifest.get("schema_version") != HANDOFF_SCHEMA_VERSION:
        raise NotebookHandoffError(
            f"Unsupported handoff schema: {manifest.get('schema_version')!r}"
        )
    # Only the explicitly selected hosted profile is allowed to produce this transport.
    if manifest.get("execution_profile") != "colab":
        raise NotebookHandoffError("Handoff manifest does not record the Colab profile")
    # Partition selection is a fixed policy boundary, not caller-controlled metadata.
    if manifest.get("included_partitions") != list(INCLUDED_PARTITIONS):
        raise NotebookHandoffError(
            "Handoff manifest must include train/validation and exclude protected test"
        )
    return manifest


def _validated_manifest_entries(
    manifest: Mapping[str, Any], bundle_names: Iterable[str]
) -> tuple[_FileEntry, ...]:
    """Validate member paths, digests, sizes, and exact ZIP membership.

    Args:
        manifest: Parsed handoff manifest.
        bundle_names: Names present in the open ZIP file.

    Returns:
        Validated immutable sequence of file entries.
    """

    raw_entries = manifest.get("files")
    # Empty/non-list entries cannot describe a usable downstream handoff.
    if not isinstance(raw_entries, list) or not raw_entries:
        raise NotebookHandoffError("Handoff manifest has no file entries")
    entries: list[_FileEntry] = []
    paths: set[str] = set()
    # Validate each manifest entry before writing any archive content to the checkout.
    for raw_entry in raw_entries:
        # Each list item must expose the four fixed entry fields below.
        if not isinstance(raw_entry, dict):
            raise NotebookHandoffError("Handoff manifest contains a non-object file entry")
        role = raw_entry.get("role")
        path_value = raw_entry.get("path")
        digest = raw_entry.get("sha256")
        size_bytes = raw_entry.get("size_bytes")
        # Semantic role is required for later dataset-index boundary checks.
        if not isinstance(role, str) or not role:
            raise NotebookHandoffError("Handoff file entry has no semantic role")
        # Empty/non-string paths cannot safely map to archive members.
        if not isinstance(path_value, str) or not path_value:
            raise NotebookHandoffError("Handoff file entry has no path")
        pure_path = PurePosixPath(path_value)
        # ZIP member paths are POSIX regardless of host OS.  Absolute paths, parent
        # traversal, backslashes, and directory entries are all forbidden.
        if (
            pure_path.is_absolute()
            or ".." in pure_path.parts
            or "\\" in path_value
            or path_value.endswith("/")
        ):
            raise NotebookHandoffError(f"Unsafe handoff member path: {path_value}")
        # One repository path may appear only once regardless of semantic role.
        if path_value in paths:
            raise NotebookHandoffError(f"Duplicate handoff file path: {path_value}")
        # Digests must be canonical lowercase SHA-256 strings.
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise NotebookHandoffError(f"Invalid SHA-256 for handoff member: {path_value}")
        # Negative or non-integral sizes cannot be compared with streamed byte counts.
        if not isinstance(size_bytes, int) or size_bytes < 0:
            raise NotebookHandoffError(f"Invalid size for handoff member: {path_value}")
        paths.add(path_value)
        entries.append(
            {
                "role": role,
                "path": path_value,
                "sha256": digest,
                "size_bytes": size_bytes,
            }
        )
    # Reject extra ZIP members because an unmanifested payload would bypass digest and
    # protected-partition selection checks even if callers never intended to use it.
    expected_names = paths | {HANDOFF_MANIFEST_MEMBER}
    # Exact membership prevents unverified payloads from hiding beside declared files.
    if set(bundle_names) != expected_names:
        extra = sorted(set(bundle_names) - expected_names)
        missing = sorted(expected_names - set(bundle_names))
        raise NotebookHandoffError(
            f"Handoff ZIP membership differs from manifest; extra={extra}, missing={missing}"
        )
    return tuple(entries)


def _validate_partition_boundary(
    staging_root: Path, entries: tuple[_FileEntry, ...], *, manifest_run_id: object
) -> None:
    """Prove staged shard members belong only to train or validation.

    Args:
        staging_root: Temporary extraction root containing verified files.
        entries: Validated manifest entries.
        manifest_run_id: Run identity recorded independently at manifest top level.
    """

    roles = [entry["role"] for entry in entries]
    # Roles, like paths, must be unique so one semantic slot cannot be shadowed.
    if len(roles) != len(set(roles)):
        raise NotebookHandoffError("Handoff manifest contains duplicate file roles")
    index_entries = [entry for entry in entries if entry["role"] == "dataset_index"]
    # A single index anchors all transported partition membership.
    if len(index_entries) != 1:
        raise NotebookHandoffError("Handoff must contain exactly one dataset index")
    index = _load_json_object(
        staging_root / str(index_entries[0]["path"]), description="staged dataset index"
    )
    index_path = Path(index_entries[0]["path"])
    # The index location itself establishes the one run namespace restored below.
    if (
        len(index_path.parts) != 5
        or index_path.parts[:3] != ("data", "processed", "runs")
        or index_path.name != "dataset-index.json"
    ):
        raise NotebookHandoffError(
            f"Dataset index uses an unsupported handoff path: {index_path.as_posix()}"
        )
    run_id = index_path.parent.name
    # Top-level manifest lineage and the index's relative namespace must agree.
    if not isinstance(manifest_run_id, str) or manifest_run_id != run_id:
        raise NotebookHandoffError("Handoff run ID does not match the dataset-index path")

    expected_static_roles = {
        "step0_status": "notebooks/local/step0-pipeline-status.json",
        "dataset_index": index_path.as_posix(),
        "split_manifest": f"artifacts/runs/{run_id}/split.json",
        "split_quality": f"artifacts/runs/{run_id}/split_quality_summary.json",
        "run_manifest": f"artifacts/runs/{run_id}/run-manifest.json",
    }
    actual_by_role = {entry["role"]: entry["path"] for entry in entries}
    # Required evidence roles each have one supported repository-relative path.
    for role, expected_path in expected_static_roles.items():
        # A mismatched role/path pair could otherwise overwrite unrelated ignored state.
        if actual_by_role.get(role) != expected_path:
            raise NotebookHandoffError(f"Handoff role {role!r} does not map to {expected_path}")
    optional_baseline = actual_by_role.get("validation_baseline_metrics")
    # The one optional artifact is validation-only and remains inside the same run.
    if optional_baseline is not None and optional_baseline != (
        f"artifacts/runs/{run_id}/evaluation/validation-metrics.json"
    ):
        raise NotebookHandoffError(
            "Validation baseline role maps outside its supported run-relative path"
        )

    allowed_nonshard_roles = set(expected_static_roles) | {"validation_baseline_metrics"}
    # Any other role must be a train/validation shard selected from the index below.
    for role in actual_by_role:
        # Static required/optional roles have already passed exact path checks.
        if role in allowed_nonshard_roles:
            continue
        # Reject arbitrary role names even when their member paths are syntactically safe.
        if not role.startswith(("train_shard:", "validation_shard:")):
            raise NotebookHandoffError(f"Unsupported handoff file role: {role}")

    status = _load_json_object(
        staging_root / expected_static_roles["step0_status"],
        description="staged Step 0 status",
    )
    # Only the strict success state may cross the VM boundary as downstream-ready.
    if status.get("status") != "complete":
        raise NotebookHandoffError("Staged Step 0 status is not complete")
    status_artifacts = status.get("artifacts")
    status_roles = (
        ("dataset_index", "dataset_index"),
        ("split_manifest", "split_manifest"),
        ("split_quality", "split_quality"),
        ("run_manifest", "run_manifest"),
    )
    # The status must point to the same exact evidence paths checked independently.
    if not isinstance(status_artifacts, dict) or any(
        status_artifacts.get(status_key) != expected_static_roles[role]
        for status_key, role in status_roles
    ):
        raise NotebookHandoffError(
            "Staged Step 0 status artifact paths do not match the handoff run"
        )
    partitions = index.get("partitions")
    # Missing partitions make it impossible to prove protected-test exclusion.
    if not isinstance(partitions, dict):
        raise NotebookHandoffError("Staged dataset index has no partitions object")

    expected_shards: set[str] = set()
    protected_shards: set[str] = set()
    # Read path metadata only; no protected-test shard bytes are present or opened.
    for partition_name in ("train", "validation", "test"):
        partition = partitions.get(partition_name)
        # Every partition, including protected test metadata, must be structurally present.
        if not isinstance(partition, dict) or not isinstance(partition.get("shards"), list):
            raise NotebookHandoffError(f"Staged dataset index has no {partition_name} shard list")
        # Inspect only path metadata; no protected shard file is in the archive.
        for shard in partition["shards"]:
            # Each index shard must expose its nested file descriptor.
            if not isinstance(shard, dict) or not isinstance(shard.get("file"), dict):
                raise NotebookHandoffError(
                    f"Staged dataset index has invalid {partition_name} shard metadata"
                )
            path_value = shard["file"].get("path")
            # A missing path cannot participate in the exact membership proof.
            if not isinstance(path_value, str):
                raise NotebookHandoffError(
                    f"Staged dataset index has missing {partition_name} shard path"
                )
            # Keep protected paths separate so intersection is asserted explicitly.
            if partition_name == "test":
                protected_shards.add(path_value)
            else:
                expected_shards.add(path_value)

    included_shards = {
        str(entry["path"])
        for entry in entries
        if str(entry["role"]).startswith(("train_shard:", "validation_shard:"))
    }
    # Require all and only train/validation shard paths from the transported index.
    if included_shards != expected_shards:
        raise NotebookHandoffError(
            "Handoff shard membership does not exactly match train/validation index entries"
        )
    # This redundant disjointness assertion makes the protected boundary auditable.
    if included_shards & protected_shards:
        raise NotebookHandoffError("Handoff includes a protected-test shard")


def restore_handoff(repository_root: Path, archive: Path) -> HandoffResult:
    """Verify and restore one Colab handoff into a fresh exact-commit checkout.

    Args:
        repository_root: Checkout whose current commit must match the manifest.
        archive: Handoff ZIP in the reviewer's persistent storage.

    Returns:
        Summary of the verified restored state.
    """

    root = _repository_root(repository_root)
    archive_path = archive.expanduser().resolve()
    # Refuse a missing Drive selection before opening any repository destination.
    if not archive_path.is_file():
        raise NotebookHandoffError(f"Handoff archive does not exist: {archive_path}")
    # Keep member reads within one validated ZIP handle for consistent membership.
    with zipfile.ZipFile(archive_path, mode="r") as bundle:
        manifest = _parse_manifest(bundle)
        entries = _validated_manifest_entries(manifest, bundle.namelist())
        repository_commit = manifest.get("repository_commit")
        # Exact source identity is checked before staging any generated file.
        if not isinstance(repository_commit, str) or _git_commit(root) != repository_commit:
            raise NotebookHandoffError(
                "Checkout commit does not match the handoff repository commit"
            )
        # Stage every verified member before changing the checkout's ignored generated
        # state, so digest, size, or partition failures leave destinations untouched.
        staging_parent = root / "notebooks/local"
        staging_parent.mkdir(parents=True, exist_ok=True)
        # Staging inside the checkout keeps later moves atomic on one filesystem.
        with tempfile.TemporaryDirectory(
            dir=staging_parent, prefix=".colab-handoff-restore-"
        ) as temporary_directory:
            staging_root = Path(temporary_directory)
            # Extract every member into staging while hashing the actual bytes read.
            for entry in entries:
                relative = Path(str(entry["path"]))
                staged = staging_root / relative
                staged.parent.mkdir(parents=True, exist_ok=True)
                # Open source/destination together so both handles close before validation.
                with (
                    bundle.open(str(entry["path"]), mode="r") as archive_source,
                    staged.open("wb") as destination,
                ):
                    digest = hashlib.sha256()
                    size_bytes = 0
                    # Copy and verify in one pass to avoid reading every waveform file
                    # twice during restore.
                    # Bound memory while verifying the complete uncompressed member.
                    for chunk in iter(lambda: archive_source.read(COPY_CHUNK_BYTES), b""):
                        destination.write(chunk)
                        digest.update(chunk)
                        size_bytes += len(chunk)
                # Both digest and byte count must match before partition inspection.
                if digest.hexdigest() != entry["sha256"] or size_bytes != entry["size_bytes"]:
                    raise NotebookHandoffError(
                        f"Handoff member failed digest/size verification: {relative.as_posix()}"
                    )

            _validate_partition_boundary(
                staging_root, entries, manifest_run_id=manifest.get("run_id")
            )
            # Refuse to overwrite different generated state.  Identical destinations
            # make restore idempotent after a harmless cell rerun.
            for entry in entries:
                relative = Path(str(entry["path"]))
                destination = root / relative
                # Existing ignored state is acceptable only when byte-identical.
                if destination.exists():
                    observed_digest, observed_size = _sha256_file(destination)
                    # A difference signals another run/local edit and must not be overwritten.
                    if observed_digest != entry["sha256"] or observed_size != entry["size_bytes"]:
                        raise NotebookHandoffError(
                            f"Refusing to overwrite different generated file: {relative.as_posix()}"
                        )
            # Only after the complete staging and overwrite audit succeeds do files
            # become visible at their repository-relative generated locations.
            for entry in entries:
                relative = Path(str(entry["path"]))
                staged_source = staging_root / relative
                destination = root / relative
                # Identical destinations were verified above and need no rewrite.
                if destination.exists():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(staged_source), str(destination))

    run_id = manifest.get("run_id")
    # Run identity is required in the returned notebook-facing confirmation.
    if not isinstance(run_id, str) or not run_id:
        raise NotebookHandoffError("Handoff manifest has no run identifier")
    return HandoffResult(
        operation="restored",
        archive=archive_path,
        run_id=run_id,
        repository_commit=str(manifest["repository_commit"]),
        file_count=len(entries),
        total_bytes=sum(entry["size_bytes"] for entry in entries),
    )
