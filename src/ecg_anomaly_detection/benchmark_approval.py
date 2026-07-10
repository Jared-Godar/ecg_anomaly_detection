"""Fail-closed approval-and-lineage verification gate for a future held-out benchmark.

This module implements governance steps 1-2 of `docs/benchmark-governance.md`'s execution
procedure: recording approval and verifying immutable lineage references before any future
benchmark command may run. It never opens, reads, or scores the protected `test` partition,
and it never accesses a model or dataset directly -- it only inspects an already-written
`BenchmarkPolicy` and `RunManifest`.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ecg_anomaly_detection.benchmark_policy import (
    BenchmarkPolicy,
    BenchmarkPolicyError,
    load_benchmark_policy,
)
from ecg_anomaly_detection.run_manifest import RunManifest, RunManifestError, read_run_manifest

# Centralize CONFIGURATION_HASH_REFERENCES so every caller shares the same documented invariant.
CONFIGURATION_HASH_REFERENCES = frozenset(
    {"dataset_configuration_hash", "training_configuration_hash", "evaluation_configuration_hash"}
)


class BenchmarkApprovalError(ValueError):
    """Raised when benchmark approval or lineage verification fails closed."""


@dataclass(frozen=True, slots=True)
class ApprovalInput:
    """Human-authored, separately reviewed request to gate one benchmark candidate."""

    schema_version: int
    owner: str
    candidate_run_id: str
    purpose: str
    prior_attempt_exists: bool
    lineage_configuration_paths: dict[str, str]


@dataclass(frozen=True, slots=True)
class ApprovalRecord:
    """Audit evidence that the approval-and-lineage gate was checked for one candidate.

    This is not a benchmark result: it records that the gate was checked, not that any
    protected-test execution occurred.
    """

    schema_version: int
    policy_id: str
    policy_version: str
    owner: str
    candidate_run_id: str
    purpose: str
    prior_attempt_exists: bool
    run_manifest_reference: str
    verified_lineage_references: tuple[str, ...]

    def to_json(self) -> str:
        """Serialize this structured record as deterministic JSON.

        The helper isolates this step so its assumptions, outputs, and failure behavior remain
        reviewable.

        Returns:
            The value produced by the documented operation.
        """

        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


def load_approval_input(path: Path) -> ApprovalInput:
    """Load a human-authored approval-input record without accessing data or model artifacts."""
    # Attempt this boundary operation here so (OSError, tomllib.TOMLDecodeError) can be translated
    # or cleaned up under the repository contract.
    try:
        # Scope `path.open('rb')` here so resource cleanup occurs on both success and failure paths.
        with path.open("rb") as source:
            document = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise BenchmarkApprovalError(f"could not load approval input {path}: {error}") from error
    # Evaluate `document.get('schema_version') != 1` explicitly so invalid or alternate states
    # follow the documented contract.
    if document.get("schema_version") != 1:
        raise BenchmarkApprovalError("approval input must use schema_version = 1")
    approval = document.get("approval")
    # Evaluate `not isinstance(approval, dict)` explicitly so invalid or alternate states follow the
    # documented contract.
    if not isinstance(approval, dict):
        raise BenchmarkApprovalError("approval input requires an [approval] table")
    prior_attempt_exists = approval.get("prior_attempt_exists")
    # Evaluate `not isinstance(prior_attempt_exists, bool)` explicitly so invalid or alternate
    # states follow the documented contract.
    if not isinstance(prior_attempt_exists, bool):
        raise BenchmarkApprovalError("approval.prior_attempt_exists must be a boolean")
    return ApprovalInput(
        schema_version=1,
        owner=_string(approval, "owner"),
        candidate_run_id=_string(approval, "candidate_run_id"),
        purpose=_string(approval, "purpose"),
        prior_attempt_exists=prior_attempt_exists,
        lineage_configuration_paths=_lineage_configuration_paths(approval),
    )


def record_benchmark_approval(
    repository_root: Path,
    policy_path: Path,
    run_manifest_path: Path,
    approval_input_path: Path,
    output_path: Path,
) -> ApprovalRecord:
    """Verify eligibility, approval, and lineage, then write approval-gate evidence.

    Fails closed on a disabled policy, a candidate/manifest mismatch, or any missing
    lineage reference. Never opens, lists, or scores files under the `test` partition.
    """
    root = repository_root.resolve()
    resolved_manifest_path = _input_path(root, run_manifest_path)
    resolved_output_path = _output_path(root, output_path)

    # Attempt this boundary operation here so BenchmarkPolicyError can be translated or cleaned up
    # under the repository contract.
    try:
        policy = load_benchmark_policy(policy_path)
    except BenchmarkPolicyError as error:
        raise BenchmarkApprovalError(str(error)) from error
    approval = load_approval_input(approval_input_path)
    # Attempt this boundary operation here so RunManifestError can be translated or cleaned up under
    # the repository contract.
    try:
        manifest = read_run_manifest(resolved_manifest_path)
    except RunManifestError as error:
        raise BenchmarkApprovalError(str(error)) from error

    missing = _missing_lineage_references(policy, approval, manifest)
    # Evaluate `missing` explicitly so invalid or alternate states follow the documented contract.
    if missing:
        raise BenchmarkApprovalError(f"missing required lineage references: {missing}")

    record = ApprovalRecord(
        schema_version=1,
        policy_id=policy.policy_id,
        policy_version=policy.version,
        owner=approval.owner,
        candidate_run_id=approval.candidate_run_id,
        purpose=approval.purpose,
        prior_attempt_exists=approval.prior_attempt_exists,
        run_manifest_reference=manifest.run_id,
        verified_lineage_references=tuple(sorted(policy.required_lineage_references)),
    )
    _write_new(resolved_output_path, record.to_json())
    return record


def _missing_lineage_references(
    policy: BenchmarkPolicy, approval: ApprovalInput, manifest: RunManifest
) -> list[str]:
    """Compute and return missing lineage references for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        policy: The policy value supplied by the caller or surrounding test fixture.
        approval: The approval value supplied by the caller or surrounding test fixture.
        manifest: The manifest value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    missing: list[str] = []
    # Iterate over `sorted(policy.required_lineage_references)` one item at a time so ordering,
    # validation, and failure attribution remain explicit.
    for reference in sorted(policy.required_lineage_references):
        # Evaluate `reference == 'repository_commit_hash'` explicitly so invalid or alternate states
        # follow the documented contract.
        if reference == "repository_commit_hash":
            # Evaluate `not _valid_commit_hash(manifest.git.revision)` explicitly so invalid or
            # alternate states follow the documented contract.
            if not _valid_commit_hash(manifest.git.revision):
                missing.append(reference)
        elif reference == "split_identity":
            # Evaluate `not (manifest.split.split_name.strip() and
            # manifest.split.split_version.strip())` explicitly so invalid or alternate states
            # follow the documented contract.
            if not (manifest.split.split_name.strip() and manifest.split.split_version.strip()):
                missing.append(reference)
        elif reference == "reproducibility_evidence_reference":
            # Evaluate `not manifest.evidence_files` explicitly so invalid or alternate states
            # follow the documented contract.
            if not manifest.evidence_files:
                missing.append(reference)
        elif reference == "run_manifest_reference":
            # Evaluate `not manifest.run_id.strip() or manifest.run_id != approval.candidate_run_id`
            # explicitly so invalid or alternate states follow the documented contract.
            if not manifest.run_id.strip() or manifest.run_id != approval.candidate_run_id:
                missing.append(reference)
        elif reference in CONFIGURATION_HASH_REFERENCES:
            # Evaluate `not _configuration_hash_present(reference, approval, manifest)` explicitly
            # so invalid or alternate states follow the documented contract.
            if not _configuration_hash_present(reference, approval, manifest):
                missing.append(reference)
        else:
            # A policy may require references beyond the known 7; without a defined
            # check for it, it cannot be verified present, so it fails closed.
            missing.append(reference)
    return missing


def _configuration_hash_present(
    reference: str, approval: ApprovalInput, manifest: RunManifest
) -> bool:
    """Return whether configuration hash present under the documented validation contract.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        reference: The reference value supplied by the caller or surrounding test fixture.
        approval: The approval value supplied by the caller or surrounding test fixture.
        manifest: The manifest value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    configured_path = approval.lineage_configuration_paths.get(reference)
    # Evaluate `not configured_path` explicitly so invalid or alternate states follow the documented
    # contract.
    if not configured_path:
        return False
    return any(
        entry.path == configured_path and entry.sha256.strip()
        for entry in manifest.configuration_files
    )


def _valid_commit_hash(revision: str) -> bool:
    """Return whether valid commit hash under the documented validation contract.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        revision: The revision value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    return len(revision) == 40 and all(character in "0123456789abcdef" for character in revision)


def _lineage_configuration_paths(approval: dict[str, Any]) -> dict[str, str]:
    """Compute and return lineage configuration paths for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        approval: The approval value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    value = approval.get("lineage_configuration_paths")
    # Evaluate `not isinstance(value, dict) or not all((isinstance(key, str) and isinstance(item,
    # str) and item.strip() for key, item...` explicitly so invalid or alternate states follow the
    # documented contract.
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) and item.strip()
        for key, item in value.items()
    ):
        raise BenchmarkApprovalError(
            "approval.lineage_configuration_paths must map lineage reference names to "
            "non-empty repository-relative paths"
        )
    return {key: item.strip() for key, item in value.items()}


def _string(values: dict[str, Any], key: str) -> str:
    """Require and return a non-empty string from the requested structured field.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        values: Structured values to validate, transform, or serialize.
        key: The key value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    value = values.get(key)
    # Evaluate `not isinstance(value, str) or not value.strip()` explicitly so invalid or alternate
    # states follow the documented contract.
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkApprovalError(f"approval.{key} must be a non-empty string")
    return value.strip()


def _input_path(root: Path, path: Path) -> Path:
    """Resolve and validate input path for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        root: Repository root used to enforce path and trust boundaries.
        path: Filesystem path identifying the input or output under review.

    Returns:
        The value produced by the documented operation.
    """

    candidate = path if path.is_absolute() else root / path
    # Evaluate `candidate.is_symlink()` explicitly so invalid or alternate states follow the
    # documented contract.
    if candidate.is_symlink():
        raise BenchmarkApprovalError(f"run manifest must not be a symbolic link: {candidate}")
    resolved = candidate.resolve()
    # Attempt this boundary operation here so ValueError can be translated or cleaned up under the
    # repository contract.
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise BenchmarkApprovalError("run manifest must stay within repository root") from error
    # Evaluate `not resolved.is_file()` explicitly so invalid or alternate states follow the
    # documented contract.
    if not resolved.is_file():
        raise BenchmarkApprovalError(f"run manifest must be a regular file: {resolved}")
    return resolved


def _output_path(root: Path, path: Path) -> Path:
    """Resolve and validate output path for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        root: Repository root used to enforce path and trust boundaries.
        path: Filesystem path identifying the input or output under review.

    Returns:
        The value produced by the documented operation.
    """

    candidate = path if path.is_absolute() else root / path
    # Evaluate `candidate.is_symlink()` explicitly so invalid or alternate states follow the
    # documented contract.
    if candidate.is_symlink():
        raise BenchmarkApprovalError("approval output must not be a symbolic link")
    resolved = candidate.resolve()
    # Attempt this boundary operation here so ValueError can be translated or cleaned up under the
    # repository contract.
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise BenchmarkApprovalError("approval output must stay within repository root") from error
    # Evaluate `not relative.parts or relative.parts[0] != 'artifacts'` explicitly so invalid or
    # alternate states follow the documented contract.
    if not relative.parts or relative.parts[0] != "artifacts":
        raise BenchmarkApprovalError("approval output must be written under artifacts/")
    # Evaluate `resolved.suffix != '.json'` explicitly so invalid or alternate states follow the
    # documented contract.
    if resolved.suffix != ".json":
        raise BenchmarkApprovalError("approval output must use the .json extension")
    # Evaluate `not resolved.parent.is_dir()` explicitly so invalid or alternate states follow the
    # documented contract.
    if not resolved.parent.is_dir():
        raise BenchmarkApprovalError(
            f"approval output parent directory does not exist: {resolved.parent}"
        )
    return resolved


def _write_new(path: Path, content: str) -> None:
    """Write new according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        path: Filesystem path identifying the input or output under review.
        content: The content value supplied by the caller or surrounding test fixture.
    """

    # Attempt this boundary operation here so FileExistsError, OSError can be translated or cleaned
    # up under the repository contract.
    try:
        # Scope `path.open('x', encoding='utf-8')` here so resource cleanup occurs on both success
        # and failure paths.
        with path.open("x", encoding="utf-8") as output:
            output.write(content)
    except FileExistsError as error:
        raise BenchmarkApprovalError(f"approval output already exists: {path}") from error
    except OSError as error:
        raise BenchmarkApprovalError(f"could not write approval output {path}: {error}") from error
