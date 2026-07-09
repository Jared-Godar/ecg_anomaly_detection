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
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


def load_approval_input(path: Path) -> ApprovalInput:
    """Load a human-authored approval-input record without accessing data or model artifacts."""
    try:
        with path.open("rb") as source:
            document = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise BenchmarkApprovalError(f"could not load approval input {path}: {error}") from error
    if document.get("schema_version") != 1:
        raise BenchmarkApprovalError("approval input must use schema_version = 1")
    approval = document.get("approval")
    if not isinstance(approval, dict):
        raise BenchmarkApprovalError("approval input requires an [approval] table")
    prior_attempt_exists = approval.get("prior_attempt_exists")
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

    try:
        policy = load_benchmark_policy(policy_path)
    except BenchmarkPolicyError as error:
        raise BenchmarkApprovalError(str(error)) from error
    approval = load_approval_input(approval_input_path)
    try:
        manifest = read_run_manifest(resolved_manifest_path)
    except RunManifestError as error:
        raise BenchmarkApprovalError(str(error)) from error

    missing = _missing_lineage_references(policy, approval, manifest)
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
    missing: list[str] = []
    for reference in sorted(policy.required_lineage_references):
        if reference == "repository_commit_hash":
            if not _valid_commit_hash(manifest.git.revision):
                missing.append(reference)
        elif reference == "split_identity":
            if not (manifest.split.split_name.strip() and manifest.split.split_version.strip()):
                missing.append(reference)
        elif reference == "reproducibility_evidence_reference":
            if not manifest.evidence_files:
                missing.append(reference)
        elif reference == "run_manifest_reference":
            if not manifest.run_id.strip() or manifest.run_id != approval.candidate_run_id:
                missing.append(reference)
        elif reference in CONFIGURATION_HASH_REFERENCES:
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
    configured_path = approval.lineage_configuration_paths.get(reference)
    if not configured_path:
        return False
    return any(
        entry.path == configured_path and entry.sha256.strip()
        for entry in manifest.configuration_files
    )


def _valid_commit_hash(revision: str) -> bool:
    return len(revision) == 40 and all(character in "0123456789abcdef" for character in revision)


def _lineage_configuration_paths(approval: dict[str, Any]) -> dict[str, str]:
    value = approval.get("lineage_configuration_paths")
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
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkApprovalError(f"approval.{key} must be a non-empty string")
    return value.strip()


def _input_path(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    if candidate.is_symlink():
        raise BenchmarkApprovalError(f"run manifest must not be a symbolic link: {candidate}")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise BenchmarkApprovalError("run manifest must stay within repository root") from error
    if not resolved.is_file():
        raise BenchmarkApprovalError(f"run manifest must be a regular file: {resolved}")
    return resolved


def _output_path(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    if candidate.is_symlink():
        raise BenchmarkApprovalError("approval output must not be a symbolic link")
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise BenchmarkApprovalError("approval output must stay within repository root") from error
    if not relative.parts or relative.parts[0] != "artifacts":
        raise BenchmarkApprovalError("approval output must be written under artifacts/")
    if resolved.suffix != ".json":
        raise BenchmarkApprovalError("approval output must use the .json extension")
    if not resolved.parent.is_dir():
        raise BenchmarkApprovalError(
            f"approval output parent directory does not exist: {resolved.parent}"
        )
    return resolved


def _write_new(path: Path, content: str) -> None:
    try:
        with path.open("x", encoding="utf-8") as output:
            output.write(content)
    except FileExistsError as error:
        raise BenchmarkApprovalError(f"approval output already exists: {path}") from error
    except OSError as error:
        raise BenchmarkApprovalError(f"could not write approval output {path}: {error}") from error
