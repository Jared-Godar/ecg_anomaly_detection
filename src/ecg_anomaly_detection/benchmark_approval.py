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

# The subset of a policy's required_lineage_references that _missing_lineage_references
# resolves generically via _configuration_hash_present, rather than a dedicated
# reference-specific branch -- every name in this set follows the same "look up a
# configured path, confirm it appears with a digest in the run manifest" check.
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

        Returns:
            The approval record as a JSON string with sorted, deterministic key ordering.
        """

        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


def load_approval_input(path: Path) -> ApprovalInput:
    """Load a human-authored approval-input record without accessing data or model artifacts."""
    # Translate a missing, unreadable, or malformed-TOML file into BenchmarkApprovalError.
    try:
        # The `with` block ensures the file handle closes even if tomllib.load raises.
        with path.open("rb") as source:
            document = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise BenchmarkApprovalError(f"could not load approval input {path}: {error}") from error
    # schema_version pins this loader's understanding of the [approval] table's shape.
    if document.get("schema_version") != 1:
        raise BenchmarkApprovalError("approval input must use schema_version = 1")
    approval = document.get("approval")
    # Every field access below assumes approval is a dict.
    if not isinstance(approval, dict):
        raise BenchmarkApprovalError("approval input requires an [approval] table")
    prior_attempt_exists = approval.get("prior_attempt_exists")
    # This field is deliberately required (no default) since it's governance-relevant
    # evidence -- a config author must explicitly state whether a prior attempt exists
    # rather than it silently defaulting to False.
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

    # benchmark_policy.py's own exception type is re-raised as this module's, so
    # callers only need to catch BenchmarkApprovalError for every failure mode here.
    try:
        policy = load_benchmark_policy(policy_path)
    except BenchmarkPolicyError as error:
        raise BenchmarkApprovalError(str(error)) from error
    approval = load_approval_input(approval_input_path)
    # Same re-raising pattern as above, for run_manifest.py's exception type.
    try:
        manifest = read_run_manifest(resolved_manifest_path)
    except RunManifestError as error:
        raise BenchmarkApprovalError(str(error)) from error

    missing = _missing_lineage_references(policy, approval, manifest)
    # Any policy-required lineage reference that couldn't be verified present means
    # this candidate isn't yet eligible for the gate this module implements -- fail
    # before writing any approval evidence rather than recording a partial approval.
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
    """Check every lineage reference a policy requires and report which are unverifiable.

    Each reference name maps to a specific, independent verification rule against the
    run manifest and approval input -- this function is the single place all seven
    reference kinds (six named + the generic configuration-hash group) are checked, so
    a policy's requirements can never silently pass without being individually verified.

    Args:
        policy: The benchmark policy naming which lineage references are required.
        approval: The human-authored approval input, supplying candidate identity and
            any configured lineage paths.
        manifest: The run manifest whose recorded evidence is checked against each
            required reference.

    Returns:
        The subset of policy.required_lineage_references that could not be verified.
    """

    missing: list[str] = []
    # Check references in sorted order so the returned missing-list (and any error
    # message built from it) is deterministic regardless of policy file ordering.
    for reference in sorted(policy.required_lineage_references):
        # A valid 40-character hex commit hash is the strongest evidence this run
        # traces back to a specific, inspectable Git revision.
        if reference == "repository_commit_hash":
            # An invalid or absent hash can't serve as verifiable lineage evidence.
            if not _valid_commit_hash(manifest.git.revision):
                missing.append(reference)
        elif reference == "split_identity":
            # Both the split's name and version must be present -- either alone
            # wouldn't uniquely identify which grouped split produced this run.
            if not (manifest.split.split_name.strip() and manifest.split.split_version.strip()):
                missing.append(reference)
        elif reference == "reproducibility_evidence_reference":
            # At least one evidence file (environment/runtime capture) must be
            # recorded; an empty list means reproducibility.py's capture never ran.
            if not manifest.evidence_files:
                missing.append(reference)
        elif reference == "run_manifest_reference":
            # The manifest's own run_id must both be present and match the candidate
            # ID the approval input names -- otherwise this manifest could belong to
            # an entirely different run than the one being approved.
            if not manifest.run_id.strip() or manifest.run_id != approval.candidate_run_id:
                missing.append(reference)
        elif reference in CONFIGURATION_HASH_REFERENCES:
            # These three references share one verification rule; delegate to
            # _configuration_hash_present rather than repeating it three times.
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
    """Confirm one configuration-hash lineage reference resolves to a manifest digest.

    The approval input names which repository-relative config path corresponds to this
    reference; this checks that path actually appears in the run manifest's recorded
    configuration files with a non-empty digest, proving the run manifest itself
    captured that config's exact content.

    Args:
        reference: Which configuration-hash reference to check (a member of
            CONFIGURATION_HASH_REFERENCES).
        approval: Supplies the reference-to-path mapping from lineage_configuration_paths.
        manifest: The run manifest whose configuration_files are checked.

    Returns:
        True if the approval-configured path is present in the manifest with a digest.
    """

    configured_path = approval.lineage_configuration_paths.get(reference)
    # The approval input must itself have named a path for this reference before the
    # manifest can even be checked against it.
    if not configured_path:
        return False
    return any(
        entry.path == configured_path and entry.sha256.strip()
        for entry in manifest.configuration_files
    )


def _valid_commit_hash(revision: str) -> bool:
    """Return whether a string is a well-formed 40-character hex Git commit hash.

    Args:
        revision: The candidate revision string from the run manifest's git evidence.

    Returns:
        True if it's exactly 40 lowercase hex characters (a full SHA-1 hash).
    """

    return len(revision) == 40 and all(character in "0123456789abcdef" for character in revision)


def _lineage_configuration_paths(approval: dict[str, Any]) -> dict[str, str]:
    """Parse the approval input's reference-name-to-config-path mapping.

    Args:
        approval: The parsed `[approval]` table to read from.

    Returns:
        A validated, stripped mapping from lineage reference name to config path.
    """

    value = approval.get("lineage_configuration_paths")
    # Every key must be a string (a reference name) and every value a non-empty string
    # (a repository-relative path); this mapping later drives _configuration_hash_present's
    # lookups, which would silently find nothing for a malformed entry rather than error.
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
    """Require and return a non-empty string from the requested `[approval]` field.

    Args:
        values: The parsed `[approval]` table to read from.
        key: The field name to extract.

    Returns:
        The field's value with surrounding whitespace stripped.
    """

    value = values.get(key)
    # Reject a missing/wrong-typed value and a whitespace-only placeholder alike.
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkApprovalError(f"approval.{key} must be a non-empty string")
    return value.strip()


def _input_path(root: Path, path: Path) -> Path:
    """Resolve the run manifest path and enforce it stays within the repository root.

    Args:
        root: Repository root used to enforce path and trust boundaries.
        path: The candidate run manifest path, absolute or relative to root.

    Returns:
        The resolved, validated absolute path.
    """

    candidate = path if path.is_absolute() else root / path
    # Reject a symlink before resolving it, so a link that points outside the
    # repository can't be validated against a resolved target it doesn't actually name.
    if candidate.is_symlink():
        raise BenchmarkApprovalError(f"run manifest must not be a symbolic link: {candidate}")
    resolved = candidate.resolve()
    # relative_to raises ValueError when resolved escapes root (e.g. via `..` segments);
    # translate that into this module's own exception type.
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise BenchmarkApprovalError("run manifest must stay within repository root") from error
    # A directory or special file at this path would fail read_run_manifest downstream
    # with a less specific error; check it's a regular file up front.
    if not resolved.is_file():
        raise BenchmarkApprovalError(f"run manifest must be a regular file: {resolved}")
    return resolved


def _output_path(root: Path, path: Path) -> Path:
    """Resolve the approval-record output path and enforce it lands under artifacts/.

    Args:
        root: Repository root used to enforce path and trust boundaries.
        path: The candidate output path, absolute or relative to root.

    Returns:
        The resolved, validated absolute path.
    """

    candidate = path if path.is_absolute() else root / path
    # Reject a symlink before resolving it: resolving would silently follow the link and
    # write to wherever it points, defeating the repository-root containment check below.
    if candidate.is_symlink():
        raise BenchmarkApprovalError("approval output must not be a symbolic link")
    resolved = candidate.resolve()
    # relative_to raises ValueError when resolved escapes root.
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise BenchmarkApprovalError("approval output must stay within repository root") from error
    # Approval records are pipeline-generated evidence, matching this repository's
    # directory contract for artifacts.
    if not relative.parts or relative.parts[0] != "artifacts":
        raise BenchmarkApprovalError("approval output must be written under artifacts/")
    # Enforce the extension so downstream tooling that globs for *.json approval
    # records can rely on finding this file without a separate content sniff.
    if resolved.suffix != ".json":
        raise BenchmarkApprovalError("approval output must use the .json extension")
    # Fail before attempting the write rather than letting a missing parent directory
    # surface as a generic OSError from _write_new.
    if not resolved.parent.is_dir():
        raise BenchmarkApprovalError(
            f"approval output parent directory does not exist: {resolved.parent}"
        )
    return resolved


def _write_new(path: Path, content: str) -> None:
    """Write content to a path that must not already exist.

    Args:
        path: Destination path; must not already exist.
        content: The text to write.
    """

    # Collapse "already exists" and other OS-level failures into BenchmarkApprovalError.
    try:
        # Open with mode "x" (exclusive create) so an approval record can never
        # silently overwrite a previous candidate's evidence.
        with path.open("x", encoding="utf-8") as output:
            output.write(content)
    except FileExistsError as error:
        raise BenchmarkApprovalError(f"approval output already exists: {path}") from error
    except OSError as error:
        raise BenchmarkApprovalError(f"could not write approval output {path}: {error}") from error
