"""Deterministic validation for the benchmark governance policy."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The fixed, closed set of lineage references a policy's execution.required_lineage_references
# must name, matching every reference kind benchmark_approval.py's
# _missing_lineage_references knows how to verify. A policy that named fewer would leave
# some approval evidence unverified before a future benchmark could run.
REQUIRED_LINEAGE_REFERENCES = frozenset(
    {
        "repository_commit_hash",
        "dataset_configuration_hash",
        "split_identity",
        "training_configuration_hash",
        "evaluation_configuration_hash",
        "reproducibility_evidence_reference",
        "run_manifest_reference",
    }
)
# The fixed, closed set of disclosures a policy's publication.required_disclosures must
# name, per docs/benchmark-governance.md's publication requirements -- these are the
# minimum evidence categories any future benchmark result must be published alongside.
REQUIRED_DISCLOSURES = frozenset(
    {
        "repository_commit_hash",
        "dataset_configuration_hash",
        "split_identity",
        "training_configuration_hash",
        "evaluation_configuration_hash",
        "reproducibility_evidence_reference",
        "hardware_summary",
        "runtime_summary",
        "assumptions",
        "limitations",
    }
)
# The fixed, closed set of limitation categories a policy's publication.required_limitations
# must name, ensuring any future benchmark result is published alongside an explicit
# statement of this project's non-clinical, research-only scope.
REQUIRED_LIMITATIONS = frozenset(
    {
        "dataset_limitations",
        "annotation_limitations",
        "class_imbalance_limitations",
        "binary_mapping_limitations",
        "split_methodology_limitations",
        "historical_dataset_limitations",
        "lack_of_clinical_validation",
        "lack_of_medical_utility",
    }
)
# The fixed, closed set of claims a policy's publication.prohibited_claims must name --
# these are exactly the overclaiming statements this repository's non-clinical framing
# forbids making about any future benchmark result.
REQUIRED_PROHIBITED_CLAIMS = frozenset(
    {
        "model_quality_established",
        "generalization_established",
        "clinical_validity_established",
        "medical_utility_established",
    }
)
# The fixed, closed set of archival record categories a policy's archival.required_records
# must name, ensuring every future benchmark attempt (successful or not) leaves a
# permanent, append-only audit trail rather than an overwritable result.
REQUIRED_ARCHIVAL_RECORDS = frozenset(
    {
        "approval_record",
        "immutable_lineage_references",
        "benchmark_results",
        "publication_disclosures",
        "runtime_and_hardware_evidence",
        "rerun_history",
    }
)


class BenchmarkPolicyError(ValueError):
    """Raised when benchmark governance configuration fails closed."""


@dataclass(frozen=True, slots=True)
class BenchmarkPolicy:
    """Validated controls for future protected-test benchmark activity."""

    schema_version: int
    policy_id: str
    version: str
    protected_partition: str
    test_evaluation_enabled: bool
    explicit_future_opt_in_required: bool
    eligibility_criteria: tuple[str, ...]
    execution_procedure: tuple[str, ...]
    required_lineage_references: frozenset[str]
    required_disclosures: frozenset[str]
    required_limitations: frozenset[str]
    prohibited_claims: frozenset[str]
    rerun_allowed_reasons: tuple[str, ...]
    required_archival_records: frozenset[str]


def load_benchmark_policy(path: Path, *, allow_test_evaluation: bool = False) -> BenchmarkPolicy:
    """Load and validate policy metadata without accessing data or model artifacts."""
    # Translate a missing, unreadable, or malformed-TOML file into BenchmarkPolicyError.
    try:
        # The `with` block ensures the file handle closes even if tomllib.load raises.
        with path.open("rb") as source:
            document = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise BenchmarkPolicyError(f"could not load benchmark policy {path}: {error}") from error

    # schema_version pins this loader's understanding of the policy document's shape.
    if document.get("schema_version") != 1:
        raise BenchmarkPolicyError("benchmark policy must use schema_version = 1")
    benchmark = _table(document, "benchmark")
    eligibility = _table(document, "eligibility")
    execution = _table(document, "execution")
    publication = _table(document, "publication")
    reruns = _table(document, "reruns")
    archival = _table(document, "archival")

    policy = BenchmarkPolicy(
        schema_version=1,
        policy_id=_string(benchmark, "policy_id", "benchmark"),
        version=_string(benchmark, "version", "benchmark"),
        protected_partition=_string(benchmark, "protected_partition", "benchmark"),
        test_evaluation_enabled=_boolean(benchmark, "test_evaluation_enabled", "benchmark"),
        explicit_future_opt_in_required=_boolean(
            benchmark, "explicit_future_opt_in_required", "benchmark"
        ),
        eligibility_criteria=_strings(eligibility, "criteria", "eligibility"),
        execution_procedure=_strings(execution, "procedure", "execution"),
        required_lineage_references=_string_set(
            execution, "required_lineage_references", "execution"
        ),
        required_disclosures=_string_set(publication, "required_disclosures", "publication"),
        required_limitations=_string_set(publication, "required_limitations", "publication"),
        prohibited_claims=_string_set(publication, "prohibited_claims", "publication"),
        rerun_allowed_reasons=_strings(reruns, "allowed_reasons", "reruns"),
        required_archival_records=_string_set(archival, "required_records", "archival"),
    )
    # This module governs exactly one protected partition, named "test"; a policy
    # naming any other value would be describing a different, unsupported gate.
    if policy.protected_partition != "test":
        raise BenchmarkPolicyError("benchmark.protected_partition must be 'test'")
    # Routine callers retain the disabled-by-default boundary; the separately reviewed
    # evaluator is the sole caller allowed to load an explicitly enabled policy copy.
    if policy.test_evaluation_enabled and not allow_test_evaluation:
        raise BenchmarkPolicyError("benchmark.test_evaluation_enabled must be false")
    # Symmetric to the check above: a future benchmark run must require deliberate,
    # explicit opt-in rather than running automatically once other conditions are met.
    if not policy.explicit_future_opt_in_required:
        raise BenchmarkPolicyError("benchmark.explicit_future_opt_in_required must be true")
    _require_set(
        policy.required_lineage_references,
        REQUIRED_LINEAGE_REFERENCES,
        "execution.required_lineage_references",
    )
    _require_set(
        policy.required_disclosures,
        REQUIRED_DISCLOSURES,
        "publication.required_disclosures",
    )
    _require_set(
        policy.required_limitations,
        REQUIRED_LIMITATIONS,
        "publication.required_limitations",
    )
    _require_set(
        policy.prohibited_claims,
        REQUIRED_PROHIBITED_CLAIMS,
        "publication.prohibited_claims",
    )
    _require_set(
        policy.required_archival_records,
        REQUIRED_ARCHIVAL_RECORDS,
        "archival.required_records",
    )
    # These boolean governance flags (rerun safeguards, archival immutability) must all
    # be true; loop over both tables' key lists together since they share the same
    # "every listed flag must be true" validation rather than duplicating the check
    # once per table.
    for table, keys in (
        (
            reruns,
            (
                "require_new_approval",
                "require_prior_result_retention",
                "prohibit_model_selection_after_result",
            ),
        ),
        (archival, ("retain_superseded_results", "append_only_history")),
    ):
        # Check every flag in this table's key list.
        for key in keys:
            # `table is reruns` distinguishes which table's name to report in the
            # error, since both tables are validated by this same shared loop.
            if not _boolean(table, key, "reruns" if table is reruns else "archival"):
                raise BenchmarkPolicyError(f"{key} must be true")
    return policy


def _table(document: dict[str, Any], key: str) -> dict[str, Any]:
    """Require and return one top-level TOML table from the parsed policy document.

    Args:
        document: The parsed policy document to read from.
        key: The top-level table name to extract (e.g. "benchmark", "execution").

    Returns:
        The requested table.
    """

    value = document.get(key)
    # Every field access downstream assumes value is a dict.
    if not isinstance(value, dict):
        raise BenchmarkPolicyError(f"benchmark policy requires a [{key}] table")
    return value


def _string(values: dict[str, Any], key: str, table: str) -> str:
    """Require and return a non-empty string from the requested structured field.

    Args:
        values: The parsed table to read from.
        key: The field name to extract.
        table: The table name, used in the error message's dotted-path prefix.

    Returns:
        The field's value with surrounding whitespace stripped.
    """

    value = values.get(key)
    # Reject a missing/wrong-typed value and a whitespace-only placeholder alike.
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkPolicyError(f"{table}.{key} must be a non-empty string")
    return value.strip()


def _boolean(values: dict[str, Any], key: str, table: str) -> bool:
    """Require and return a strict boolean from the requested structured field.

    "Strict" here means TOML's actual boolean type, not a truthy value like the
    string "true" or the integer 1 -- governance flags shouldn't be ambiguous about
    what counts as enabled.

    Args:
        values: The parsed table to read from.
        key: The field name to extract.
        table: The table name, used in the error message's dotted-path prefix.

    Returns:
        The field's boolean value.
    """

    value = values.get(key)
    # bool is checked with isinstance, not truthiness, so "true"/1/etc. are rejected.
    if not isinstance(value, bool):
        raise BenchmarkPolicyError(f"{table}.{key} must be a boolean")
    return value


def _strings(values: dict[str, Any], key: str, table: str) -> tuple[str, ...]:
    """Require and return a non-empty array of unique, non-empty, stripped strings.

    Args:
        values: The parsed table to read from.
        key: The field name to extract.
        table: The table name, used in the error message's dotted-path prefix.

    Returns:
        The validated, stripped values.
    """

    value = values.get(key)
    # Reject a missing/empty list or any non-string element before normalization below.
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise BenchmarkPolicyError(f"{table}.{key} must be a non-empty string array")
    normalized = tuple(item.strip() for item in value)
    # Stripping whitespace could reduce a previously-distinct entry to empty or to a
    # duplicate of another entry; re-check both after normalization.
    if any(not item for item in normalized) or len(normalized) != len(set(normalized)):
        raise BenchmarkPolicyError(f"{table}.{key} must contain unique, non-empty strings")
    return normalized


def _string_set(values: dict[str, Any], key: str, table: str) -> frozenset[str]:
    """Require and return an unordered set of unique, non-empty strings.

    A thin wrapper over _strings for the four fields (required_lineage_references,
    required_disclosures, required_limitations, prohibited_claims,
    required_archival_records) whose validation only cares about set membership, not
    the TOML array's declared order.

    Args:
        values: The parsed table to read from.
        key: The field name to extract.
        table: The table name, used in the error message's dotted-path prefix.

    Returns:
        The validated values as a frozenset.
    """

    return frozenset(_strings(values, key, table))


def _require_set(actual: frozenset[str], required: frozenset[str], field: str) -> None:
    """Require a configuration value to equal the complete expected string set.

    Used for every REQUIRED_* module-level constant: a policy author can't silently
    omit an entry from one of these closed sets, since every required entry is checked
    present.

    Args:
        actual: The set of values the policy document actually declared.
        required: The complete, fixed set this field is required to contain.
        field: The dotted-path field name, used in the error message.
    """

    missing = sorted(required - actual)
    # A policy declaring exactly the required set (or a superset) passes; anything
    # missing means that specific governance requirement isn't actually enforced.
    if missing:
        raise BenchmarkPolicyError(f"{field} is missing required values: {missing}")
