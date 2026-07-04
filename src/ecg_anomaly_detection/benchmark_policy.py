"""Deterministic validation for the benchmark governance policy."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
REQUIRED_PROHIBITED_CLAIMS = frozenset(
    {
        "model_quality_established",
        "generalization_established",
        "clinical_validity_established",
        "medical_utility_established",
    }
)
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


def load_benchmark_policy(path: Path) -> BenchmarkPolicy:
    """Load and validate policy metadata without accessing data or model artifacts."""
    try:
        with path.open("rb") as source:
            document = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise BenchmarkPolicyError(f"could not load benchmark policy {path}: {error}") from error

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
    if policy.protected_partition != "test":
        raise BenchmarkPolicyError("benchmark.protected_partition must be 'test'")
    if policy.test_evaluation_enabled:
        raise BenchmarkPolicyError("benchmark.test_evaluation_enabled must be false")
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
        for key in keys:
            if not _boolean(table, key, "reruns" if table is reruns else "archival"):
                raise BenchmarkPolicyError(f"{key} must be true")
    return policy


def _table(document: dict[str, Any], key: str) -> dict[str, Any]:
    value = document.get(key)
    if not isinstance(value, dict):
        raise BenchmarkPolicyError(f"benchmark policy requires a [{key}] table")
    return value


def _string(values: dict[str, Any], key: str, table: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkPolicyError(f"{table}.{key} must be a non-empty string")
    return value.strip()


def _boolean(values: dict[str, Any], key: str, table: str) -> bool:
    value = values.get(key)
    if not isinstance(value, bool):
        raise BenchmarkPolicyError(f"{table}.{key} must be a boolean")
    return value


def _strings(values: dict[str, Any], key: str, table: str) -> tuple[str, ...]:
    value = values.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise BenchmarkPolicyError(f"{table}.{key} must be a non-empty string array")
    normalized = tuple(item.strip() for item in value)
    if any(not item for item in normalized) or len(normalized) != len(set(normalized)):
        raise BenchmarkPolicyError(f"{table}.{key} must contain unique, non-empty strings")
    return normalized


def _string_set(values: dict[str, Any], key: str, table: str) -> frozenset[str]:
    return frozenset(_strings(values, key, table))


def _require_set(actual: frozenset[str], required: frozenset[str], field: str) -> None:
    missing = sorted(required - actual)
    if missing:
        raise BenchmarkPolicyError(f"{field} is missing required values: {missing}")
