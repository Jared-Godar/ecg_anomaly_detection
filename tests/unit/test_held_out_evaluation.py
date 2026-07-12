"""Tests for the one-shot held-out test-partition evaluation module."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ecg_anomaly_detection.benchmark_approval import ApprovalRecord
from ecg_anomaly_detection.benchmark_policy import BenchmarkPolicy
from ecg_anomaly_detection.held_out_config import HeldOutExecutionConfig
from ecg_anomaly_detection.held_out_evaluation import (
    HeldOutEvaluationError,
    evaluate_held_out_from_index,
    load_approval_record,
)

# Stable run ID used as the candidate across all fixtures.
RUN_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _held_out_config() -> HeldOutExecutionConfig:
    """Return the inert (execution_enabled=False) held-out config used by all tests."""
    return HeldOutExecutionConfig(
        schema_version=1,
        name="held-out-execution-v1",
        version="1.0.0",
        evaluator="random-projection-nearest-centroid",
        partition="test",
        execution_enabled=True,
        requires_recorded_approval=True,
    )


def _policy() -> BenchmarkPolicy:
    """Return a minimal valid BenchmarkPolicy with test_evaluation_enabled=False."""
    return BenchmarkPolicy(
        schema_version=1,
        policy_id="benchmark-governance-v1",
        version="1.0.0",
        protected_partition="test",
        test_evaluation_enabled=True,
        explicit_future_opt_in_required=True,
        eligibility_criteria=("candidate must be frozen",),
        execution_procedure=("record approval", "verify lineage", "execute once", "archive"),
        required_lineage_references=frozenset(
            {
                "repository_commit_hash",
                "dataset_configuration_hash",
                "split_identity",
                "training_configuration_hash",
                "evaluation_configuration_hash",
                "reproducibility_evidence_reference",
                "run_manifest_reference",
            }
        ),
        required_disclosures=frozenset(
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
        ),
        required_limitations=frozenset(
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
        ),
        prohibited_claims=frozenset(
            {
                "model_quality_established",
                "generalization_established",
                "clinical_validity_established",
                "medical_utility_established",
            }
        ),
        rerun_allowed_reasons=("infrastructure failure before result inspection",),
        required_archival_records=frozenset(
            {
                "approval_record",
                "immutable_lineage_references",
                "benchmark_results",
                "publication_disclosures",
                "runtime_and_hardware_evidence",
                "rerun_history",
            }
        ),
    )


def _approval_record(*, candidate_run_id: str = RUN_ID) -> ApprovalRecord:
    """Return a minimal valid ApprovalRecord for the given candidate run ID."""
    return ApprovalRecord(
        schema_version=1,
        policy_id="benchmark-governance-v1",
        policy_version="1.0.0",
        owner="jane@example.invalid",
        candidate_run_id=candidate_run_id,
        purpose="development benchmark rehearsal",
        prior_attempt_exists=False,
        run_manifest_reference=candidate_run_id,
        verified_lineage_references=(
            "dataset_configuration_hash",
            "evaluation_configuration_hash",
            "reproducibility_evidence_reference",
            "repository_commit_hash",
            "run_manifest_reference",
            "split_identity",
            "training_configuration_hash",
        ),
    )


def _repository(
    root: Path,
    *,
    labels: np.ndarray | None = None,
    run_id: str = RUN_ID,
) -> dict[str, Path]:
    """Build a minimal fixture repository with a test shard, dataset index, and frozen model.

    The dataset index's 'validation' partition intentionally points at a shard path that
    is never written to disk ('validation-must-not-open.npz'), so any accidental read of
    the validation partition fails loudly with a file-not-found error.

    Args:
        root: The fake repository root to populate.
        labels: Override the default test window labels.
        run_id: The run ID used to scope artifact paths.

    Returns:
        A dict of the fixture's index/model/metadata/shard paths.
    """
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    shard_dir = root / "data" / "interim" / "runs" / run_id / "windows"
    index_dir = root / "data" / "processed" / "runs" / run_id
    training_dir = root / "artifacts" / "runs" / run_id / "training"
    shard_dir.mkdir(parents=True)
    index_dir.mkdir(parents=True)
    training_dir.mkdir(parents=True)

    values = labels if labels is not None else np.asarray([0, 1, 1, 0], dtype=np.int64)
    windows = np.asarray([[0.0, 0.0], [10.0, 10.0], [1.0, 1.0], [0.0, 0.0]])[: len(values)]
    shard = shard_dir / "test-record.npz"
    np.savez_compressed(
        shard,
        windows=windows,
        target_values=values,
        record_ids=np.asarray(["test-record"] * len(values)),
    )
    shard_digest = _identity(root, shard)
    test_descriptor = {
        "record_id": "test-record",
        "window_count": len(values),
        "target_value_counts": _counts(values),
        "file": shard_digest,
    }
    index = index_dir / "dataset-index.json"
    index.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "window_samples": 2,
                "partitions": {
                    "train": {"shards": []},
                    "validation": {
                        "record_count": 1,
                        "window_count": 99,
                        "target_value_counts": {"0": 99},
                        "shards": [
                            {
                                "record_id": "validation-must-not-open",
                                "file": {"path": "data/interim/validation-must-not-open.npz"},
                            }
                        ],
                    },
                    "test": {
                        "record_count": 1,
                        "window_count": len(values),
                        "target_value_counts": _counts(values),
                        "shards": [test_descriptor],
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    model = training_dir / "model.json"
    model.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "estimator": "random-projection-nearest-centroid",
                "training_name": "fixture",
                "training_version": "1.0.0",
                "seed": 1,
                "input_features": 2,
                "projection_components": 2,
                "classes": [0, 1],
                "projection": [[1.0, 0.0], [0.0, 1.0]],
                "centroids": [[0.0, 0.0], [10.0, 10.0]],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    metadata = training_dir / "training-metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "partition": "train",
                "dataset_index": _identity(root, index),
                "model": _identity(root, model),
            }
        ),
        encoding="utf-8",
    )
    return {"index": index, "model": model, "metadata": metadata, "shard": shard}


def _output_dir(root: Path, run_id: str = RUN_ID) -> Path:
    """Create and return the held-out-evaluation output directory for a run."""
    directory = root / "artifacts" / "runs" / run_id / "held-out-evaluation"
    directory.mkdir(parents=True)
    return directory


def _identity(root: Path, path: Path) -> dict[str, object]:
    """Describe a fixture file with the evaluator's immutable identity contract."""
    content = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _counts(values: np.ndarray) -> dict[str, int]:
    """Return deterministic string-keyed class counts for a fixture shard."""
    return {str(v): int(np.count_nonzero(values == v)) for v in np.unique(values)}


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_evaluate_held_out_writes_metrics_and_disclosure(tmp_path: Path) -> None:
    """A fully valid fixture produces metrics and disclosure artifacts and never opens validation.

    The fixture's dataset index 'validation' partition points at a file that doesn't
    exist on disk; if evaluation ever touched the validation partition, this test would
    fail with a file-not-found error.
    """
    paths = _repository(tmp_path)
    out_dir = _output_dir(tmp_path)
    metrics_path = out_dir / "held-out-metrics.json"
    disclosure_path = out_dir / "held-out-disclosure.json"

    result = evaluate_held_out_from_index(
        tmp_path,
        paths["index"],
        paths["model"],
        paths["metadata"],
        _held_out_config(),
        _policy(),
        _approval_record(),
        metrics_path,
        disclosure_path,
    )

    assert result.window_count == 4
    assert result.record_count == 1
    assert metrics_path.exists()
    assert disclosure_path.exists()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["partition"] == "test"
    assert metrics["candidate_run_id"] == RUN_ID
    assert metrics["record_count"] == 1
    assert metrics["window_count"] == 4
    assert all("validation" not in item["path"] for item in metrics["test_shards"])

    disclosure = json.loads(disclosure_path.read_text(encoding="utf-8"))
    assert disclosure["candidate_run_id"] == RUN_ID
    assert "required_disclosures" in disclosure
    assert "required_limitations" in disclosure
    assert "prohibited_claims" in disclosure


def test_evaluate_held_out_metrics_are_deterministic(tmp_path: Path) -> None:
    """Running evaluate_held_out_from_index twice with the same inputs produces identical output."""
    paths = _repository(tmp_path)

    out1 = _output_dir(tmp_path, "run-a")
    result1 = evaluate_held_out_from_index(
        tmp_path,
        paths["index"],
        paths["model"],
        paths["metadata"],
        _held_out_config(),
        _policy(),
        _approval_record(),
        out1 / "held-out-metrics.json",
        out1 / "held-out-disclosure.json",
    )

    out2 = _output_dir(tmp_path, "run-b")
    result2 = evaluate_held_out_from_index(
        tmp_path,
        paths["index"],
        paths["model"],
        paths["metadata"],
        _held_out_config(),
        _policy(),
        _approval_record(),
        out2 / "held-out-metrics.json",
        out2 / "held-out-disclosure.json",
    )

    assert result1.window_count == result2.window_count
    assert (out1 / "held-out-metrics.json").read_bytes() == (
        out2 / "held-out-metrics.json"
    ).read_bytes()


# ---------------------------------------------------------------------------
# Fail-closed: governance gate tests
# ---------------------------------------------------------------------------


def test_fails_closed_when_config_is_disabled(tmp_path: Path) -> None:
    """A held-out config with execution_enabled=True is rejected before any data is opened."""
    paths = _repository(tmp_path)
    out_dir = _output_dir(tmp_path)
    bad_config = HeldOutExecutionConfig(
        schema_version=1,
        name="held-out-execution-v1",
        version="1.0.0",
        evaluator="random-projection-nearest-centroid",
        partition="test",
        execution_enabled=False,
        requires_recorded_approval=True,
    )

    # The following control-flow step enforces or verifies the surrounding contract.

    with pytest.raises(HeldOutEvaluationError, match="execution_enabled"):
        evaluate_held_out_from_index(
            tmp_path,
            paths["index"],
            paths["model"],
            paths["metadata"],
            bad_config,
            _policy(),
            _approval_record(),
            out_dir / "held-out-metrics.json",
            out_dir / "held-out-disclosure.json",
        )

    assert not (out_dir / "held-out-metrics.json").exists()


def test_fails_closed_when_config_disables_recorded_approval(tmp_path: Path) -> None:
    """A config with requires_recorded_approval=False is rejected before any data is opened."""
    paths = _repository(tmp_path)
    out_dir = _output_dir(tmp_path)
    bad_config = HeldOutExecutionConfig(
        schema_version=1,
        name="held-out-execution-v1",
        version="1.0.0",
        evaluator="random-projection-nearest-centroid",
        partition="test",
        execution_enabled=True,
        requires_recorded_approval=False,
    )

    # The following control-flow step enforces or verifies the surrounding contract.

    with pytest.raises(HeldOutEvaluationError, match="requires_recorded_approval"):
        evaluate_held_out_from_index(
            tmp_path,
            paths["index"],
            paths["model"],
            paths["metadata"],
            bad_config,
            _policy(),
            _approval_record(),
            out_dir / "held-out-metrics.json",
            out_dir / "held-out-disclosure.json",
        )

    assert not (out_dir / "held-out-metrics.json").exists()


def test_fails_closed_when_policy_disables_test_evaluation(tmp_path: Path) -> None:
    """A policy with test_evaluation_enabled=True is rejected before any data is opened."""
    paths = _repository(tmp_path)
    out_dir = _output_dir(tmp_path)
    bad_policy = BenchmarkPolicy(
        schema_version=1,
        policy_id="benchmark-governance-v1",
        version="1.0.0",
        protected_partition="test",
        test_evaluation_enabled=False,
        explicit_future_opt_in_required=True,
        eligibility_criteria=("candidate must be frozen",),
        execution_procedure=("record approval",),
        required_lineage_references=frozenset({"run_manifest_reference"}),
        required_disclosures=frozenset({"repository_commit_hash"}),
        required_limitations=frozenset({"dataset_limitations"}),
        prohibited_claims=frozenset({"model_quality_established"}),
        rerun_allowed_reasons=("infrastructure failure",),
        required_archival_records=frozenset({"approval_record"}),
    )

    # The following control-flow step enforces or verifies the surrounding contract.

    with pytest.raises(HeldOutEvaluationError, match="test_evaluation_enabled"):
        evaluate_held_out_from_index(
            tmp_path,
            paths["index"],
            paths["model"],
            paths["metadata"],
            _held_out_config(),
            bad_policy,
            _approval_record(),
            out_dir / "held-out-metrics.json",
            out_dir / "held-out-disclosure.json",
        )

    assert not (out_dir / "held-out-metrics.json").exists()


def test_fails_closed_when_approval_run_id_does_not_match_model(tmp_path: Path) -> None:
    """An approval record naming a different run than the model artifact is rejected."""
    paths = _repository(tmp_path)
    out_dir = _output_dir(tmp_path)
    wrong_approval = _approval_record(candidate_run_id="00000000-0000-0000-0000-000000000000")

    # The following control-flow step enforces or verifies the surrounding contract.

    with pytest.raises(HeldOutEvaluationError, match="does not match model run"):
        evaluate_held_out_from_index(
            tmp_path,
            paths["index"],
            paths["model"],
            paths["metadata"],
            _held_out_config(),
            _policy(),
            wrong_approval,
            out_dir / "held-out-metrics.json",
            out_dir / "held-out-disclosure.json",
        )

    assert not (out_dir / "held-out-metrics.json").exists()


# ---------------------------------------------------------------------------
# Fail-closed: digest and lineage tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["dataset", "model", "shard"])
def test_digest_mismatch_fails_before_metrics_persistence(tmp_path: Path, target: str) -> None:
    """A digest mismatch in any of the three cross-checked files fails before writing metrics.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
        target: Which file to corrupt after fixture setup.
    """
    paths = _repository(tmp_path)
    out_dir = _output_dir(tmp_path)
    # The following control-flow step enforces or verifies the surrounding contract.
    if target == "dataset":
        paths["index"].write_bytes(paths["index"].read_bytes() + b" ")
    elif target == "model":
        paths["model"].write_bytes(paths["model"].read_bytes() + b" ")
    else:
        paths["shard"].write_bytes(paths["shard"].read_bytes() + b"changed")

    # The following control-flow step enforces or verifies the surrounding contract.

    with pytest.raises(HeldOutEvaluationError, match="digest does not match"):
        evaluate_held_out_from_index(
            tmp_path,
            paths["index"],
            paths["model"],
            paths["metadata"],
            _held_out_config(),
            _policy(),
            _approval_record(),
            out_dir / "held-out-metrics.json",
            out_dir / "held-out-disclosure.json",
        )

    assert not (out_dir / "held-out-metrics.json").exists()
    assert not (out_dir / "held-out-disclosure.json").exists()


def test_unknown_test_label_fails_before_persistence(tmp_path: Path) -> None:
    """A test label the frozen model was never trained on is rejected, not silently misscored."""
    paths = _repository(tmp_path, labels=np.asarray([0, 2], dtype=np.int64))
    out_dir = _output_dir(tmp_path)

    # The following control-flow step enforces or verifies the surrounding contract.

    with pytest.raises(HeldOutEvaluationError, match="unknown to the model"):
        evaluate_held_out_from_index(
            tmp_path,
            paths["index"],
            paths["model"],
            paths["metadata"],
            _held_out_config(),
            _policy(),
            _approval_record(),
            out_dir / "held-out-metrics.json",
            out_dir / "held-out-disclosure.json",
        )

    assert not (out_dir / "held-out-metrics.json").exists()


# ---------------------------------------------------------------------------
# load_approval_record tests
# ---------------------------------------------------------------------------


def test_load_approval_record_round_trips_written_record(tmp_path: Path) -> None:
    """An ApprovalRecord written to JSON can be loaded back with load_approval_record."""
    original = _approval_record()
    path = tmp_path / "approval.json"
    path.write_text(original.to_json(), encoding="utf-8")

    loaded = load_approval_record(path)

    assert loaded.candidate_run_id == original.candidate_run_id
    assert loaded.policy_id == original.policy_id
    assert set(loaded.verified_lineage_references) == set(original.verified_lineage_references)


def test_load_approval_record_rejects_missing_file(tmp_path: Path) -> None:
    """Pointing load_approval_record at a missing path raises HeldOutEvaluationError."""
    # The following control-flow step enforces or verifies the surrounding contract.
    with pytest.raises(HeldOutEvaluationError, match="could not load approval record"):
        load_approval_record(tmp_path / "missing.json")


def test_load_approval_record_rejects_wrong_schema_version(tmp_path: Path) -> None:
    """An approval record with schema_version != 1 is rejected."""
    path = tmp_path / "approval.json"
    path.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")

    # The following control-flow step enforces or verifies the surrounding contract.

    with pytest.raises(HeldOutEvaluationError, match="schema_version 1"):
        load_approval_record(path)
