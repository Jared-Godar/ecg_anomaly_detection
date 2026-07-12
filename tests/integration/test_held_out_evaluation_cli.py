"""Integration tests for the evaluate-held-out CLI subcommand."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ecg_anomaly_detection.benchmark_approval import ApprovalRecord
from ecg_anomaly_detection.cli import main

# Stable run ID matching the fixture model artifact path.
RUN_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

# Minimal explicitly enabled held-out config used only by the reviewed execution command.
_HELD_OUT_CONFIG = (
    "schema_version = 1\n\n"
    "[execution]\n"
    'name = "held-out-execution-v1"\n'
    'version = "1.0.0"\n'
    'evaluator = "random-projection-nearest-centroid"\n'
    'partition = "test"\n'
    "execution_enabled = true\n"
    "requires_recorded_approval = true\n"
)


def _policy_text() -> str:
    """Return an explicitly enabled copy of the committed policy for execution tests."""
    root = Path(__file__).parents[2]
    policy = (root / "configs" / "benchmark-policy-v1.toml").read_text(encoding="utf-8")
    return policy.replace("test_evaluation_enabled = false", "test_evaluation_enabled = true")


def _approval_record(*, candidate_run_id: str = RUN_ID) -> ApprovalRecord:
    """Build verified approval evidence for the synthetic candidate fixture."""
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


def _repository(root: Path, *, run_id: str = RUN_ID) -> dict[str, Path]:
    """Build a minimal fixture repository with a test shard, dataset index, and frozen model."""
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    shard_dir = root / "data" / "interim" / "runs" / run_id / "windows"
    index_dir = root / "data" / "processed" / "runs" / run_id
    training_dir = root / "artifacts" / "runs" / run_id / "training"
    shard_dir.mkdir(parents=True)
    index_dir.mkdir(parents=True)
    training_dir.mkdir(parents=True)

    values = np.asarray([0, 1, 1, 0], dtype=np.int64)
    windows = np.asarray([[0.0, 0.0], [10.0, 10.0], [1.0, 1.0], [0.0, 0.0]])
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


def _write(path: Path, content: str) -> Path:
    """Persist one synthetic governance input and return its path."""
    path.write_text(content, encoding="utf-8")
    return path


def _identity(root: Path, path: Path) -> dict[str, object]:
    """Describe a fixture file with the repository's immutable identity contract."""
    content = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _counts(values: np.ndarray) -> dict[str, int]:
    """Return deterministic string-keyed class counts for a fixture shard."""
    return {str(v): int(np.count_nonzero(values == v)) for v in np.unique(values)}


def _run_command(
    tmp_path: Path,
    paths: dict[str, Path],
    config_path: Path,
    policy_path: Path,
    approval_path: Path,
    out_dir: Path,
) -> tuple[int, str]:
    """Invoke the evaluate-held-out CLI subcommand and return (exit_code, stdout)."""
    from pytest import CaptureFixture  # noqa: F401 — used only for type hint below

    metrics_path = out_dir / "held-out-metrics.json"
    disclosure_path = out_dir / "held-out-disclosure.json"
    exit_code = main(
        [
            "evaluate-held-out",
            "--repository-root",
            str(tmp_path),
            "--dataset-index",
            str(paths["index"]),
            "--model",
            str(paths["model"]),
            "--training-metadata",
            str(paths["metadata"]),
            "--held-out-config",
            str(config_path),
            "--policy",
            str(policy_path),
            "--approval-record",
            str(approval_path),
            "--output",
            str(metrics_path),
            "--disclosure",
            str(disclosure_path),
        ]
    )
    return exit_code, str(metrics_path)


# ---------------------------------------------------------------------------
# Happy-path CLI test
# ---------------------------------------------------------------------------


def test_evaluate_held_out_command_writes_metrics_and_disclosure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`ecg-data evaluate-held-out` writes metrics and disclosure, prints progress banners.

    The fixture's validation partition points at a file that doesn't exist on disk;
    if the command ever touched the validation partition, this test would fail with a
    file-not-found error.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
        capsys: Used to capture stdout and confirm the command's progress banners.
    """
    paths = _repository(tmp_path)
    config_path = _write(tmp_path / "held-out.toml", _HELD_OUT_CONFIG)
    policy_path = _write(tmp_path / "policy.toml", _policy_text())
    approval_path = _write(tmp_path / "approval.json", _approval_record().to_json())
    out_dir = tmp_path / "artifacts" / "runs" / RUN_ID / "held-out-evaluation"
    out_dir.mkdir(parents=True)

    exit_code, _ = _run_command(tmp_path, paths, config_path, policy_path, approval_path, out_dir)

    output = capsys.readouterr().out
    assert exit_code == 0
    assert (out_dir / "held-out-metrics.json").exists()
    assert (out_dir / "held-out-disclosure.json").exists()
    metrics = json.loads((out_dir / "held-out-metrics.json").read_text(encoding="utf-8"))
    assert metrics["partition"] == "test"
    assert metrics["candidate_run_id"] == RUN_ID
    assert all("validation" not in item["path"] for item in metrics["test_shards"])
    assert "[1/1] evaluate-held-out: starting" in output
    assert "[1/1] evaluate-held-out: complete in" in output


# ---------------------------------------------------------------------------
# Fail-closed CLI tests
# ---------------------------------------------------------------------------


def test_evaluate_held_out_fails_closed_on_wrong_candidate_run_id(tmp_path: Path) -> None:
    """An approval record naming a different run than the model exits 1 and leaves no output.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """
    paths = _repository(tmp_path)
    config_path = _write(tmp_path / "held-out.toml", _HELD_OUT_CONFIG)
    policy_path = _write(tmp_path / "policy.toml", _policy_text())
    wrong_approval = _approval_record(candidate_run_id="00000000-0000-0000-0000-000000000000")
    approval_path = _write(tmp_path / "approval.json", wrong_approval.to_json())
    out_dir = tmp_path / "artifacts" / "runs" / RUN_ID / "held-out-evaluation"
    out_dir.mkdir(parents=True)

    exit_code, _ = _run_command(tmp_path, paths, config_path, policy_path, approval_path, out_dir)

    assert exit_code == 1
    assert not (out_dir / "held-out-metrics.json").exists()
    assert not (out_dir / "held-out-disclosure.json").exists()


def test_evaluate_held_out_fails_closed_on_corrupted_model(tmp_path: Path) -> None:
    """A corrupted model file makes `evaluate-held-out` exit 1 and leave no output file.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """
    paths = _repository(tmp_path)
    paths["model"].write_bytes(paths["model"].read_bytes() + b" ")
    config_path = _write(tmp_path / "held-out.toml", _HELD_OUT_CONFIG)
    policy_path = _write(tmp_path / "policy.toml", _policy_text())
    approval_path = _write(tmp_path / "approval.json", _approval_record().to_json())
    out_dir = tmp_path / "artifacts" / "runs" / RUN_ID / "held-out-evaluation"
    out_dir.mkdir(parents=True)

    exit_code, _ = _run_command(tmp_path, paths, config_path, policy_path, approval_path, out_dir)

    assert exit_code == 1
    assert not (out_dir / "held-out-metrics.json").exists()


def test_evaluate_held_out_fails_closed_on_disabled_config(tmp_path: Path) -> None:
    """A held-out config with execution_enabled=true is rejected; command exits 1.

    The held_out_config loader itself rejects execution_enabled=true, so this test
    exercises the CLI's error-handling path for HeldOutConfigError.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """
    paths = _repository(tmp_path)
    bad_config = _HELD_OUT_CONFIG.replace("execution_enabled = true", "execution_enabled = false")
    config_path = _write(tmp_path / "held-out.toml", bad_config)
    policy_path = _write(tmp_path / "policy.toml", _policy_text())
    approval_path = _write(tmp_path / "approval.json", _approval_record().to_json())
    out_dir = tmp_path / "artifacts" / "runs" / RUN_ID / "held-out-evaluation"
    out_dir.mkdir(parents=True)

    exit_code, _ = _run_command(tmp_path, paths, config_path, policy_path, approval_path, out_dir)

    assert exit_code == 1
    assert not (out_dir / "held-out-metrics.json").exists()


def test_evaluate_held_out_fails_closed_on_missing_approval_record(tmp_path: Path) -> None:
    """A missing approval record file makes `evaluate-held-out` exit 1.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """
    paths = _repository(tmp_path)
    config_path = _write(tmp_path / "held-out.toml", _HELD_OUT_CONFIG)
    policy_path = _write(tmp_path / "policy.toml", _policy_text())
    out_dir = tmp_path / "artifacts" / "runs" / RUN_ID / "held-out-evaluation"
    out_dir.mkdir(parents=True)
    metrics_path = out_dir / "held-out-metrics.json"
    disclosure_path = out_dir / "held-out-disclosure.json"

    exit_code = main(
        [
            "evaluate-held-out",
            "--repository-root",
            str(tmp_path),
            "--dataset-index",
            str(paths["index"]),
            "--model",
            str(paths["model"]),
            "--training-metadata",
            str(paths["metadata"]),
            "--held-out-config",
            str(config_path),
            "--policy",
            str(policy_path),
            "--approval-record",
            str(tmp_path / "does-not-exist.json"),
            "--output",
            str(metrics_path),
            "--disclosure",
            str(disclosure_path),
        ]
    )

    assert exit_code == 1
    assert not metrics_path.exists()
