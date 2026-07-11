"""Integration test for the record-benchmark-approval CLI boundary."""

import json
from pathlib import Path

import pytest

from ecg_anomaly_detection.cli import main
from ecg_anomaly_detection.run_manifest import (
    DatasetEvidence,
    EnvironmentSnapshot,
    FileEvidence,
    GitState,
    PartitionEvidence,
    RunManifest,
    SplitEvidence,
)

# Centralize RUN_ID so every caller shares the same documented invariant.
RUN_ID = "12345678-1234-5678-1234-567812345678"


def _policy_text() -> str:
    """Read the repository's real, committed configs/benchmark-policy-v1.toml as text.

    Returns:
        The committed benchmark policy file's full text.
    """

    root = Path(__file__).parents[2]
    return (root / "configs" / "benchmark-policy-v1.toml").read_text(encoding="utf-8")


def _manifest_json() -> str:
    """A complete, minimal RunManifest for RUN_ID, serialized to JSON.

    Every RunManifest field is populated with a small but structurally valid
    value, since record-benchmark-approval reads and cross-checks the whole
    manifest (not just run_id) against the policy and approval file.

    Returns:
        The manifest's JSON serialization.
    """

    manifest = RunManifest(
        schema_version=1,
        run_id=RUN_ID,
        created_at_utc="2026-01-02T03:04:05Z",
        git=GitState(revision="1" * 40, dirty=False),
        environment=EnvironmentSnapshot(
            python_version="3.12.13",
            python_implementation="CPython",
            platform="darwin",
            machine="arm64",
            installed_packages={"numpy": "2.4.2"},
        ),
        dependency_lock=FileEvidence("uv.lock", 10, "a" * 64),
        dataset=DatasetEvidence(
            dataset_slug="synthetic",
            dataset_version="1.0.0",
            inventory_created_at_utc="2026-01-01T00:00:00Z",
            file_count=1,
            total_size_bytes=10,
            source_files=(FileEvidence("100.dat", 10, "b" * 64),),
            inventory_manifest=FileEvidence("artifacts/inventory.json", 20, "c" * 64),
        ),
        split=SplitEvidence(
            split_name="grouped",
            split_version="1.0.0",
            strategy="seeded-record-shuffle",
            seed=7,
            mapping_name="binary",
            mapping_version="1.0.0",
            window_config_name="six-second",
            window_config_version="1.0.0",
            total_subject_count=1,
            total_record_count=1,
            total_window_count=2,
            partitions={
                "train": PartitionEvidence(
                    subject_ids=("100",),
                    subject_count=1,
                    record_ids=("100",),
                    record_subjects={"100": "100"},
                    record_count=1,
                    window_count=2,
                    target_value_counts={"0": 1, "1": 1},
                )
            },
            split_manifest=FileEvidence("artifacts/split.json", 30, "d" * 64),
        ),
        configuration_files=(
            FileEvidence("configs/dataset.toml", 1, "f" * 64),
            FileEvidence("configs/training.toml", 1, "g" * 64),
            FileEvidence("configs/evaluation.toml", 1, "h" * 64),
        ),
        evidence_files=(FileEvidence("artifacts/runs/x/environment_summary.json", 2, "e" * 64),),
        artifact_files=(),
    )
    return manifest.to_json()


def _approval_text() -> str:
    """An approval TOML document whose candidate_run_id matches _manifest_json's RUN_ID.

    Returns:
        The approval file's full text.
    """

    return (
        "schema_version = 1\n\n"
        "[approval]\n"
        'owner = "jane@example.invalid"\n'
        f'candidate_run_id = "{RUN_ID}"\n'
        'purpose = "development benchmark rehearsal"\n'
        "prior_attempt_exists = false\n\n"
        "[approval.lineage_configuration_paths]\n"
        'dataset_configuration_hash = "configs/dataset.toml"\n'
        'training_configuration_hash = "configs/training.toml"\n'
        'evaluation_configuration_hash = "configs/evaluation.toml"\n'
    )


def test_validate_benchmark_policy_command_reports_policy_identity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`ecg-data validate-benchmark-policy` validates the repository's real, committed benchmark
    policy and prints its policy ID and version alongside its progress banners (#61).

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
        capsys: Used to capture stdout and confirm both the existing
            completion message and the new progress banners.
    """

    policy_path = tmp_path / "policy.toml"
    policy_path.write_text(_policy_text(), encoding="utf-8")

    exit_code = main(["validate-benchmark-policy", "--policy", str(policy_path)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "validated benchmark policy benchmark-governance-v1" in output
    assert "[1/1] validate-benchmark-policy: starting" in output
    assert "[1/1] validate-benchmark-policy: complete in" in output


def test_record_benchmark_approval_command_records_approval_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`ecg-data record-benchmark-approval` with a matching manifest and approval writes an
    approval record carrying the run ID and the policy's own ID.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
        capsys: Used to capture stdout and confirm the command's progress
            banners (#61).
    """

    (tmp_path / "artifacts").mkdir()
    policy_path = tmp_path / "policy.toml"
    policy_path.write_text(_policy_text(), encoding="utf-8")
    manifest_path = tmp_path / "run-manifest.json"
    manifest_path.write_text(_manifest_json(), encoding="utf-8")
    approval_path = tmp_path / "approval.toml"
    approval_path.write_text(_approval_text(), encoding="utf-8")
    output_path = tmp_path / "artifacts" / "benchmark_approval.json"

    exit_code = main(
        [
            "record-benchmark-approval",
            "--repository-root",
            str(tmp_path),
            "--policy",
            str(policy_path),
            "--run-manifest",
            str(manifest_path),
            "--approval",
            str(approval_path),
            "--output",
            str(output_path),
        ]
    )

    captured_output = capsys.readouterr().out
    assert exit_code == 0
    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["candidate_run_id"] == RUN_ID
    assert record["run_manifest_reference"] == RUN_ID
    assert record["policy_id"] == "benchmark-governance-v1"
    assert "[1/1] record-benchmark-approval: starting" in captured_output
    assert "[1/1] record-benchmark-approval: complete in" in captured_output


def test_record_benchmark_approval_command_fails_closed_on_candidate_mismatch(
    tmp_path: Path,
) -> None:
    """An approval file whose candidate_run_id doesn't match the run manifest's run_id is
    rejected, and no approval record is written.

    This is the direct enforcement point of "an approval can only ratify the
    specific run it names" -- a mismatched ID would let one approval be
    reused (accidentally or otherwise) to certify a different run.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    (tmp_path / "artifacts").mkdir()
    policy_path = tmp_path / "policy.toml"
    policy_path.write_text(_policy_text(), encoding="utf-8")
    manifest_path = tmp_path / "run-manifest.json"
    manifest_path.write_text(_manifest_json(), encoding="utf-8")
    approval_path = tmp_path / "approval.toml"
    approval_path.write_text(
        _approval_text().replace(RUN_ID, "87654321-4321-8765-4321-876543218765", 1),
        encoding="utf-8",
    )
    output_path = tmp_path / "artifacts" / "benchmark_approval.json"

    exit_code = main(
        [
            "record-benchmark-approval",
            "--repository-root",
            str(tmp_path),
            "--policy",
            str(policy_path),
            "--run-manifest",
            str(manifest_path),
            "--approval",
            str(approval_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 1
    assert not output_path.exists()
