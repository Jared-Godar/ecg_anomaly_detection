"""Tests for the benchmark approval-and-lineage gate."""

import json
from pathlib import Path
from typing import Any

import pytest

from ecg_anomaly_detection.benchmark_approval import (
    BenchmarkApprovalError,
    record_benchmark_approval,
)
from ecg_anomaly_detection.run_manifest import (
    UNKNOWN_GIT_REVISION,
    DatasetEvidence,
    EnvironmentSnapshot,
    FileEvidence,
    GitState,
    PartitionEvidence,
    RunManifest,
    SplitEvidence,
)

# A well-formed 40-char-hex-looking UUID used as the "real" candidate run's ID across
# these tests; matches _manifest()'s default run_id so the happy-path tests align by default.
RUN_ID = "12345678-1234-5678-1234-567812345678"
# A second, distinct run ID used only by the run_manifest_reference mismatch test, to
# simulate an approval input naming a different run than the manifest actually describes.
OTHER_RUN_ID = "87654321-4321-8765-4321-876543218765"
# Fake but consistently reused config paths, shared between _manifest's
# configuration_files and _approval_text's lineage_configuration_paths so the two
# agree by default, matching what _configuration_hash_present requires.
DATASET_CONFIG_PATH = "configs/dataset.toml"
# See DATASET_CONFIG_PATH's comment above; same reasoning applies to this path.
TRAINING_CONFIG_PATH = "configs/training.toml"
# See DATASET_CONFIG_PATH's comment above; same reasoning applies to this path.
EVALUATION_CONFIG_PATH = "configs/evaluation.toml"
# Mirrors benchmark_policy.py's REQUIRED_LINEAGE_REFERENCES; kept as a separate local
# constant (not imported) so a test comparing against it would catch an accidental
# drift between the two, rather than the comparison trivially matching itself.
ALL_REFERENCES = frozenset(
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


def _policy_text() -> str:
    """Read the repository's real, committed benchmark-policy TOML as a test fixture.

    Using the real config (rather than a synthetic one) means these tests exercise
    load_benchmark_policy against the actual governance document this repository ships.

    Returns:
        The raw TOML content of configs/benchmark-policy-v1.toml.
    """

    root = Path(__file__).parents[2]
    return (root / "configs" / "benchmark-policy-v1.toml").read_text(encoding="utf-8")


def _manifest(
    *,
    run_id: str = RUN_ID,
    git_revision: str = "1" * 40,
    split_name: str = "grouped",
    evidence_files: tuple[FileEvidence, ...] = (
        FileEvidence("artifacts/runs/x/environment_summary.json", 2, "e" * 64),
    ),
    configuration_files: tuple[FileEvidence, ...] = (
        FileEvidence(DATASET_CONFIG_PATH, 1, "f" * 64),
        FileEvidence(TRAINING_CONFIG_PATH, 1, "g" * 64),
        FileEvidence(EVALUATION_CONFIG_PATH, 1, "h" * 64),
    ),
) -> RunManifest:
    """Build a complete, valid RunManifest, with every field overridable per-test.

    Every keyword defaults to a value that makes record_benchmark_approval succeed;
    individual tests override exactly the field(s) needed to trigger one specific
    missing-lineage-reference failure, keeping each test's intent focused.

    Args:
        run_id: The manifest's own run_id.
        git_revision: The manifest's git.revision; override with a malformed value or
            UNKNOWN_GIT_REVISION to test the repository_commit_hash lineage check.
        split_name: The manifest's split.split_name; override with "" to test the
            split_identity lineage check.
        evidence_files: The manifest's evidence_files; override with () to test the
            reproducibility_evidence_reference lineage check.
        configuration_files: The manifest's configuration_files.

    Returns:
        A fully populated RunManifest.
    """

    return RunManifest(
        schema_version=1,
        run_id=run_id,
        created_at_utc="2026-01-02T03:04:05Z",
        git=GitState(revision=git_revision, dirty=False),
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
            split_name=split_name,
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
        configuration_files=configuration_files,
        evidence_files=evidence_files,
        artifact_files=(),
    )


def _approval_text(
    *,
    owner: str = "jane@example.invalid",
    candidate_run_id: str = RUN_ID,
    purpose: str = "development benchmark rehearsal",
    prior_attempt_exists: str = "false",
    lineage_configuration_paths: dict[str, str] | None = None,
) -> str:
    """Build a valid `[approval]` TOML document, with every field overridable per-test.

    Args:
        owner: The approval.owner field.
        candidate_run_id: The approval.candidate_run_id field; override to mismatch
            _manifest()'s run_id and test the run_manifest_reference lineage check.
        purpose: The approval.purpose field.
        prior_attempt_exists: The approval.prior_attempt_exists field, as a literal
            TOML boolean token (not a Python bool), since this builds raw TOML text.
        lineage_configuration_paths: Override to point a reference at a nonexistent
            path (testing that config's hash-presence check) or omit an entry entirely.

    Returns:
        The raw TOML content for one `[approval]` document.
    """

    paths = (
        {
            "dataset_configuration_hash": DATASET_CONFIG_PATH,
            "training_configuration_hash": TRAINING_CONFIG_PATH,
            "evaluation_configuration_hash": EVALUATION_CONFIG_PATH,
        }
        if lineage_configuration_paths is None
        else lineage_configuration_paths
    )
    lineage_lines = "\n".join(f'{key} = "{value}"' for key, value in paths.items())
    return (
        "schema_version = 1\n\n"
        "[approval]\n"
        f'owner = "{owner}"\n'
        f'candidate_run_id = "{candidate_run_id}"\n'
        f'purpose = "{purpose}"\n'
        f"prior_attempt_exists = {prior_attempt_exists}\n\n"
        "[approval.lineage_configuration_paths]\n"
        f"{lineage_lines}\n"
    )


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """A fake repository root with an empty artifacts/ directory for approval output.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.

    Returns:
        The fake repository root path.
    """

    (tmp_path / "artifacts").mkdir()
    return tmp_path


def _write(path: Path, content: str) -> Path:
    """Write text content to a path and return that same path, for compact test setup.

    Args:
        path: Destination path to write to.
        content: Text content to write.

    Returns:
        The same `path`, unchanged, so callers can inline this in an assignment.
    """

    path.write_text(content, encoding="utf-8")
    return path


def test_record_benchmark_approval_succeeds_and_writes_evidence(repository: Path) -> None:
    """A fully valid policy/manifest/approval trio succeeds and writes a complete record.

    Confirms the happy path end to end: every one of the 7 required lineage
    references is verified present (matching ALL_REFERENCES exactly), and the written
    JSON output reflects the approval input's own fields.
    """

    policy_path = _write(repository / "policy.toml", _policy_text())
    manifest_path = _write(repository / "run-manifest.json", _manifest().to_json())
    approval_path = _write(repository / "approval.toml", _approval_text())
    output_path = repository / "artifacts" / "benchmark_approval.json"

    record = record_benchmark_approval(
        repository, policy_path, manifest_path, approval_path, output_path
    )

    assert record.candidate_run_id == RUN_ID
    assert record.run_manifest_reference == RUN_ID
    assert record.policy_id == "benchmark-governance-v1"
    assert set(record.verified_lineage_references) == ALL_REFERENCES
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["owner"] == "jane@example.invalid"
    assert written["prior_attempt_exists"] is False


def test_record_benchmark_approval_rejects_missing_owner(repository: Path) -> None:
    """An approval input missing the required `owner` field is rejected before verification.

    Protects load_approval_input's own field validation, independent of the lineage
    checks: a structurally incomplete approval document must fail before
    record_benchmark_approval even reaches policy/manifest cross-checking.
    """

    policy_path = _write(repository / "policy.toml", _policy_text())
    manifest_path = _write(repository / "run-manifest.json", _manifest().to_json())
    approval_path = _write(
        repository / "approval.toml",
        _approval_text().replace('owner = "jane@example.invalid"\n', ""),
    )
    output_path = repository / "artifacts" / "benchmark_approval.json"

    # The owner line was stripped from the approval TOML above.
    with pytest.raises(BenchmarkApprovalError, match="approval.owner"):
        record_benchmark_approval(
            repository, policy_path, manifest_path, approval_path, output_path
        )


def test_record_benchmark_approval_rejects_disabled_policy(repository: Path) -> None:
    """A policy with test_evaluation_enabled flipped to true is rejected outright.

    Protects the core governance invariant enforced by load_benchmark_policy: no
    policy may enable test-partition evaluation, regardless of how well-formed
    everything else about it is.
    """

    policy_path = _write(
        repository / "policy.toml",
        _policy_text().replace("test_evaluation_enabled = false", "test_evaluation_enabled = true"),
    )
    manifest_path = _write(repository / "run-manifest.json", _manifest().to_json())
    approval_path = _write(repository / "approval.toml", _approval_text())
    output_path = repository / "artifacts" / "benchmark_approval.json"

    # test_evaluation_enabled was flipped to true in the policy TOML above.
    with pytest.raises(BenchmarkApprovalError, match="must be false"):
        record_benchmark_approval(
            repository, policy_path, manifest_path, approval_path, output_path
        )


@pytest.mark.parametrize(
    "missing_reference,manifest_kwargs,approval_kwargs",
    [
        (
            "repository_commit_hash",
            {"git_revision": "not-a-valid-hash"},
            {},
        ),
        (
            "dataset_configuration_hash",
            {},
            {
                "lineage_configuration_paths": {
                    "training_configuration_hash": TRAINING_CONFIG_PATH,
                    "evaluation_configuration_hash": EVALUATION_CONFIG_PATH,
                }
            },
        ),
        (
            "split_identity",
            {"split_name": ""},
            {},
        ),
        (
            "training_configuration_hash",
            {},
            {
                "lineage_configuration_paths": {
                    "dataset_configuration_hash": DATASET_CONFIG_PATH,
                    "training_configuration_hash": "configs/does-not-exist.toml",
                    "evaluation_configuration_hash": EVALUATION_CONFIG_PATH,
                }
            },
        ),
        (
            "evaluation_configuration_hash",
            {},
            {
                "lineage_configuration_paths": {
                    "dataset_configuration_hash": DATASET_CONFIG_PATH,
                    "training_configuration_hash": TRAINING_CONFIG_PATH,
                    "evaluation_configuration_hash": "configs/does-not-exist.toml",
                }
            },
        ),
        (
            "reproducibility_evidence_reference",
            {"evidence_files": ()},
            {},
        ),
        (
            "run_manifest_reference",
            {"run_id": OTHER_RUN_ID},
            {},
        ),
    ],
)
def test_record_benchmark_approval_fails_closed_on_missing_lineage_reference(
    repository: Path,
    missing_reference: str,
    manifest_kwargs: dict[str, Any],
    approval_kwargs: dict[str, Any],
) -> None:
    """Each of the 7 required lineage references independently blocks approval when missing.

    One parametrized regression test covering every branch of
    _missing_lineage_references: each case breaks exactly one lineage reference (via
    the manifest or approval kwargs) while leaving the other 6 valid, confirming the
    checks are independent rather than one masking another.
    """

    policy_path = _write(repository / "policy.toml", _policy_text())
    manifest_path = _write(repository / "run-manifest.json", _manifest(**manifest_kwargs).to_json())
    approval_path = _write(repository / "approval.toml", _approval_text(**approval_kwargs))
    output_path = repository / "artifacts" / "benchmark_approval.json"

    # Exactly one lineage reference (named by missing_reference) was broken by this
    # parametrization's manifest_kwargs/approval_kwargs.
    with pytest.raises(BenchmarkApprovalError, match=missing_reference):
        record_benchmark_approval(
            repository, policy_path, manifest_path, approval_path, output_path
        )


def test_record_benchmark_approval_fails_closed_on_degraded_git_state(repository: Path) -> None:
    """The `#133` graceful-degradation sentinel must not pass the `#135` lineage gate.

    A manifest whose Git state degraded (unavailable/non-zero exit, not a raised error) records
    `git.revision = UNKNOWN_GIT_REVISION` rather than a 40-char hash. This proves that path still
    fails closed on `repository_commit_hash` instead of letting an unverifiable-provenance run
    pass benchmark approval -- the reason #133 and #135 were bundled into one PR.
    """
    policy_path = _write(repository / "policy.toml", _policy_text())
    manifest_path = _write(
        repository / "run-manifest.json",
        _manifest(git_revision=UNKNOWN_GIT_REVISION).to_json(),
    )
    approval_path = _write(repository / "approval.toml", _approval_text())
    output_path = repository / "artifacts" / "benchmark_approval.json"

    # UNKNOWN_GIT_REVISION is not a valid 40-char hex commit hash.
    with pytest.raises(BenchmarkApprovalError, match="repository_commit_hash"):
        record_benchmark_approval(
            repository, policy_path, manifest_path, approval_path, output_path
        )


def test_record_benchmark_approval_fails_closed_on_unknown_lineage_reference(
    repository: Path,
) -> None:
    """`_missing_lineage_references`'s else-branch fails closed on a reference beyond the known 7.

    `future_requirement` is deliberately not added to `ALL_REFERENCES`, which tracks only the
    known 7 -- this proves the fail-closed catch-all, not a newly recognized reference.
    """
    policy_text = _policy_text().replace(
        '  "run_manifest_reference",\n]',
        '  "run_manifest_reference",\n  "future_requirement",\n]',
    )
    assert "future_requirement" in policy_text
    policy_path = _write(repository / "policy.toml", policy_text)
    manifest_path = _write(repository / "run-manifest.json", _manifest().to_json())
    approval_path = _write(repository / "approval.toml", _approval_text())
    output_path = repository / "artifacts" / "benchmark_approval.json"

    # "future_requirement" has no dedicated check in _missing_lineage_references, so
    # it must fall through to the catch-all else branch and fail closed.
    with pytest.raises(BenchmarkApprovalError, match="future_requirement"):
        record_benchmark_approval(
            repository, policy_path, manifest_path, approval_path, output_path
        )
