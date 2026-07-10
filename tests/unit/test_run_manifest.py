"""Tests for auditable run-manifest creation."""

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

import ecg_anomaly_detection.run_manifest as run_manifest
from ecg_anomaly_detection.run_manifest import (
    UNKNOWN_GIT_REVISION,
    EnvironmentSnapshot,
    GitState,
    RunManifest,
    RunManifestError,
    create_run_manifest,
    read_run_manifest,
    write_run_manifest,
)

# A well-formed canonical UUID used as the fixed run ID across these tests, injected
# via run_id_factory so manifest content is deterministic and assertable.
RUN_ID = "12345678-1234-5678-1234-567812345678"


class _FailingSubprocess:
    """Stand-in for the `subprocess` module when Git is unavailable."""

    CalledProcessError = subprocess.CalledProcessError

    @staticmethod
    def run(*_args: object, **_kwargs: object) -> None:
        """Simulate `git` being absent from PATH, regardless of what command was requested.

        Args:
            _args: Unused; every call raises regardless of the command.
            _kwargs: Unused; every call raises regardless of the command.
        """

        raise FileNotFoundError("git executable not found")


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """A fake repository with a valid, cross-referenced inventory manifest and split manifest.

    Three records (100/101/102) are split one-per-partition with distinct target-class
    distributions, so tests exercising leakage detection or manifest field values have
    non-degenerate data to work with.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.

    Returns:
        The fake repository root path.
    """

    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (tmp_path / "configs").mkdir()
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "configs" / "dataset.toml").write_text("schema_version=1\n", encoding="utf-8")
    (tmp_path / "artifacts" / "inventory.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_slug": "synthetic",
                "dataset_version": "1.0.0",
                "created_at_utc": "2026-01-01T00:00:00Z",
                "files": [
                    {"path": "100.dat", "size_bytes": 10, "sha256": "a" * 64},
                    {"path": "100.hea", "size_bytes": 5, "sha256": "b" * 64},
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "artifacts" / "split.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "split_name": "grouped",
                "split_version": "1.0.0",
                "strategy": "seeded-record-shuffle",
                "seed": 7,
                "mapping_name": "binary",
                "mapping_version": "1.0.0",
                "window_config_name": "six-second",
                "window_config_version": "1.0.0",
                "source_artifacts": ["data/interim/windows.npz"],
                "total_record_count": 3,
                "total_window_count": 6,
                "partitions": {
                    "train": {
                        "record_ids": ["100"],
                        "record_count": 1,
                        "window_count": 2,
                        "target_value_counts": {"0": 1, "1": 1},
                    },
                    "validation": {
                        "record_ids": ["101"],
                        "record_count": 1,
                        "window_count": 2,
                        "target_value_counts": {"0": 2, "1": 0},
                    },
                    "test": {
                        "record_ids": ["102"],
                        "record_count": 1,
                        "window_count": 2,
                        "target_value_counts": {"0": 0, "1": 2},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "artifacts" / "windows.npz").write_bytes(b"synthetic-artifact")
    return tmp_path


def test_run_manifest_is_deterministic_with_injected_runtime_evidence(
    repository: Path,
) -> None:
    """Every injected evidence provider's output flows through into the written manifest.

    Confirms create_run_manifest correctly assembles fields from every one of its
    inputs -- inventory (file_count/total_size_bytes/digests), split (record counts,
    per-partition record IDs), config/artifact paths (stored repository-relative, never
    absolute), and the injected environment/clock/run-ID providers -- into one coherent document.
    """

    manifest = _create_manifest(repository)
    output = repository / "artifacts" / "run.json"

    write_run_manifest(manifest, repository, output)

    assert manifest.run_id == RUN_ID
    assert manifest.created_at_utc == "2026-01-02T03:04:05Z"
    assert manifest.git.revision == "1" * 40
    assert manifest.dataset.file_count == 2
    assert manifest.dataset.total_size_bytes == 15
    assert manifest.dataset.source_files[0].sha256 == "a" * 64
    assert manifest.dependency_lock.path == "uv.lock"
    assert manifest.split.total_record_count == 3
    assert manifest.split.partitions["test"].record_ids == ("102",)
    assert manifest.configuration_files[0].path == "configs/dataset.toml"
    assert manifest.artifact_files[0].path == "artifacts/windows.npz"
    assert all(not evidence.path.startswith("/") for evidence in manifest.artifact_files)
    assert json.loads(output.read_text(encoding="utf-8"))["environment"]["installed_packages"] == {
        "numpy": "2.4.2"
    }


def test_run_manifest_rejects_evidence_outside_repository(
    repository: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """A config path outside the repository root is rejected, not silently included.

    Protects _resolve_evidence_path's containment check: `external` lives in a
    completely separate temp directory, outside `repository`'s tree entirely.
    """

    external = tmp_path_factory.mktemp("external") / "outside.toml"
    external.write_text("secret=false\n", encoding="utf-8")

    # `external` is outside `repository`'s directory tree entirely.
    with pytest.raises(RunManifestError, match="within repository root"):
        create_run_manifest(
            repository,
            repository / "artifacts" / "inventory.json",
            repository / "artifacts" / "split.json",
            [external],
            git_state_provider=lambda _: GitState("1" * 40, False),
            environment_provider=_environment,
        )


def test_run_manifest_rejects_split_record_leakage(repository: Path) -> None:
    """A split manifest with a record duplicated across partitions is rejected on read.

    create_run_manifest reads the split manifest via read_split_manifest, which
    re-verifies leakage-freedom on every load; this confirms that check actually
    propagates up through _read_split_evidence rather than being silently swallowed.
    """

    split_path = repository / "artifacts" / "split.json"
    document = json.loads(split_path.read_text(encoding="utf-8"))
    document["partitions"]["test"]["record_ids"] = ["100"]
    split_path.write_text(json.dumps(document), encoding="utf-8")

    # Record "100" was copied from train into test's record_ids above.
    with pytest.raises(RunManifestError, match="record leakage"):
        _create_manifest(repository)


def test_run_manifest_rejects_naive_clock(repository: Path) -> None:
    """A clock returning a naive (timezone-unaware) datetime is rejected.

    created_at_utc is serialized with an explicit "Z" suffix, which would be a lie
    about the actual timezone if the source datetime carried no timezone info at all.
    """

    # datetime(2026, 1, 2) has no tzinfo.
    with pytest.raises(RunManifestError, match="timezone-aware"):
        create_run_manifest(
            repository,
            repository / "artifacts" / "inventory.json",
            repository / "artifacts" / "split.json",
            [repository / "configs" / "dataset.toml"],
            clock=lambda: datetime(2026, 1, 2),
            run_id_factory=lambda: RUN_ID,
            git_state_provider=lambda _: GitState("1" * 40, False),
            environment_provider=_environment,
        )


def test_run_manifest_output_must_stay_under_ignored_artifacts(repository: Path) -> None:
    """An output path outside artifacts/ is rejected, matching the directory contract."""

    # Path("run.json") is repository-root-relative, not under artifacts/.
    with pytest.raises(RunManifestError, match="under artifacts"):
        write_run_manifest(_create_manifest(repository), repository, Path("run.json"))


def test_read_run_manifest_round_trips_a_written_manifest(repository: Path) -> None:
    """read_run_manifest reconstructs an object equal to the one write_run_manifest wrote.

    Protects _manifest_from_document's field-by-field reconstruction: every nested
    structure (GitState, EnvironmentSnapshot, DatasetEvidence, SplitEvidence and its
    partitions) must round-trip through JSON without loss or reordering.
    """

    manifest = _create_manifest(repository)
    output = repository / "artifacts" / "run.json"
    write_run_manifest(manifest, repository, output)

    reloaded = read_run_manifest(output)

    assert reloaded == manifest


def test_read_run_manifest_rejects_invalid_json(tmp_path: Path) -> None:
    """Malformed JSON is rejected with a message naming the file, not a bare JSONDecodeError."""

    path = tmp_path / "run.json"
    path.write_text("not json", encoding="utf-8")

    # "not json" isn't valid JSON at all.
    with pytest.raises(RunManifestError, match="could not read run manifest"):
        read_run_manifest(path)


def test_read_run_manifest_rejects_incomplete_document(tmp_path: Path) -> None:
    """A structurally valid but field-incomplete JSON document is rejected on read.

    Protects read_run_manifest's KeyError-to-RunManifestError translation: a document
    with only schema_version (missing every other required field) must raise the
    module's own exception type, not propagate a bare KeyError.
    """

    path = tmp_path / "run.json"
    path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")

    # Only schema_version is present; every other required field is missing.
    with pytest.raises(RunManifestError, match="invalid run manifest"):
        read_run_manifest(path)


def test_capture_git_state_degrades_gracefully_when_git_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_capture_git_state degrades to the UNKNOWN_GIT_REVISION sentinel when git isn't installed.

    Protects the graceful-degradation contract directly at the unit level, using
    _FailingSubprocess to simulate a missing `git` executable (FileNotFoundError)
    rather than depending on the test environment actually lacking Git.
    """

    monkeypatch.setattr(run_manifest, "subprocess", _FailingSubprocess)

    state = run_manifest._capture_git_state(tmp_path)

    assert state == GitState(revision=UNKNOWN_GIT_REVISION, dirty=None)


def test_create_run_manifest_completes_when_git_is_unavailable(
    repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """create_run_manifest still succeeds end to end when Git is unavailable.

    Same simulated-missing-git setup as the _capture_git_state unit test above, but
    exercised through the full create_run_manifest entry point (using the default
    git_state_provider, not an injected fake) to confirm the degraded GitState
    propagates into a complete, otherwise-valid manifest rather than aborting the whole call.
    """

    monkeypatch.setattr(run_manifest, "subprocess", _FailingSubprocess)

    manifest = create_run_manifest(
        repository,
        Path("artifacts/inventory.json"),
        Path("artifacts/split.json"),
        [Path("configs/dataset.toml")],
        artifact_paths=[Path("artifacts/windows.npz")],
        clock=lambda: datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        run_id_factory=lambda: RUN_ID,
        environment_provider=_environment,
    )

    assert manifest.git == GitState(revision=UNKNOWN_GIT_REVISION, dirty=None)


def test_run_manifest_round_trips_degraded_git_state_as_json_null(repository: Path) -> None:
    """A degraded GitState (UNKNOWN_GIT_REVISION, dirty=None) round-trips through JSON correctly.

    Confirms `dirty: None` serializes as JSON `null` (not omitted or coerced to
    false), and that read_run_manifest reconstructs the exact same degraded GitState
    rather than misinterpreting null as some other sentinel.
    """

    manifest = create_run_manifest(
        repository,
        Path("artifacts/inventory.json"),
        Path("artifacts/split.json"),
        [Path("configs/dataset.toml")],
        artifact_paths=[Path("artifacts/windows.npz")],
        clock=lambda: datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        run_id_factory=lambda: RUN_ID,
        git_state_provider=lambda _: GitState(UNKNOWN_GIT_REVISION, None),
        environment_provider=_environment,
    )
    output = repository / "artifacts" / "run.json"
    write_run_manifest(manifest, repository, output)

    assert json.loads(output.read_text(encoding="utf-8"))["git"] == {
        "revision": UNKNOWN_GIT_REVISION,
        "dirty": None,
    }
    reloaded = read_run_manifest(output)
    assert reloaded.git == GitState(revision=UNKNOWN_GIT_REVISION, dirty=None)
    assert reloaded == manifest


def _create_manifest(repository: Path) -> RunManifest:
    """Build a complete, valid RunManifest against the `repository` fixture's evidence.

    Shared by every test that needs a working manifest without repeating the same
    six-argument create_run_manifest call; a fixed clock, run ID, and Git state make
    the result deterministic across tests.

    Args:
        repository: The fake repository root to build a manifest against.

    Returns:
        A fully populated RunManifest.
    """

    return create_run_manifest(
        repository,
        Path("artifacts/inventory.json"),
        Path("artifacts/split.json"),
        [Path("configs/dataset.toml")],
        artifact_paths=[Path("artifacts/windows.npz")],
        clock=lambda: datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        run_id_factory=lambda: RUN_ID,
        git_state_provider=lambda _: GitState("1" * 40, False),
        environment_provider=_environment,
    )


def _environment() -> EnvironmentSnapshot:
    """A fixed, deterministic EnvironmentSnapshot, injected in place of the real environment_provider.

    Returns:
        A fixed EnvironmentSnapshot with one installed package (numpy).
    """

    return EnvironmentSnapshot(
        python_version="3.12.13",
        python_implementation="CPython",
        platform="darwin",
        machine="arm64",
        installed_packages={"numpy": "2.4.2"},
    )
