"""Tests for bounded, digest-verified Colab notebook handoff archives."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import pytest

from ecg_anomaly_detection import notebook_handoff
from ecg_anomaly_detection.notebook_handoff import (
    HANDOFF_MANIFEST_MEMBER,
    HANDOFF_POINTER_NAME,
    NotebookHandoffError,
    create_handoff,
    restore_handoff,
)

# Fixed run identity keeps fixture paths readable and makes cross-run checks explicit.
RUN_ID = "11111111-2222-3333-4444-555555555555"
# Two exact but distinct commits exercise source-lineage matching without invoking Git.
SOURCE_COMMIT = "a" * 40
# A different valid object ID represents a fresh checkout at the wrong revision.
OTHER_COMMIT = "b" * 40


def _write_json(path: Path, payload: object) -> None:
    """Write one readable JSON fixture after creating its parent directories.

    Args:
        path: Fixture destination.
        payload: JSON-serializable value.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _init_handoff_repository(root: Path, *, status: str = "complete") -> dict[str, Path]:
    """Create one minimal Step 0 source/target fixture with three partition shards.

    Args:
        root: Temporary checkout root.
        status: Step 0 status value written into the local status contract.

    Returns:
        Named paths used by assertions and mutation tests.
    """

    # Both markers are required by production repository-root validation.
    (root / "src").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")

    train = root / f"data/interim/runs/{RUN_ID}/windows/train.npz"
    validation = root / f"data/interim/runs/{RUN_ID}/windows/validation.npz"
    protected_test = root / f"data/interim/runs/{RUN_ID}/windows/test.npz"
    raw = root / "data/raw/mitdb/100.dat"
    # Distinct bytes let tests prove the protected and raw files were not accidentally
    # selected merely because their directories exist beside development data.
    for path, content in (
        (train, b"train-waveforms"),
        (validation, b"validation-waveforms"),
        (protected_test, b"protected-test-waveforms"),
        (raw, b"raw-source-record"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    index = root / f"data/processed/runs/{RUN_ID}/dataset-index.json"
    # Only path/record fields are required by the handoff selector; production indexes
    # contain richer counts and hashes that remain transported unchanged.
    _write_json(
        index,
        {
            "partitions": {
                "train": {
                    "shards": [
                        {
                            "record_id": "train-record",
                            "file": {"path": train.relative_to(root).as_posix()},
                        }
                    ]
                },
                "validation": {
                    "shards": [
                        {
                            "record_id": "validation-record",
                            "file": {"path": validation.relative_to(root).as_posix()},
                        }
                    ]
                },
                "test": {
                    "shards": [
                        {
                            "record_id": "test-record",
                            "file": {"path": protected_test.relative_to(root).as_posix()},
                        }
                    ]
                },
            }
        },
    )

    artifact_root = root / f"artifacts/runs/{RUN_ID}"
    split = artifact_root / "split.json"
    split_quality = artifact_root / "split_quality_summary.json"
    run_manifest = artifact_root / "run-manifest.json"
    baseline = artifact_root / "evaluation/validation-metrics.json"
    # Each required artifact gets recognizable content so restoration assertions can
    # compare exact bytes rather than only file existence.
    for path, payload in (
        (split, {"split": "grouped"}),
        (split_quality, {"status": "passed"}),
        (run_manifest, {"run_id": RUN_ID}),
        (baseline, {"accuracy": 0.5}),
    ):
        _write_json(path, payload)

    status_path = root / "notebooks/local/step0-pipeline-status.json"
    _write_json(
        status_path,
        {
            "status": status,
            "artifacts": {
                "dataset_index": index.relative_to(root).as_posix(),
                "split_manifest": split.relative_to(root).as_posix(),
                "split_quality": split_quality.relative_to(root).as_posix(),
                "run_manifest": run_manifest.relative_to(root).as_posix(),
            },
        },
    )
    return {
        "status": status_path,
        "index": index,
        "train": train,
        "validation": validation,
        "test": protected_test,
        "raw": raw,
        "baseline": baseline,
    }


def _archive_manifest(archive: Path) -> dict[str, object]:
    """Read one test archive's fixed manifest member.

    Args:
        archive: Created handoff ZIP.

    Returns:
        Parsed manifest object.
    """

    # The helper reads only the fixed small JSON member, never any waveform content.
    with zipfile.ZipFile(archive) as bundle:
        value = json.loads(bundle.read(HANDOFF_MANIFEST_MEMBER))
    assert isinstance(value, dict)
    return value


def test_create_handoff_includes_only_downstream_development_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Creation includes required evidence/train/validation but excludes raw and test.

    Args:
        tmp_path: Isolated source and Drive-like directories.
        monkeypatch: Replaces Git discovery with a deterministic exact commit.
    """

    source = tmp_path / "source"
    paths = _init_handoff_repository(source)
    monkeypatch.setattr(notebook_handoff, "_git_commit", lambda _root: SOURCE_COMMIT)

    result = create_handoff(source, tmp_path / "drive")

    manifest = _archive_manifest(result.archive)
    assert result.operation == "created"
    included = {str(entry["path"]) for entry in manifest["files"]}  # type: ignore[index]
    assert paths["train"].relative_to(source).as_posix() in included
    assert paths["validation"].relative_to(source).as_posix() in included
    assert paths["baseline"].relative_to(source).as_posix() in included
    assert paths["test"].relative_to(source).as_posix() not in included
    assert paths["raw"].relative_to(source).as_posix() not in included
    assert manifest["included_partitions"] == ["train", "validation"]
    assert "clinical" in str(manifest["claim_boundary"])

    pointer = json.loads((tmp_path / "drive" / HANDOFF_POINTER_NAME).read_text())
    assert pointer["archive_name"] == result.archive.name
    assert pointer["repository_commit"] == SOURCE_COMMIT
    assert pointer["run_id"] == RUN_ID


def test_create_handoff_rejects_noncomplete_step0_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A blocked Step 0 run cannot be transported as downstream-ready state.

    Args:
        tmp_path: Isolated source and Drive-like directories.
        monkeypatch: Replaces Git discovery with a deterministic exact commit.
    """

    source = tmp_path / "source"
    _init_handoff_repository(source, status="blocked")
    monkeypatch.setattr(notebook_handoff, "_git_commit", lambda _root: SOURCE_COMMIT)

    # Creation must fail before a destination archive or pointer is written.
    with pytest.raises(NotebookHandoffError, match="not complete"):
        create_handoff(source, tmp_path / "drive")


def test_restore_handoff_verifies_and_recreates_required_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restore writes verified development state and leaves protected/raw files absent.

    Args:
        tmp_path: Isolated source, target, and Drive-like directories.
        monkeypatch: Supplies matching source/target Git identity.
    """

    source = tmp_path / "source"
    source_paths = _init_handoff_repository(source)
    monkeypatch.setattr(notebook_handoff, "_git_commit", lambda _root: SOURCE_COMMIT)
    created = create_handoff(source, tmp_path / "drive")

    target = tmp_path / "target"
    target_paths = _init_handoff_repository(target)
    # Remove all generated content from the target while retaining repository markers.
    for directory in (target / "data", target / "artifacts", target / "notebooks"):
        # The target models a fresh checkout with no ignored generated state.
        if directory.exists():
            shutil.rmtree(directory)

    restored = restore_handoff(target, created.archive)

    assert restored.operation == "restored"
    assert restored.run_id == RUN_ID
    assert target_paths["train"].read_bytes() == source_paths["train"].read_bytes()
    assert target_paths["validation"].read_bytes() == source_paths["validation"].read_bytes()
    assert not target_paths["test"].exists()
    assert not target_paths["raw"].exists()


def test_create_handoff_rerun_verifies_existing_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repeated final notebook cell verifies rather than rewrites its versioned ZIP.

    Args:
        tmp_path: Isolated source and Drive-like directories.
        monkeypatch: Supplies a deterministic exact commit.
    """

    source = tmp_path / "source"
    _init_handoff_repository(source)
    monkeypatch.setattr(notebook_handoff, "_git_commit", lambda _root: SOURCE_COMMIT)
    first = create_handoff(source, tmp_path / "drive")
    first_bytes = first.archive.read_bytes()

    repeated = create_handoff(source, tmp_path / "drive")

    assert repeated.operation == "verified_existing"
    assert repeated.archive == first.archive
    assert repeated.archive.read_bytes() == first_bytes


def test_restore_handoff_rejects_wrong_checkout_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restore fails before extraction when the checkout revision differs.

    Args:
        tmp_path: Isolated source, target, and Drive-like directories.
        monkeypatch: Changes the reported Git identity between create and restore.
    """

    source = tmp_path / "source"
    _init_handoff_repository(source)
    monkeypatch.setattr(notebook_handoff, "_git_commit", lambda _root: SOURCE_COMMIT)
    created = create_handoff(source, tmp_path / "drive")

    target = tmp_path / "target"
    _init_handoff_repository(target)
    monkeypatch.setattr(notebook_handoff, "_git_commit", lambda _root: OTHER_COMMIT)

    # Source mismatch is checked before any archive member reaches the target.
    with pytest.raises(NotebookHandoffError, match="commit does not match"):
        restore_handoff(target, created.archive)


def test_restore_handoff_rejects_unmanifested_archive_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An extra ZIP payload cannot bypass manifest digest and partition controls.

    Args:
        tmp_path: Isolated source, target, and Drive-like directories.
        monkeypatch: Supplies a deterministic exact commit.
    """

    source = tmp_path / "source"
    _init_handoff_repository(source)
    monkeypatch.setattr(notebook_handoff, "_git_commit", lambda _root: SOURCE_COMMIT)
    created = create_handoff(source, tmp_path / "drive")
    # Append one file after creation without adding a manifest entry for it.
    # Mutate only the test archive to simulate an undeclared payload.
    with zipfile.ZipFile(created.archive, mode="a") as bundle:
        bundle.writestr("data/interim/unmanifested-test.bin", b"unexpected")

    target = tmp_path / "target"
    _init_handoff_repository(target)
    # Exact ZIP membership must detect the appended undeclared content.
    with pytest.raises(NotebookHandoffError, match="membership differs"):
        restore_handoff(target, created.archive)


def test_restore_handoff_refuses_different_existing_generated_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rerun is idempotent only when existing destination bytes match the archive.

    Args:
        tmp_path: Isolated source, target, and Drive-like directories.
        monkeypatch: Supplies a deterministic exact commit.
    """

    source = tmp_path / "source"
    _init_handoff_repository(source)
    monkeypatch.setattr(notebook_handoff, "_git_commit", lambda _root: SOURCE_COMMIT)
    created = create_handoff(source, tmp_path / "drive")

    target = tmp_path / "target"
    target_paths = _init_handoff_repository(target)
    target_paths["train"].write_bytes(b"different-local-generated-state")

    # Existing divergent ignored state is never silently replaced.
    with pytest.raises(NotebookHandoffError, match="Refusing to overwrite"):
        restore_handoff(target, created.archive)
