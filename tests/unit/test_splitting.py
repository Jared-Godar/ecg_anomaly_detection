"""Tests for deterministic record-grouped splitting."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ecg_anomaly_detection.config import RepositoryPaths
from ecg_anomaly_detection.splitting import (
    SplitConfig,
    SplitError,
    SplitManifest,
    SplitQualityConfig,
    WindowMetadata,
    create_split_manifest,
    create_split_quality_summary,
    enforce_split_quality,
    load_split_config,
    load_window_metadata,
    read_split_manifest,
    write_split_manifest,
)


@pytest.fixture
def split_config() -> SplitConfig:
    """A schema-2 subject-grouped config: subjects a/a/b/c across records 100-103.

    Records 100 and 101 deliberately share subject-a, so tests can verify those two
    records are always assigned to the same partition together.

    Returns:
        A SplitConfig with 4 records spanning 3 subjects.
    """

    return SplitConfig(
        schema_version=2,
        name="test-subject-split",
        version="2.0.0",
        strategy="seeded-subject-shuffle",
        seed=42,
        train_ratio=0.5,
        validation_ratio=0.25,
        test_ratio=0.25,
        record_subjects={
            "100": "subject-a",
            "101": "subject-a",
            "102": "subject-b",
            "103": "subject-c",
        },
    )


@pytest.fixture
def metadata() -> WindowMetadata:
    """Window-level metadata matching split_config's 4 records, 2 windows each, both classes present.

    Returns:
        WindowMetadata with 8 total windows across records 100-103.
    """

    targets = np.asarray([0, 1, 0, 1, 1, 0, 0, 0], dtype=np.int64)
    targets.setflags(write=False)
    return WindowMetadata(
        record_ids=("100", "100", "101", "101", "102", "102", "103", "103"),
        target_values=targets,
        source_artifacts=("windows.npz",),
        mapping_name="binary-map",
        mapping_version="1.0.0",
        window_config_name="six-second",
        window_config_version="1.0.0",
    )


def test_repository_split_config_is_versioned_and_subject_grouped() -> None:
    """The committed splitting-v2.toml config declares schema-2 subject grouping correctly.

    Regression test guarding against an accidental edit to the repository's real
    split config: confirms it declares the seeded-subject-shuffle strategy, groups
    records 201/202 under the same subject, and its ratios still sum to 1.0.
    """

    paths = RepositoryPaths.discover(Path(__file__))
    config = load_split_config(paths.configs / "splitting-v2.toml")

    assert config.name == "subject-aware-holdout"
    assert config.strategy == "seeded-subject-shuffle"
    assert config.record_subjects["201"] == config.record_subjects["202"]
    assert config.seed == 2022
    assert config.train_ratio + config.validation_ratio + config.test_ratio == pytest.approx(1.0)


def test_split_is_deterministic_complete_and_subject_disjoint(
    split_config: SplitConfig, metadata: WindowMetadata, tmp_path: Path
) -> None:
    """The seeded shuffle is deterministic, covers every record/subject, and is leakage-free.

    The core anti-leakage regression test: two independent create_split_manifest calls
    with the same seed must produce byte-identical manifests, every subject/record must
    appear in exactly one partition (union covers the input, no pairwise overlap), and
    subjects sharing records (100/101 both under subject-a) must land in the same
    partition together. Also round-trips the manifest through JSON to confirm
    read_split_manifest reconstructs an equal object.

    Args:
        split_config: The 4-record, 3-subject fixture config.
        metadata: Matching 8-window metadata for split_config's records.
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    first = create_split_manifest(split_config, metadata)
    second = create_split_manifest(split_config, metadata)
    output_path = tmp_path / "split.json"
    write_split_manifest(first, output_path)

    assert first == second
    memberships = [set(summary.record_ids) for summary in first.partitions.values()]
    assert set().union(*memberships) == {"100", "101", "102", "103"}
    assert all(
        not left & right
        for index, left in enumerate(memberships)
        for right in memberships[index + 1 :]
    )
    subject_memberships = [set(summary.subject_ids) for summary in first.partitions.values()]
    assert all(
        not left & right
        for index, left in enumerate(subject_memberships)
        for right in subject_memberships[index + 1 :]
    )
    assert next(
        name for name, summary in first.partitions.items() if "100" in summary.record_ids
    ) == next(name for name, summary in first.partitions.items() if "101" in summary.record_ids)
    assert sum(summary.window_count for summary in first.partitions.values()) == 8
    assert (
        sum(sum(summary.target_value_counts.values()) for summary in first.partitions.values()) == 8
    )
    assert all(
        set(summary.target_value_counts) == {"0", "1"} for summary in first.partitions.values()
    )
    assert '"strategy": "seeded-subject-shuffle"' in output_path.read_text(encoding="utf-8")
    assert read_split_manifest(output_path) == first


def test_three_subjects_produce_three_nonempty_partitions(split_config: SplitConfig) -> None:
    """The minimum viable case (exactly 3 subjects) still produces one subject per partition.

    Boundary test at create_split_manifest's documented floor: subject-grouped
    splitting requires at least 3 subjects, so the smallest valid input should still
    partition cleanly, one subject landing in each of train/validation/test.

    Args:
        split_config: The 4-record, 3-subject fixture config.
    """

    metadata = WindowMetadata(
        record_ids=("100", "101", "102", "103"),
        target_values=np.asarray([0, 0, 0, 1], dtype=np.int64),
        source_artifacts=("windows.npz",),
        mapping_name="map",
        mapping_version="1",
        window_config_name="window",
        window_config_version="1",
    )

    manifest = create_split_manifest(split_config, metadata)

    assert {summary.subject_count for summary in manifest.partitions.values()} == {1}


def test_split_rejects_too_few_subjects(split_config: SplitConfig) -> None:
    """Below the 3-subject floor, create_split_manifest fails rather than degrading silently.

    Args:
        split_config: The base fixture config, overridden below to only 2 subjects.
    """

    metadata = WindowMetadata(
        record_ids=("100", "101", "102"),
        target_values=np.asarray([0, 0, 1], dtype=np.int64),
        source_artifacts=("windows.npz",),
        mapping_name="map",
        mapping_version="1",
        window_config_name="window",
        window_config_version="1",
    )

    # Only 2 distinct subjects (subject-a, subject-b) are declared below, one short of
    # create_split_manifest's documented 3-subject minimum.
    with pytest.raises(SplitError, match="at least 3 subjects"):
        create_split_manifest(
            replace(
                split_config,
                record_subjects={
                    "100": "subject-a",
                    "101": "subject-a",
                    "102": "subject-b",
                },
            ),
            metadata,
        )


def test_window_metadata_loader_rejects_record_reuse_across_artifacts(tmp_path: Path) -> None:
    """The same record ID appearing in two separate NPZ shards is rejected, not silently merged.

    Protects the load-bearing leakage guard in load_window_metadata: a record must
    originate from exactly one artifact, since a duplicate would make subsequent
    subject-grouped partitioning unable to guarantee that record lands in only one split.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_metadata_artifact(first, ["100"], [0])
    _write_metadata_artifact(second, ["100"], [1])

    # Record "100" appears in both `first` and `second`.
    with pytest.raises(SplitError, match="multiple window artifacts"):
        load_window_metadata([first, second])


def test_manifest_reader_rejects_subject_crossing_partitions(
    split_config: SplitConfig, metadata: WindowMetadata
) -> None:
    """A hand-edited manifest with a subject duplicated across partitions is rejected on read.

    Protects _validate_serialized_manifest's re-verification of the leakage invariant:
    a manifest read back from disk is untrusted input (possibly hand-edited), so
    SplitManifest.from_json must independently re-check subject disjointness rather
    than assuming a JSON file on disk is still valid just because it once was.

    Args:
        split_config: The 4-record, 3-subject fixture config.
        metadata: Matching 8-window metadata for split_config's records.
    """

    document = json.loads(create_split_manifest(split_config, metadata).to_json())
    leaked_subject = document["partitions"]["train"]["subject_ids"][0]
    validation = document["partitions"]["validation"]
    displaced_subject = validation["subject_ids"][0]
    validation["subject_ids"] = [leaked_subject]
    validation["record_subjects"] = {
        record_id: leaked_subject for record_id in validation["record_ids"]
    }
    assert displaced_subject != leaked_subject

    # leaked_subject was copied from train into validation's subject_ids above.
    with pytest.raises(SplitError, match="subject leakage"):
        SplitManifest.from_json(json.dumps(document))


def test_quality_summary_reports_deterministic_distributions_and_disjointness(
    split_config: SplitConfig, metadata: WindowMetadata
) -> None:
    """create_split_quality_summary is deterministic and reports a clean pass for a valid split.

    Args:
        split_config: The 4-record, 3-subject fixture config.
        metadata: Matching 8-window metadata for split_config's records.
    """

    manifest = create_split_manifest(split_config, metadata)

    first = create_split_quality_summary(split_config, manifest, metadata)
    second = create_split_quality_summary(split_config, manifest, metadata)

    assert first.to_json() == second.to_json()
    assert first.status == "passed"
    assert first.subject_disjoint is True
    assert first.record_disjoint is True
    assert first.acceptance_checks["min_subjects_per_partition"] == 1
    assert tuple(first.partitions) == ("train", "validation", "test")
    assert sum(item["window_count"] for item in first.partitions.values()) == 8
    assert all(item["class_counts"].keys() == {"0", "1"} for item in first.partitions.values())
    assert all(item["binary_counts"] is not None for item in first.partitions.values())


@pytest.mark.parametrize(
    "severity, expected_status", [("warning", "warning"), ("failure", "failed")]
)
def test_quality_summary_handles_missing_class_and_insufficient_counts(
    split_config: SplitConfig,
    metadata: WindowMetadata,
    severity: str,
    expected_status: str,
) -> None:
    """default_severity controls whether a quality violation is a warning or a hard failure.

    Configures min_subjects_per_partition=2 (violated, since each partition here has 1
    subject) and required_classes=(0, 1, 2) (class 2 never occurs, violating
    required_class_coverage), then confirms both the reported status and whether
    enforce_split_quality actually raises track the configured severity.

    Args:
        split_config: The 4-record, 3-subject fixture config, overridden below with a
            stricter quality policy.
        metadata: Matching 8-window metadata for split_config's records.
        severity: The configured default_severity ("warning" or "failure").
        expected_status: The summary status that severity should produce.
    """

    config = replace(
        split_config,
        quality=SplitQualityConfig(
            min_subjects_per_partition=2,
            required_class_coverage=("validation", "test"),
            required_classes=(0, 1, 2),
            default_severity=severity,
        ),
    )
    summary = create_split_quality_summary(
        config, create_split_manifest(config, metadata), metadata
    )

    assert summary.status == expected_status
    assert {item.check for item in summary.violations} == {
        "minimum_subjects",
        "required_class_coverage",
    }
    # Only the "failure" severity should make enforce_split_quality actually raise;
    # "warning" severity records the same violations but must not block execution.
    if severity == "failure":
        # default_severity="failure" means enforce_split_quality must raise.
        with pytest.raises(SplitError, match="quality checks failed"):
            enforce_split_quality(summary)
    else:
        enforce_split_quality(summary)


def test_quality_summary_shard_count_and_ratios_with_multiple_source_artifacts(
    split_config: SplitConfig,
) -> None:
    """shard_count and actual_ratios["shards"] must be correct when len(source_artifacts) > 1.

    Regression test for #131: the old comprehension used record_id as a fallback shard path
    instead of the real shard path, corrupting both shard_count and the shards ratio.

    Setup: three shards, one per partition (disjoint), so actual_ratios["shards"] sums to 1.0.
    split_config assigns: subject-a → train, subject-b → validation, subject-c → test.
    Records:  100/101 → subject-a → shard_a (train)
              102     → subject-b → shard_b (validation)
              103     → subject-c → shard_c (test)
    """
    shard_a = "data/shard_a.npz"
    shard_b = "data/shard_b.npz"
    shard_c = "data/shard_c.npz"
    metadata = WindowMetadata(
        record_ids=("100", "100", "101", "101", "102", "102", "103", "103"),
        target_values=np.asarray([0, 1, 0, 1, 1, 0, 0, 0], dtype=np.int64),
        source_artifacts=(shard_a, shard_b, shard_c),
        mapping_name="binary-map",
        mapping_version="1.0.0",
        window_config_name="six-second",
        window_config_version="1.0.0",
        record_shards={
            "100": shard_a,
            "101": shard_a,
            "102": shard_b,
            "103": shard_c,
        },
    )

    manifest = create_split_manifest(split_config, metadata)
    summary = create_split_quality_summary(split_config, manifest, metadata)

    assert summary.status == "passed"

    # Each partition's shard_count must equal the number of distinct shard paths in that partition.
    # train (100, 101) → {shard_a}       → shard_count == 1
    # validation (102) → {shard_b}       → shard_count == 1
    # test (103)       → {shard_c}       → shard_count == 1
    for partition_name, diag in summary.partitions.items():
        record_ids = set(manifest.partitions[partition_name].record_ids)
        expected_shards = {metadata.record_shards[r] for r in record_ids}
        assert diag["shard_count"] == len(expected_shards), (
            f"partition {partition_name}: expected shard_count={len(expected_shards)}, "
            f"got {diag['shard_count']}"
        )

    # Shards are disjoint across partitions (3 unique total, 1 per partition),
    # so actual_ratios["shards"] must sum to exactly 1.0.
    shards_ratio_sum = sum(diag["actual_ratios"]["shards"] for diag in summary.partitions.values())
    assert shards_ratio_sum == pytest.approx(1.0), (
        f"actual_ratios['shards'] sum expected 1.0, got {shards_ratio_sum}"
    )


def _write_metadata_artifact(path: Path, record_ids: list[str], target_values: list[int]) -> None:
    """Write a minimal, valid window-metadata NPZ artifact for load_window_metadata tests.

    Args:
        path: Where to write the NPZ artifact.
        record_ids: Record ID for each window row.
        target_values: Target label for each window row.
    """

    np.savez_compressed(
        path,
        schema_version=np.asarray(1, dtype=np.int64),
        record_ids=np.asarray(record_ids, dtype=np.str_),
        target_values=np.asarray(target_values, dtype=np.int64),
        mapping_name=np.asarray("binary-map", dtype=np.str_),
        mapping_version=np.asarray("1.0.0", dtype=np.str_),
        window_config_name=np.asarray("six-second", dtype=np.str_),
        window_config_version=np.asarray("1.0.0", dtype=np.str_),
    )
