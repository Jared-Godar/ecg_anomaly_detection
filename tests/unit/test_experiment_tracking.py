"""Tests for local experiment checkpoint, resume, and progress helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ecg_anomaly_detection.experiment_tracking import (
    ExperimentTracker,
    ExperimentTrackingError,
)


def test_constructor_rejects_empty_candidate_ids(tmp_path: Path) -> None:
    """A tracker with zero candidates is rejected, since progress/finalize would be degenerate."""

    # candidate_ids=[] is the empty case this constructor must reject.
    with pytest.raises(ExperimentTrackingError, match="must not be empty"):
        ExperimentTracker(tmp_path, [])


def test_constructor_rejects_duplicate_candidate_ids(tmp_path: Path) -> None:
    """Duplicate candidate IDs are rejected, since they'd make checkpoint lookups ambiguous."""

    # "a" appears twice in candidate_ids.
    with pytest.raises(ExperimentTrackingError, match="must be unique"):
        ExperimentTracker(tmp_path, ["a", "a"])


@pytest.mark.parametrize("candidate_id", ["", "../evil", "a/b", "..", ".hidden", "trailing-"])
def test_constructor_rejects_unsafe_candidate_ids(tmp_path: Path, candidate_id: str) -> None:
    """Every filesystem-unsafe candidate ID shape is rejected by _CANDIDATE_ID_PATTERN.

    Covers path traversal ("../evil", ".."), path separators ("a/b"), hidden-file
    dots (".hidden"), and a trailing non-alphanumeric character ("trailing-") in one
    parametrized sweep, confirming the pattern rejects each independently.
    """

    # candidate_id is this test's parametrized unsafe value.
    with pytest.raises(ExperimentTrackingError, match="filesystem-safe"):
        ExperimentTracker(tmp_path, [candidate_id])


def test_constructor_refuses_a_symlinked_checkpoint_directory(tmp_path: Path) -> None:
    """A checkpoint_directory that is itself a symlink is rejected, not silently followed."""

    real_target = tmp_path / "elsewhere"
    real_target.mkdir()
    checkpoint_link = tmp_path / "checkpoints"
    checkpoint_link.symlink_to(real_target, target_is_directory=True)

    # checkpoint_link is a symlink to real_target, not a regular directory.
    with pytest.raises(ExperimentTrackingError, match="symbolic link"):
        ExperimentTracker(checkpoint_link, ["a"])


def test_new_tracker_has_no_completed_candidates(tmp_path: Path) -> None:
    """A freshly constructed tracker (no prior checkpoints) reports zero progress."""

    tracker = ExperimentTracker(tmp_path, ["a", "b"])

    assert tracker.completed_candidate_ids() == ()
    assert tracker.is_completed("a") is False
    snapshot = tracker.progress()
    assert snapshot.completed_count == 0
    assert snapshot.remaining_count == 2
    assert snapshot.total_count == 2
    assert snapshot.average_seconds_per_candidate is None
    assert snapshot.estimated_remaining_seconds is None


def test_track_persists_metrics_and_predictions_on_success(tmp_path: Path) -> None:
    """A successful track() block checkpoints both metrics and predictions to disk."""

    tracker = ExperimentTracker(tmp_path, ["a"])

    # Attach both metrics and predictions via the recorder before the block exits.
    with tracker.track("a") as recorder:
        recorder.set_metrics({"accuracy": 0.9})
        recorder.set_predictions(np.array([1, 0, 1]))

    assert tracker.is_completed("a") is True
    result = tracker.load_result("a")
    assert result.candidate_id == "a"
    assert result.metrics == {"accuracy": 0.9}
    assert result.duration_seconds >= 0
    predictions = tracker.load_predictions("a")
    assert predictions is not None
    np.testing.assert_array_equal(predictions, np.array([1, 0, 1]))


def test_track_without_predictions_leaves_predictions_unset(tmp_path: Path) -> None:
    """A track() block that never calls set_predictions leaves predictions unset (None)."""

    tracker = ExperimentTracker(tmp_path, ["a"])

    # Only metrics are attached; set_predictions is deliberately never called.
    with tracker.track("a") as recorder:
        recorder.set_metrics({"accuracy": 0.5})

    assert tracker.load_predictions("a") is None


def test_track_does_not_persist_a_candidate_whose_body_raises(tmp_path: Path) -> None:
    """A raised exception inside track() leaves no checkpoint for that candidate.

    Protects the documented "nothing persisted if the body raises" guarantee: even
    though set_metrics was called before the raise, no result file should end up on disk.
    """

    tracker = ExperimentTracker(tmp_path, ["a"])

    # The RuntimeError raised inside this block must prevent any checkpoint write.
    with pytest.raises(RuntimeError, match="boom"), tracker.track("a") as recorder:
        recorder.set_metrics({"accuracy": 0.5})
        raise RuntimeError("boom")

    assert tracker.is_completed("a") is False
    assert tracker.completed_candidate_ids() == ()


def test_track_rejects_an_unknown_candidate_id(tmp_path: Path) -> None:
    """track() rejects a candidate ID outside the tracker's fixed candidate_ids set."""

    tracker = ExperimentTracker(tmp_path, ["a"])

    # "not-a-candidate" was never passed to ExperimentTracker's constructor.
    with (
        pytest.raises(ExperimentTrackingError, match="unknown candidate ID"),
        tracker.track("not-a-candidate"),
    ):
        pass


def test_load_result_raises_for_a_candidate_with_no_checkpoint(tmp_path: Path) -> None:
    """load_result raises a clear error for a candidate that was never tracked."""

    tracker = ExperimentTracker(tmp_path, ["a"])

    # "a" is a known candidate ID, but track() was never called for it.
    with pytest.raises(ExperimentTrackingError, match="no checkpointed result"):
        tracker.load_result("a")


def test_a_new_tracker_instance_resumes_prior_completions_from_disk(tmp_path: Path) -> None:
    """A second ExperimentTracker over the same directory sees the first instance's completions.

    This is the resume contract this module exists for: checkpoint state lives on
    disk, not in the ExperimentTracker instance, so a fresh process re-running the same
    checkpoint_directory picks up exactly where a prior, possibly-interrupted run left off.
    """

    first = ExperimentTracker(tmp_path, ["a", "b"])
    # Complete candidate "a" with the first tracker instance, then discard it.
    with first.track("a") as recorder:
        recorder.set_metrics({"accuracy": 0.8})

    resumed = ExperimentTracker(tmp_path, ["a", "b"])

    assert resumed.is_completed("a") is True
    assert resumed.is_completed("b") is False
    assert resumed.load_result("a").metrics == {"accuracy": 0.8}


def test_progress_reports_average_and_estimated_remaining_after_a_completion(
    tmp_path: Path,
) -> None:
    """progress() computes a correct average duration and ETA from one completed candidate.

    The injected clock yields exactly the four monotonic() calls track() and the
    constructor make, in order: tracker construction, track() entry, track() exit, and
    the later progress() call reusing the same "now" as track()'s exit -- producing a
    deterministic 3.0s measured duration.
    """

    # Calls in order: constructor start, track() start, track() end, progress().
    clock = iter([0.0, 0.0, 3.0, 3.0])
    tracker = ExperimentTracker(tmp_path, ["a", "b"], monotonic=lambda: next(clock))

    # Complete candidate "a"; the injected clock reports a 3.0s duration for this block.
    with tracker.track("a") as recorder:
        recorder.set_metrics({"accuracy": 0.8})

    snapshot = tracker.progress()

    assert snapshot.completed_count == 1
    assert snapshot.remaining_count == 1
    assert snapshot.average_seconds_per_candidate == 3.0
    assert snapshot.estimated_remaining_seconds == 3.0


def test_finalize_refuses_when_candidates_remain_incomplete(tmp_path: Path) -> None:
    """finalize() refuses to run while any candidate still lacks a checkpointed result."""

    tracker = ExperimentTracker(tmp_path, ["a", "b"])
    # Only "a" is completed; "b" is deliberately left untracked.
    with tracker.track("a") as recorder:
        recorder.set_metrics({"accuracy": 0.8})

    # "b" has no checkpointed result yet.
    with pytest.raises(ExperimentTrackingError, match="1 candidate.*not yet completed"):
        tracker.finalize("accuracy")


def test_finalize_sorts_descending_by_default_and_writes_final_results(tmp_path: Path) -> None:
    """finalize() ranks by the given metric, highest first, by default."""

    tracker = ExperimentTracker(tmp_path, ["a", "b", "c"])
    # Complete every candidate with a distinct accuracy, deliberately out of rank order.
    for candidate_id, accuracy in (("a", 0.7), ("b", 0.95), ("c", 0.8)):
        # Record this candidate's metrics before track()'s block exits.
        with tracker.track(candidate_id) as recorder:
            recorder.set_metrics({"accuracy": accuracy})

    final_path = tracker.finalize("accuracy")

    assert final_path == tmp_path / "final-results.json"

    payload = json.loads(final_path.read_text(encoding="utf-8"))
    assert payload["ranking_metric"] == "accuracy"
    assert payload["higher_is_better"] is True
    assert [row["candidate_id"] for row in payload["results"]] == ["b", "c", "a"]


def test_finalize_sorts_ascending_when_lower_is_better(tmp_path: Path) -> None:
    """higher_is_better=False ranks by the given metric lowest first (e.g. for a loss metric)."""

    tracker = ExperimentTracker(tmp_path, ["a", "b"])
    # Complete both candidates with distinct loss values, deliberately out of rank order.
    for candidate_id, loss in (("a", 0.5), ("b", 0.1)):
        # Record this candidate's metrics before track()'s block exits.
        with tracker.track(candidate_id) as recorder:
            recorder.set_metrics({"loss": loss})

    final_path = tracker.finalize("loss", higher_is_better=False)

    payload = json.loads(final_path.read_text(encoding="utf-8"))
    assert [row["candidate_id"] for row in payload["results"]] == ["b", "a"]


def test_no_leftover_temp_files_after_successful_writes(tmp_path: Path) -> None:
    """Successful checkpoint and finalize writes leave no `.tmp` files behind.

    Protects the atomic-write contract in _atomic_write_json/_atomic_write_npy: the
    temp file used for the rename-into-place trick must always be gone after a
    successful write, not merely after a crash-recovery cleanup.
    """

    tracker = ExperimentTracker(tmp_path, ["a"])
    # Complete the one candidate with both metrics and predictions attached.
    with tracker.track("a") as recorder:
        recorder.set_metrics({"accuracy": 0.9})
        recorder.set_predictions(np.array([1, 0]))
    tracker.finalize("accuracy")

    leftover = list(tmp_path.rglob("*.tmp*"))
    assert leftover == []
