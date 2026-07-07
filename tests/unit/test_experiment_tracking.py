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
    with pytest.raises(ExperimentTrackingError, match="must not be empty"):
        ExperimentTracker(tmp_path, [])


def test_constructor_rejects_duplicate_candidate_ids(tmp_path: Path) -> None:
    with pytest.raises(ExperimentTrackingError, match="must be unique"):
        ExperimentTracker(tmp_path, ["a", "a"])


@pytest.mark.parametrize("candidate_id", ["", "../evil", "a/b", "..", ".hidden", "trailing-"])
def test_constructor_rejects_unsafe_candidate_ids(tmp_path: Path, candidate_id: str) -> None:
    with pytest.raises(ExperimentTrackingError, match="filesystem-safe"):
        ExperimentTracker(tmp_path, [candidate_id])


def test_constructor_refuses_a_symlinked_checkpoint_directory(tmp_path: Path) -> None:
    real_target = tmp_path / "elsewhere"
    real_target.mkdir()
    checkpoint_link = tmp_path / "checkpoints"
    checkpoint_link.symlink_to(real_target, target_is_directory=True)

    with pytest.raises(ExperimentTrackingError, match="symbolic link"):
        ExperimentTracker(checkpoint_link, ["a"])


def test_new_tracker_has_no_completed_candidates(tmp_path: Path) -> None:
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
    tracker = ExperimentTracker(tmp_path, ["a"])

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
    tracker = ExperimentTracker(tmp_path, ["a"])

    with tracker.track("a") as recorder:
        recorder.set_metrics({"accuracy": 0.5})

    assert tracker.load_predictions("a") is None


def test_track_does_not_persist_a_candidate_whose_body_raises(tmp_path: Path) -> None:
    tracker = ExperimentTracker(tmp_path, ["a"])

    with pytest.raises(RuntimeError, match="boom"):
        with tracker.track("a") as recorder:
            recorder.set_metrics({"accuracy": 0.5})
            raise RuntimeError("boom")

    assert tracker.is_completed("a") is False
    assert tracker.completed_candidate_ids() == ()


def test_track_rejects_an_unknown_candidate_id(tmp_path: Path) -> None:
    tracker = ExperimentTracker(tmp_path, ["a"])

    with pytest.raises(ExperimentTrackingError, match="unknown candidate ID"):
        with tracker.track("not-a-candidate"):
            pass


def test_load_result_raises_for_a_candidate_with_no_checkpoint(tmp_path: Path) -> None:
    tracker = ExperimentTracker(tmp_path, ["a"])

    with pytest.raises(ExperimentTrackingError, match="no checkpointed result"):
        tracker.load_result("a")


def test_a_new_tracker_instance_resumes_prior_completions_from_disk(tmp_path: Path) -> None:
    first = ExperimentTracker(tmp_path, ["a", "b"])
    with first.track("a") as recorder:
        recorder.set_metrics({"accuracy": 0.8})

    resumed = ExperimentTracker(tmp_path, ["a", "b"])

    assert resumed.is_completed("a") is True
    assert resumed.is_completed("b") is False
    assert resumed.load_result("a").metrics == {"accuracy": 0.8}


def test_progress_reports_average_and_estimated_remaining_after_a_completion(
    tmp_path: Path,
) -> None:
    # Calls in order: constructor start, track() start, track() end, progress().
    clock = iter([0.0, 0.0, 3.0, 3.0])
    tracker = ExperimentTracker(tmp_path, ["a", "b"], monotonic=lambda: next(clock))

    with tracker.track("a") as recorder:
        recorder.set_metrics({"accuracy": 0.8})

    snapshot = tracker.progress()

    assert snapshot.completed_count == 1
    assert snapshot.remaining_count == 1
    assert snapshot.average_seconds_per_candidate == 3.0
    assert snapshot.estimated_remaining_seconds == 3.0


def test_finalize_refuses_when_candidates_remain_incomplete(tmp_path: Path) -> None:
    tracker = ExperimentTracker(tmp_path, ["a", "b"])
    with tracker.track("a") as recorder:
        recorder.set_metrics({"accuracy": 0.8})

    with pytest.raises(ExperimentTrackingError, match="1 candidate.*not yet completed"):
        tracker.finalize("accuracy")


def test_finalize_sorts_descending_by_default_and_writes_final_results(tmp_path: Path) -> None:
    tracker = ExperimentTracker(tmp_path, ["a", "b", "c"])
    for candidate_id, accuracy in (("a", 0.7), ("b", 0.95), ("c", 0.8)):
        with tracker.track(candidate_id) as recorder:
            recorder.set_metrics({"accuracy": accuracy})

    final_path = tracker.finalize("accuracy")

    assert final_path == tmp_path / "final-results.json"

    payload = json.loads(final_path.read_text(encoding="utf-8"))
    assert payload["ranking_metric"] == "accuracy"
    assert payload["higher_is_better"] is True
    assert [row["candidate_id"] for row in payload["results"]] == ["b", "c", "a"]


def test_finalize_sorts_ascending_when_lower_is_better(tmp_path: Path) -> None:
    tracker = ExperimentTracker(tmp_path, ["a", "b"])
    for candidate_id, loss in (("a", 0.5), ("b", 0.1)):
        with tracker.track(candidate_id) as recorder:
            recorder.set_metrics({"loss": loss})

    final_path = tracker.finalize("loss", higher_is_better=False)

    payload = json.loads(final_path.read_text(encoding="utf-8"))
    assert [row["candidate_id"] for row in payload["results"]] == ["b", "a"]


def test_no_leftover_temp_files_after_successful_writes(tmp_path: Path) -> None:
    tracker = ExperimentTracker(tmp_path, ["a"])
    with tracker.track("a") as recorder:
        recorder.set_metrics({"accuracy": 0.9})
        recorder.set_predictions(np.array([1, 0]))
    tracker.finalize("accuracy")

    leftover = list(tmp_path.rglob("*.tmp*"))
    assert leftover == []
