"""Checkpoint, resume, and progress helpers for long-running local experiment loops.

This is for ad hoc local experimentation (for example, hyperparameter search
notebooks under `notebooks/local/`) and is distinct from
`ecg_anomaly_detection.local_execution`, which manages governed
`run_pipeline()` output. Checkpoint artifacts written here are local,
disposable, and excluded from Git: never run manifests, benchmark artifacts,
model card evidence, or reproducibility evidence bundles.

Results and predictions are written atomically, one candidate at a time,
immediately after that candidate completes. An interruption at any point
loses at most the in-progress candidate, never previously completed work.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Centralize _CANDIDATE_ID_PATTERN so every caller shares the same documented invariant.
_CANDIDATE_ID_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?$")


class ExperimentTrackingError(ValueError):
    """Raised when a checkpoint directory or candidate identifier violates its contract."""


@dataclass(frozen=True, slots=True)
class CandidateResult:
    """One completed candidate's persisted metrics and timing."""

    candidate_id: str
    metrics: Mapping[str, float]
    duration_seconds: float
    completed_at_epoch: float


@dataclass(frozen=True, slots=True)
class ProgressSnapshot:
    """Point-in-time progress for an experiment loop."""

    completed_count: int
    remaining_count: int
    total_count: int
    elapsed_seconds: float
    average_seconds_per_candidate: float | None
    estimated_remaining_seconds: float | None


class CandidateRecorder:
    """Mutable metrics and predictions attached to one in-progress candidate."""

    __slots__ = ("_metrics", "_predictions")

    def __init__(self) -> None:
        """Initialize this object with the validated state required by its contract.

        The helper isolates this step so its assumptions, outputs, and failure behavior remain
        reviewable.
        """

        self._metrics: dict[str, float] = {}
        self._predictions: np.ndarray | None = None

    def set_metrics(self, metrics: Mapping[str, float]) -> None:
        """Attach this candidate's metrics, replacing any previously set value."""
        self._metrics = dict(metrics)

    def set_predictions(self, predictions: np.ndarray) -> None:
        """Attach this candidate's prediction array, replacing any previous value."""
        self._predictions = predictions


class ExperimentTracker:
    """Checkpoint, resume, and progress state for one local experiment loop."""

    def __init__(
        self,
        checkpoint_directory: Path,
        candidate_ids: Sequence[str],
        *,
        monotonic: Callable[[], float] = time.perf_counter,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Initialize this object with the validated state required by its contract.

        The helper isolates this step so its assumptions, outputs, and failure behavior remain
        reviewable.

        Args:
            checkpoint_directory: The checkpoint directory value supplied by the caller or surrounding test fixture.
            candidate_ids: The candidate ids value supplied by the caller or surrounding test fixture.
            monotonic: The monotonic value supplied by the caller or surrounding test fixture.
            clock: The clock value supplied by the caller or surrounding test fixture.
        """

        # Evaluate `not candidate_ids` explicitly so invalid or alternate states follow the
        # documented contract.
        if not candidate_ids:
            raise ExperimentTrackingError("candidate_ids must not be empty")
        # Iterate over `candidate_ids` one item at a time so ordering, validation, and failure
        # attribution remain explicit.
        for candidate_id in candidate_ids:
            _validate_candidate_id(candidate_id)
        # Evaluate `len(set(candidate_ids)) != len(candidate_ids)` explicitly so invalid or
        # alternate states follow the documented contract.
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ExperimentTrackingError("candidate_ids must be unique")
        # Evaluate `checkpoint_directory.is_symlink()` explicitly so invalid or alternate states
        # follow the documented contract.
        if checkpoint_directory.is_symlink():
            raise ExperimentTrackingError(
                f"checkpoint directory must not be a symbolic link: {checkpoint_directory}"
            )
        self._results_dir = checkpoint_directory / "results"
        self._predictions_dir = checkpoint_directory / "predictions"
        self._final_path = checkpoint_directory / "final-results.json"
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._predictions_dir.mkdir(parents=True, exist_ok=True)
        self._candidate_ids = tuple(candidate_ids)
        self._monotonic = monotonic
        self._clock = clock
        self._started_at = monotonic()
        self._session_durations: list[float] = []

    def is_completed(self, candidate_id: str) -> bool:
        """Return whether a candidate already has a checkpointed result."""
        _validate_candidate_id(candidate_id)
        return self._result_path(candidate_id).is_file()

    def completed_candidate_ids(self) -> tuple[str, ...]:
        """Return this tracker's candidate IDs that already have a checkpointed result."""
        return tuple(
            candidate_id for candidate_id in self._candidate_ids if self.is_completed(candidate_id)
        )

    def load_result(self, candidate_id: str) -> CandidateResult:
        """Load one candidate's previously checkpointed result."""
        path = self._result_path(candidate_id)
        # Evaluate `not path.is_file()` explicitly so invalid or alternate states follow the
        # documented contract.
        if not path.is_file():
            raise ExperimentTrackingError(f"no checkpointed result for candidate: {candidate_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return CandidateResult(
            candidate_id=payload["candidate_id"],
            metrics=payload["metrics"],
            duration_seconds=payload["duration_seconds"],
            completed_at_epoch=payload["completed_at_epoch"],
        )

    def load_predictions(self, candidate_id: str) -> np.ndarray | None:
        """Load one candidate's previously checkpointed predictions, if any were saved."""
        path = self._predictions_path(candidate_id)
        # Evaluate `not path.is_file()` explicitly so invalid or alternate states follow the
        # documented contract.
        if not path.is_file():
            return None
        return np.load(path, allow_pickle=False)

    @contextmanager
    def track(self, candidate_id: str) -> Iterator[CandidateRecorder]:
        """Time and checkpoint one candidate; nothing is persisted if the body raises."""
        _validate_candidate_id(candidate_id)
        # Evaluate `candidate_id not in self._candidate_ids` explicitly so invalid or alternate
        # states follow the documented contract.
        if candidate_id not in self._candidate_ids:
            raise ExperimentTrackingError(f"unknown candidate ID for this tracker: {candidate_id}")
        started_at = self._monotonic()
        recorder = CandidateRecorder()
        yield recorder
        duration_seconds = self._monotonic() - started_at
        # Evaluate `duration_seconds < 0` explicitly so invalid or alternate states follow the
        # documented contract.
        if duration_seconds < 0:
            raise ExperimentTrackingError("monotonic timer moved backwards")
        result = CandidateResult(
            candidate_id=candidate_id,
            metrics=dict(recorder._metrics),  # noqa: SLF001 - same-module cooperating class
            duration_seconds=duration_seconds,
            completed_at_epoch=self._clock(),
        )
        _atomic_write_json(
            {
                "candidate_id": result.candidate_id,
                "metrics": dict(result.metrics),
                "duration_seconds": result.duration_seconds,
                "completed_at_epoch": result.completed_at_epoch,
            },
            self._result_path(candidate_id),
        )
        # Evaluate `recorder._predictions is not None` explicitly so invalid or alternate states
        # follow the documented contract.
        if recorder._predictions is not None:  # noqa: SLF001 - same-module cooperating class
            _atomic_write_npy(recorder._predictions, self._predictions_path(candidate_id))  # noqa: SLF001
        self._session_durations.append(duration_seconds)

    def progress(self) -> ProgressSnapshot:
        """Return completed/remaining counts, elapsed time, and an ETA where practical."""
        completed_count = len(self.completed_candidate_ids())
        total_count = len(self._candidate_ids)
        elapsed_seconds = self._monotonic() - self._started_at
        average = (
            sum(self._session_durations) / len(self._session_durations)
            if self._session_durations
            else None
        )
        remaining_count = total_count - completed_count
        estimated_remaining = average * remaining_count if average is not None else None
        return ProgressSnapshot(
            completed_count=completed_count,
            remaining_count=remaining_count,
            total_count=total_count,
            elapsed_seconds=elapsed_seconds,
            average_seconds_per_candidate=average,
            estimated_remaining_seconds=estimated_remaining,
        )

    def finalize(self, ranking_metric: str, *, higher_is_better: bool = True) -> Path:
        """Write a deterministically sorted summary once every candidate has completed."""
        missing = [
            candidate_id
            for candidate_id in self._candidate_ids
            if not self.is_completed(candidate_id)
        ]
        # Evaluate `missing` explicitly so invalid or alternate states follow the documented
        # contract.
        if missing:
            raise ExperimentTrackingError(
                f"cannot finalize: {len(missing)} candidate(s) not yet completed: "
                + ", ".join(missing)
            )
        results = [self.load_result(candidate_id) for candidate_id in self._candidate_ids]
        results.sort(
            key=lambda result: result.metrics.get(ranking_metric, float("-inf")),
            reverse=higher_is_better,
        )
        payload = {
            "ranking_metric": ranking_metric,
            "higher_is_better": higher_is_better,
            "results": [
                {
                    "candidate_id": result.candidate_id,
                    "metrics": dict(result.metrics),
                    "duration_seconds": result.duration_seconds,
                    "completed_at_epoch": result.completed_at_epoch,
                }
                for result in results
            ],
        }
        _atomic_write_json(payload, self._final_path)
        return self._final_path

    def _result_path(self, candidate_id: str) -> Path:
        """Resolve result path for the documented repository workflow.

        The helper isolates this step so its assumptions, outputs, and failure behavior remain
        reviewable.

        Args:
            candidate_id: The candidate id value supplied by the caller or surrounding test fixture.

        Returns:
            The value produced by the documented operation.
        """

        return self._results_dir / f"{candidate_id}.json"

    def _predictions_path(self, candidate_id: str) -> Path:
        """Resolve predictions path for the documented repository workflow.

        The helper isolates this step so its assumptions, outputs, and failure behavior remain
        reviewable.

        Args:
            candidate_id: The candidate id value supplied by the caller or surrounding test fixture.

        Returns:
            The value produced by the documented operation.
        """

        return self._predictions_dir / f"{candidate_id}.npy"


def _validate_candidate_id(candidate_id: str) -> None:
    """Validate candidate id according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        candidate_id: The candidate id value supplied by the caller or surrounding test fixture.
    """

    # Evaluate `not isinstance(candidate_id, str) or not _CANDIDATE_ID_PATTERN.match(candidate_id)`
    # explicitly so invalid or alternate states follow the documented contract.
    if not isinstance(candidate_id, str) or not _CANDIDATE_ID_PATTERN.match(candidate_id):
        raise ExperimentTrackingError(
            f"candidate ID must be a non-empty filesystem-safe token: {candidate_id!r}"
        )


def _temp_path(path: Path) -> Path:
    """Construct temp path for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        path: Filesystem path identifying the input or output under review.

    Returns:
        The value produced by the documented operation.
    """

    return path.with_name(f"{path.stem}.tmp{path.suffix}")


def _atomic_write_json(payload: dict, path: Path) -> None:
    """Persist atomic write json for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        payload: The payload value supplied by the caller or surrounding test fixture.
        path: Filesystem path identifying the input or output under review.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temp_path(path)
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(path)


def _atomic_write_npy(array: np.ndarray, path: Path) -> None:
    """Persist atomic write npy for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        array: The array value supplied by the caller or surrounding test fixture.
        path: Filesystem path identifying the input or output under review.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temp_path(path)
    np.save(temp_path, array, allow_pickle=False)
    temp_path.replace(path)
