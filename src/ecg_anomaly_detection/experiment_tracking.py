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

# Candidate IDs become filesystem file stems (see _result_path/_predictions_path), so
# this pattern restricts them to characters safe across common filesystems: starts and
# ends with an alphanumeric, with only `_`, `.`, `-` permitted in between -- ruling out
# path separators, leading dots (hidden files), and trailing whitespace.
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
        """Start with no metrics and no predictions recorded yet."""

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
        """Validate the candidate set and prepare (or resume) a checkpoint directory.

        Calling this on a checkpoint_directory from a previous, interrupted run is the
        supported resume path: existing result/prediction files under it are left in
        place (mkdir uses exist_ok=True), so is_completed()/completed_candidate_ids()
        immediately reflect prior progress without any explicit "resume" step.

        Args:
            checkpoint_directory: Where results/ and predictions/ subdirectories are
                created (or reused) for this tracker's checkpoint files.
            candidate_ids: The complete, fixed set of candidates this tracker will
                track across its lifetime.
            monotonic: Clock used for duration/elapsed-time measurement; overridable
                in tests for deterministic timing instead of real wall-clock time.
            clock: Clock used for completed_at_epoch timestamps; overridable
                separately from `monotonic` since it serves a different purpose
                (a wall-clock timestamp, not an elapsed-duration measurement).
        """

        # An empty candidate set would make progress()/finalize() degenerate.
        if not candidate_ids:
            raise ExperimentTrackingError("candidate_ids must not be empty")
        # Validate every candidate ID up front, before any directory is touched, so a
        # malformed ID fails immediately rather than partway through a long run.
        for candidate_id in candidate_ids:
            _validate_candidate_id(candidate_id)
        # A duplicated ID would make is_completed/_result_path ambiguous about which
        # candidate a given checkpoint file actually belongs to.
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ExperimentTrackingError("candidate_ids must be unique")
        # Reject a symlinked checkpoint directory, so results/predictions can't be
        # silently written to (or resumed from) an unrelated location.
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
        # A candidate with no checkpoint file hasn't been tracked yet; loading it
        # would otherwise raise a bare FileNotFoundError from read_text below.
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
        # Predictions are optional (set_predictions may never be called for a
        # candidate); absence is a valid state, not an error.
        if not path.is_file():
            return None
        return np.load(path, allow_pickle=False)

    @contextmanager
    def track(self, candidate_id: str) -> Iterator[CandidateRecorder]:
        """Time and checkpoint one candidate; nothing is persisted if the body raises."""
        _validate_candidate_id(candidate_id)
        # A caller-supplied ID outside the fixed candidate_ids set from __init__ would
        # otherwise silently checkpoint a "candidate" this tracker was never told about.
        if candidate_id not in self._candidate_ids:
            raise ExperimentTrackingError(f"unknown candidate ID for this tracker: {candidate_id}")
        started_at = self._monotonic()
        recorder = CandidateRecorder()
        # No try/except here is deliberate: if the `with` block's body raises, nothing
        # below this yield ever executes, so no partial/incorrect checkpoint is
        # written -- matching this module's documented "nothing persisted if the body
        # raises" guarantee.
        yield recorder
        duration_seconds = self._monotonic() - started_at
        # A monotonic clock should never move backwards; if it does, the timer
        # implementation itself (or the injected fake in a test) is broken.
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
        # Predictions are optional per candidate; only write the .npy file if the
        # caller actually attached one via recorder.set_predictions.
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
        # Finalizing before every candidate has a checkpointed result would produce a
        # summary that silently omits some candidates from the ranking.
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
        """Build the checkpoint file path for one candidate's result JSON.

        Args:
            candidate_id: The candidate whose result path to build.

        Returns:
            The candidate's result checkpoint path under results/.
        """

        return self._results_dir / f"{candidate_id}.json"

    def _predictions_path(self, candidate_id: str) -> Path:
        """Build the checkpoint file path for one candidate's prediction array.

        Args:
            candidate_id: The candidate whose predictions path to build.

        Returns:
            The candidate's predictions checkpoint path under predictions/.
        """

        return self._predictions_dir / f"{candidate_id}.npy"


def _validate_candidate_id(candidate_id: str) -> None:
    """Confirm a candidate ID is a non-empty, filesystem-safe token.

    Every candidate ID becomes a checkpoint file stem (see _result_path,
    _predictions_path), so this check is what keeps a malicious or malformed ID from
    escaping the checkpoint directory (e.g. via `../`) or colliding with reserved
    filesystem names.

    Args:
        candidate_id: The candidate ID to validate.
    """

    # See _CANDIDATE_ID_PATTERN's own comment for exactly which characters this allows.
    if not isinstance(candidate_id, str) or not _CANDIDATE_ID_PATTERN.match(candidate_id):
        raise ExperimentTrackingError(
            f"candidate ID must be a non-empty filesystem-safe token: {candidate_id!r}"
        )


def _temp_path(path: Path) -> Path:
    """Build the temporary-file path used for one destination path's atomic write.

    Shared by _atomic_write_json and _atomic_write_npy: writing to this temp path
    first, then calling Path.replace (an atomic rename on POSIX and Windows), ensures
    a reader never observes a partially written file at `path`.

    Args:
        path: The final destination path this write is targeting.

    Returns:
        A sibling path with the same directory and suffix, named `<stem>.tmp<suffix>`.
    """

    return path.with_name(f"{path.stem}.tmp{path.suffix}")


def _atomic_write_json(payload: dict, path: Path) -> None:
    """Write a JSON payload atomically: write to a temp file, then rename into place.

    Args:
        payload: The JSON-serializable object to write.
        path: The final destination path.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temp_path(path)
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(path)


def _atomic_write_npy(array: np.ndarray, path: Path) -> None:
    """Write a numpy array atomically: write to a temp file, then rename into place.

    allow_pickle=False keeps checkpointed prediction arrays free of pickle
    deserialization risk, matching this package's numpy-loading convention elsewhere.

    Args:
        array: The array to write.
        path: The final destination path.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temp_path(path)
    np.save(temp_path, array, allow_pickle=False)
    temp_path.replace(path)
