"""Deterministic, observational-only progress reporting for long-running local stages.

Output produced here is a human-facing convenience. It never influences pipeline
control flow, artifact contents, or evidence schemas, and a reporter with no
stream attached is a silent no-op.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Event, Lock, Thread
from time import perf_counter
from typing import TextIO

# How many completed units a bounded-unit phase must observe before its projected
# remaining duration is considered minimally stable. Below this threshold the
# projection would rest on one or two samples, so callers report an explicit
# "estimating..." warm-up state instead of an unstable number (#199).
DEFAULT_ESTIMATE_WARMUP_UNIT_COUNT = 3


def format_elapsed_seconds(seconds: float) -> str:
    """Render a non-negative duration as zero-padded ``MM:SS``."""
    total_seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


@dataclass(frozen=True, slots=True)
class UnitTimingSnapshot:
    """Measured timing evidence captured when one bounded work unit completes.

    ``unit_seconds`` and ``phase_elapsed_seconds`` are direct monotonic-clock
    observations. ``approx_remaining_seconds`` is the only inferred field and
    carries three deliberate states: ``None`` while the estimator is still
    warming up (too few samples for a defensible projection), ``0.0`` once the
    final unit completes (nothing remains, a fact rather than an estimate), and
    a positive projection otherwise.
    """

    unit_index: int
    total_units: int
    unit_seconds: float
    phase_elapsed_seconds: float
    approx_remaining_seconds: float | None


class UnitTimingEstimator:
    """Project qualified remaining time for a phase made of bounded, countable units.

    This is the shared timing/projection abstraction called for by #199: one
    tested home for per-unit duration measurement, phase elapsed time, and a
    current-run-only remaining-duration projection, so operations that report
    timing do not each grow a subtly different implementation. The projection
    method is deliberately simple and documented: mean completed-unit duration
    (phase elapsed divided by completed units) multiplied by the units still
    outstanding. A running mean keeps the estimate well-defined for fast reused
    units, slow outliers, and the final unit alike, and it never uses history
    from earlier runs. Output produced from these snapshots is observational
    only; the estimator never influences the observed operation.
    """

    __slots__ = (
        "_completed_units",
        "_monotonic",
        "_previous_mark",
        "_started_at",
        "_total_units",
        "_warmup_unit_count",
    )

    def __init__(
        self,
        total_units: int,
        monotonic: Callable[[], float] = perf_counter,
        warmup_unit_count: int = DEFAULT_ESTIMATE_WARMUP_UNIT_COUNT,
    ) -> None:
        """Start the phase clock for a known, fixed number of work units.

        Args:
            total_units: Positive count of units the phase will complete; the
                projection is only defensible when the denominator is known.
            monotonic: Clock used for every measurement; overridable in tests
                for deterministic timing instead of real wall-clock time.
            warmup_unit_count: Positive number of completed units required
                before a remaining-time projection is reported at all.
        """

        # A zero or negative unit count leaves the projection with no defensible
        # denominator; bool is excluded because it is an int subclass in Python.
        if isinstance(total_units, bool) or not isinstance(total_units, int) or total_units <= 0:
            raise ValueError("unit timing requires a positive total unit count")
        # A non-positive warm-up would report a projection built from zero samples.
        if (
            isinstance(warmup_unit_count, bool)
            or not isinstance(warmup_unit_count, int)
            or warmup_unit_count <= 0
        ):
            raise ValueError("unit timing requires a positive warm-up unit count")
        self._total_units = total_units
        self._warmup_unit_count = warmup_unit_count
        self._monotonic = monotonic
        # Construction marks the phase start; the caller is expected to create the
        # estimator immediately before the phase's first unit begins.
        self._started_at = monotonic()
        # The previous completion mark starts at the phase start so the first
        # unit's duration is measured from the beginning of the phase.
        self._previous_mark = self._started_at
        self._completed_units = 0

    def complete_unit(self) -> UnitTimingSnapshot:
        """Record one completed unit and return its measured/projected timing.

        Returns:
            A snapshot of this unit's measured duration, the phase's measured
            elapsed time, and the qualified remaining-time projection state.
        """

        # More completions than configured units would silently corrupt the
        # projection's denominator; fail loudly instead.
        if self._completed_units >= self._total_units:
            raise ValueError("unit timing received more completions than configured units")
        now = self._monotonic()
        # max(0.0, ...) guards against a caller-supplied clock that is not truly
        # monotonic; a negative duration would render as a nonsense estimate.
        unit_seconds = max(0.0, now - self._previous_mark)
        phase_elapsed_seconds = max(0.0, now - self._started_at)
        self._previous_mark = now
        self._completed_units += 1
        remaining_units = self._total_units - self._completed_units
        # Three deliberate projection states (see UnitTimingSnapshot's docstring):
        # a factual zero when nothing remains, an explicit warm-up before enough
        # samples exist, and the documented mean-based projection otherwise.
        if remaining_units == 0:
            approx_remaining_seconds: float | None = 0.0
        elif self._completed_units < self._warmup_unit_count:
            approx_remaining_seconds = None
        else:
            approx_remaining_seconds = (
                phase_elapsed_seconds / self._completed_units
            ) * remaining_units
        return UnitTimingSnapshot(
            unit_index=self._completed_units,
            total_units=self._total_units,
            unit_seconds=unit_seconds,
            phase_elapsed_seconds=phase_elapsed_seconds,
            approx_remaining_seconds=approx_remaining_seconds,
        )


def format_unit_timing_suffix(snapshot: UnitTimingSnapshot, *, unit_label: str) -> str:
    """Render one snapshot as a qualified `` | ...`` suffix for an existing progress line.

    The suffix deliberately distinguishes observation from inference: the unit
    and elapsed durations are measured values, while the remaining duration is
    always labeled ``approx.`` and appears either as an explicit ``estimating...``
    warm-up state or as a projection — never as a deadline or guarantee (#199).

    Args:
        snapshot: Timing evidence for the unit that just completed.
        unit_label: Operation-specific unit noun (for example ``"record"``), so
            shared formatting preserves each operation's own wording.

    Returns:
        A suffix like `` | record 00:14 | elapsed 01:09 | approx. remaining 09:54``,
        with ``estimating...`` in place of the projection during warm-up.
    """

    # None is the estimator's explicit warm-up state; communicate that the
    # estimate is still being established rather than emitting an unstable number.
    if snapshot.approx_remaining_seconds is None:
        remaining_text = "estimating..."
    else:
        remaining_text = format_elapsed_seconds(snapshot.approx_remaining_seconds)
    return (
        f" | {unit_label} {format_elapsed_seconds(snapshot.unit_seconds)}"
        f" | elapsed {format_elapsed_seconds(snapshot.phase_elapsed_seconds)}"
        f" | approx. remaining {remaining_text}"
    )


class StageHandle:
    """Mutable completion detail attached to one in-progress stage banner."""

    __slots__ = ("_detail",)

    def __init__(self) -> None:
        """Start with no completion detail attached; set later via detail()."""

        self._detail: str | None = None

    def detail(self, text: str) -> None:
        """Attach detail text shown on the stage's completion banner."""
        self._detail = text

    @property
    def current_detail(self) -> str | None:
        """Return the detail text most recently attached via detail(), if any.

        Read by ProgressReporter.stage's `finally` block to build the completion
        banner's optional suffix, after the stage body has had a chance to call
        handle.detail(...) zero or more times.

        Returns:
            The most recently attached detail text, or None if detail() was never called.
        """

        return self._detail


class ProgressReporter:
    """Print concise stage banners and interim notes to an optional stream."""

    def __init__(
        self,
        stream: TextIO | None = None,
        monotonic: Callable[[], float] = perf_counter,
    ) -> None:
        """Attach an optional output stream and clock for stage timing.

        Args:
            stream: Where progress lines are written; None makes every write a silent
                no-op, matching this module's documented "observational only" contract.
            monotonic: Clock used for stage elapsed-time measurement; overridable in
                tests for deterministic timing instead of real wall-clock time.
        """

        self._stream = stream
        self._monotonic = monotonic
        # Heartbeats can write from a small daemon thread while the main thread is busy
        # in a long-running library call. Serialize print+flush so their lines cannot
        # interleave with a stage completion banner.
        self._write_lock = Lock()

    def header(self, message: str) -> None:
        """Emit one unindented line, for example a run identifier banner."""
        self._write(message)

    def note(self, message: str) -> None:
        """Emit one indented, free-form progress line inside a stage."""
        self._write(f"    {message}")

    @contextmanager
    def heartbeat(
        self,
        message: str,
        *,
        interval_seconds: float = 60.0,
        variability_note: str = "local completion time varies",
    ) -> Iterator[None]:
        """Emit restrained elapsed-time heartbeats while a blocking operation runs.

        The operation itself stays on the calling thread, preserving normal exception
        and keyboard-interrupt behavior. A daemon helper only waits and writes status
        lines; it never executes, retries, or alters the wrapped operation.

        Args:
            message: Stable operation label, optionally including its stage prefix.
            interval_seconds: Positive cadence between observational heartbeat lines.
            variability_note: Qualification explaining that elapsed time is not a
                guaranteed completion estimate.

        Yields:
            Control to the unchanged blocking operation being observed.
        """

        # A nonpositive wait would either reject every useful cadence or spin continuously.
        if interval_seconds <= 0:
            raise ValueError("heartbeat interval must be positive")
        # A reporter with no stream is intentionally silent. Avoid creating a thread in
        # that common library-use path because there is no output for it to produce.
        if self._stream is None:
            yield
            return

        stopped = Event()
        started_at = self._monotonic()

        def emit_periodically() -> None:
            """Wait for each cadence boundary and write elapsed, qualified status."""

            # Event.wait returns True immediately after the main thread signals stop;
            # False means a full interval elapsed and one heartbeat is due.
            while not stopped.wait(interval_seconds):
                elapsed = format_elapsed_seconds(self._monotonic() - started_at)
                self.header(f"{message}: still running after {elapsed} ({variability_note})")

        worker = Thread(
            target=emit_periodically,
            name="ecg-progress-heartbeat",
            daemon=True,
        )
        worker.start()
        # Always stop the helper, including when the observed operation raises or is interrupted.
        try:
            yield
        finally:
            # Signal and join before the enclosing stage prints its completion banner,
            # preventing a late heartbeat from appearing after "complete" or "failed".
            stopped.set()
            worker.join()

    @contextmanager
    def stage(
        self, name: str, index: int, total: int, detail: str | None = None
    ) -> Iterator[StageHandle]:
        """Report a named stage's start and completion with elapsed time."""
        prefix = f"[{index}/{total}] {name}"
        suffix = f" ({detail})" if detail else ""
        self._write(f"{prefix}: starting{suffix}")
        started_at = self._monotonic()
        handle = StageHandle()
        failed = False
        # Catch BaseException (not just Exception) so even a KeyboardInterrupt or
        # SystemExit during the stage body still produces a "failed after" completion
        # banner via the `finally` block below, before the exception re-propagates.
        try:
            yield handle
        except BaseException:
            failed = True
            raise
        finally:
            elapsed = format_elapsed_seconds(self._monotonic() - started_at)
            completion_suffix = f" ({handle.current_detail})" if handle.current_detail else ""
            status = "failed after" if failed else "complete in"
            self._write(f"{prefix}: {status} {elapsed}{completion_suffix}")

    def _write(self, line: str) -> None:
        """Write and flush one progress line when an output stream is configured.

        Args:
            line: The already-formatted line to write (header/note/stage banner).
        """

        # No stream means this reporter is a silent no-op, per this module's
        # documented "observational only" contract.
        if self._stream is None:
            return
        # Keep each complete line atomic relative to a background heartbeat write.
        with self._write_lock:
            print(line, file=self._stream)
            # Flush immediately: when this stream is the write end of a subprocess pipe
            # (as in the Step 0 notebook), Python fully block-buffers non-TTY stdout, so
            # without an explicit flush every banner would arrive in one batch at process
            # exit instead of live — defeating the point of reporting progress at all.
            self._stream.flush()
