"""Tests for observational-only stage progress reporting."""

from __future__ import annotations

import io
from threading import Event

import pytest

from ecg_anomaly_detection.progress import (
    ProgressReporter,
    UnitTimingEstimator,
    UnitTimingSnapshot,
    format_elapsed_seconds,
    format_unit_timing_suffix,
)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0.0, "00:00"), (59.4, "00:59"), (59.6, "01:00"), (222.0, "03:42")],
)
def test_format_elapsed_seconds_rounds_to_zero_padded_minutes_and_seconds(
    seconds: float, expected: str
) -> None:
    """format_elapsed_seconds zero-pads MM:SS and rounds to the nearest second (59.6 -> 01:00, not 00:60).

    Args:
        seconds: An elapsed-time value in seconds to format.
        expected: The zero-padded "MM:SS" string format_elapsed_seconds must return.
    """

    assert format_elapsed_seconds(seconds) == expected


def test_reporter_with_no_stream_writes_nothing() -> None:
    """A ProgressReporter constructed with stream=None accepts every call silently, without erroring.

    This is the opt-out path: code that always calls into the reporter must
    still work when the caller doesn't want progress output at all.
    """

    reporter = ProgressReporter(stream=None)

    reporter.header("run should not appear")
    # stream=None above, so this stage's banners and detail must be silently discarded.
    with reporter.stage("acquisition", 1, 7) as stage:
        stage.detail("irrelevant")


def test_stage_reports_start_and_completion_with_elapsed_and_detail() -> None:
    """A stage prints a "starting" banner with its start-time detail and a "complete" banner
    with the elapsed time and its own completion detail.

    A fake two-tick clock (0.0 then 3.0) makes the elapsed time deterministic
    ("00:03") rather than dependent on real wall-clock timing.
    """

    clock = iter([0.0, 3.0])
    stream = io.StringIO()
    reporter = ProgressReporter(stream=stream, monotonic=lambda: next(clock))

    # The fake clock advances from 0.0 to 3.0 across this stage's lifetime.
    with reporter.stage("acquisition", 1, 7, detail="48 records") as stage:
        stage.detail("manifest written to artifacts/acquisition.json")

    lines = stream.getvalue().splitlines()
    assert lines == [
        "[1/7] acquisition: starting (48 records)",
        "[1/7] acquisition: complete in 00:03 (manifest written to artifacts/acquisition.json)",
    ]


def test_stage_without_detail_omits_parentheses() -> None:
    """A stage entered and exited with no detail text prints banners with no trailing parentheses."""

    stream = io.StringIO()
    reporter = ProgressReporter(stream=stream, monotonic=lambda: 0.0)

    # No detail is provided to this stage or set within it.
    with reporter.stage("split", 4, 7):
        pass

    lines = stream.getvalue().splitlines()
    assert lines == [
        "[4/7] split: starting",
        "[4/7] split: complete in 00:00",
    ]


def test_stage_reports_failure_and_reraises_without_a_completion_detail() -> None:
    """A stage whose body raises prints a "failed after <elapsed>" banner and lets the exception propagate.

    The failure banner intentionally has no completion-detail parenthetical
    (unlike the success banner), since the caller never reached the point of
    supplying one.
    """

    stream = io.StringIO()
    reporter = ProgressReporter(stream=stream, monotonic=lambda: 0.0)

    # The stage body below raises before calling stage.detail(...).
    with pytest.raises(ValueError, match="boom"), reporter.stage("training", 6, 7):
        raise ValueError("boom")

    lines = stream.getvalue().splitlines()
    assert lines == [
        "[6/7] training: starting",
        "[6/7] training: failed after 00:00",
    ]


def test_note_is_indented_beneath_stage_banners() -> None:
    """A standalone note() call is written with a fixed 4-space indent, visually nested under stage banners."""

    stream = io.StringIO()
    reporter = ProgressReporter(stream=stream, monotonic=lambda: 0.0)

    reporter.note("record 3/48 (102): 12 windows")

    assert stream.getvalue() == "    record 3/48 (102): 12 windows\n"


def test_each_line_is_flushed_immediately_so_a_subprocess_pipe_streams_live() -> None:
    """Regression test: piped (non-TTY) stdout is fully block-buffered by default.

    Without an explicit flush per line, a subprocess consumer (the Step 0 notebook)
    would receive every banner in one batch at process exit instead of live,
    silently defeating the "no longer appears frozen" purpose of this reporter.
    """
    stream = io.StringIO()
    snapshots: list[str] = []
    stream.flush = lambda: snapshots.append(stream.getvalue())  # type: ignore[method-assign]
    reporter = ProgressReporter(stream=stream, monotonic=lambda: 0.0)

    reporter.header("run starting")
    reporter.note("record 1/3")

    assert snapshots == ["run starting\n", "run starting\n    record 1/3\n"]


def test_heartbeat_reports_elapsed_status_without_running_the_operation_in_worker() -> None:
    """A blocking main-thread operation receives a qualified heartbeat at the chosen cadence.

    The signaling stream lets this test wait for an actual heartbeat write rather than
    sleeping for an assumed amount of scheduler time. A one-millisecond cadence keeps
    the unit test fast while production callers retain the default one-minute cadence.
    """

    heartbeat_written = Event()

    class SignalingStream(io.StringIO):
        """String stream that signals when the heartbeat text has been written."""

        def write(self, text: str) -> int:
            """Record text and signal when a heartbeat line reaches the stream.

            Args:
                text: String fragment written by ``print``.

            Returns:
                Number of characters accepted by ``io.StringIO.write``.
            """

            written = super().write(text)
            # Signal only on the semantic heartbeat fragment, not print's trailing newline.
            if "still running" in text:
                heartbeat_written.set()
            return written

    stream = SignalingStream()
    reporter = ProgressReporter(stream=stream)

    # The wrapped body stays on this test's main thread; only observational waiting and
    # output run in the reporter's daemon helper.
    with reporter.heartbeat("[2/3] fit model", interval_seconds=0.001):
        assert heartbeat_written.wait(timeout=1.0)

    lines = stream.getvalue().splitlines()
    assert lines
    assert all(
        line.startswith("[2/3] fit model: still running after ")
        and line.endswith("(local completion time varies)")
        for line in lines
    )


def test_heartbeat_fast_operation_and_silent_reporter_emit_nothing() -> None:
    """Completing before one interval prints nothing, and stream=None stays thread-free/silent."""

    stream = io.StringIO()
    reporter = ProgressReporter(stream=stream)

    # The body exits before the one-hour interval, so stop wakes the helper immediately
    # and no speculative progress line is emitted.
    with reporter.heartbeat("fast operation", interval_seconds=3600.0):
        pass
    # The silent reporter path should accept the same context-manager API without output.
    with ProgressReporter(stream=None).heartbeat("silent operation"):
        pass

    assert stream.getvalue() == ""


def test_heartbeat_rejects_nonpositive_interval() -> None:
    """A zero or negative cadence fails before starting an unusable busy-loop worker."""

    reporter = ProgressReporter(stream=io.StringIO())

    # Entering the heartbeat context performs validation; no operation body should run.
    with (
        pytest.raises(ValueError, match="heartbeat interval must be positive"),
        reporter.heartbeat("invalid", interval_seconds=0),
    ):
        raise AssertionError("unreachable heartbeat body")


def test_unit_timing_warms_up_then_projects_and_finishes_at_zero() -> None:
    """The estimator reports an explicit warm-up, then a mean-based projection, then zero.

    A fake clock (construction at 0.0, completions at fixed later ticks) makes every
    measured and projected value deterministic. With the default warm-up of three
    completed units, units one and two must decline to project, unit three projects
    mean-duration x remaining, and the final unit reports a factual zero remaining.
    """

    clock = iter([0.0, 10.0, 22.0, 30.0, 40.0, 52.0])
    estimator = UnitTimingEstimator(total_units=5, monotonic=lambda: next(clock))

    snapshots = [estimator.complete_unit() for _ in range(5)]

    # Measured values are direct clock observations: per-unit deltas and phase elapsed.
    assert [snapshot.unit_seconds for snapshot in snapshots] == [10.0, 12.0, 8.0, 10.0, 12.0]
    assert [snapshot.phase_elapsed_seconds for snapshot in snapshots] == [
        10.0,
        22.0,
        30.0,
        40.0,
        52.0,
    ]
    # Units 1-2 are inside the default three-unit warm-up: no projection yet.
    assert snapshots[0].approx_remaining_seconds is None
    assert snapshots[1].approx_remaining_seconds is None
    # Unit 3: mean 30/3 = 10 seconds/unit, 2 units remain -> 20 seconds projected.
    assert snapshots[2].approx_remaining_seconds == 20.0
    # Unit 4: mean 40/4 = 10 seconds/unit, 1 unit remains -> 10 seconds projected.
    assert snapshots[3].approx_remaining_seconds == 10.0
    # The final unit reports zero remaining as a fact, not an estimate.
    assert snapshots[4].approx_remaining_seconds == 0.0
    assert [snapshot.unit_index for snapshot in snapshots] == [1, 2, 3, 4, 5]
    assert all(snapshot.total_units == 5 for snapshot in snapshots)


def test_unit_timing_projection_stays_defined_for_fast_reused_units_and_outliers() -> None:
    """Near-zero (reused/cached) durations and one large outlier still yield a sane mean.

    Two instantaneous completions followed by one 60-second outlier produce a
    20-second mean; the projection must remain finite and well-defined rather than
    dividing by a per-unit zero or chasing the outlier alone.
    """

    clock = iter([0.0, 0.0, 0.0, 60.0])
    estimator = UnitTimingEstimator(total_units=4, monotonic=lambda: next(clock))

    snapshots = [estimator.complete_unit() for _ in range(3)]

    assert snapshots[0].unit_seconds == 0.0
    assert snapshots[1].unit_seconds == 0.0
    assert snapshots[2].unit_seconds == 60.0
    # Mean 60/3 = 20 seconds/unit with one unit remaining -> 20 seconds projected.
    assert snapshots[2].approx_remaining_seconds == 20.0


def test_unit_timing_final_unit_reports_zero_even_before_warmup_completes() -> None:
    """A phase shorter than the warm-up still ends on a factual zero-remaining line.

    With two total units and the default three-unit warm-up, the first unit must
    report the warm-up state and the second (final) unit must report zero remaining,
    keeping the "final record reports zero remaining" contract independent of warm-up.
    """

    clock = iter([0.0, 5.0, 9.0])
    estimator = UnitTimingEstimator(total_units=2, monotonic=lambda: next(clock))

    first = estimator.complete_unit()
    second = estimator.complete_unit()

    assert first.approx_remaining_seconds is None
    assert second.approx_remaining_seconds == 0.0


def test_unit_timing_rejects_invalid_configuration_and_overcounting() -> None:
    """Non-positive unit counts, non-positive warm-ups, and extra completions all fail loudly."""

    # A zero or negative total leaves the projection with no defensible denominator.
    with pytest.raises(ValueError, match="positive total unit count"):
        UnitTimingEstimator(total_units=0)
    # bool is an int subclass; True must not silently mean "one unit".
    with pytest.raises(ValueError, match="positive total unit count"):
        UnitTimingEstimator(total_units=True)
    # A non-positive warm-up would project from zero observed samples.
    with pytest.raises(ValueError, match="positive warm-up unit count"):
        UnitTimingEstimator(total_units=3, warmup_unit_count=0)

    estimator = UnitTimingEstimator(total_units=1, monotonic=lambda: 0.0)
    estimator.complete_unit()
    # A completion beyond the configured total would corrupt the projection math.
    with pytest.raises(ValueError, match="more completions than configured units"):
        estimator.complete_unit()


def test_format_unit_timing_suffix_qualifies_estimates_and_preserves_unit_wording() -> None:
    """The suffix labels inference as approx., shows explicit warm-up, and keeps the caller's noun."""

    projected = UnitTimingSnapshot(
        unit_index=5,
        total_units=48,
        unit_seconds=14.0,
        phase_elapsed_seconds=69.0,
        approx_remaining_seconds=594.0,
    )
    warming_up = UnitTimingSnapshot(
        unit_index=1,
        total_units=48,
        unit_seconds=14.2,
        phase_elapsed_seconds=14.2,
        approx_remaining_seconds=None,
    )

    assert format_unit_timing_suffix(projected, unit_label="record") == (
        " | record 00:14 | elapsed 01:09 | approx. remaining 09:54"
    )
    # None renders the explicit warm-up wording instead of an unstable number.
    assert format_unit_timing_suffix(warming_up, unit_label="record") == (
        " | record 00:14 | elapsed 00:14 | approx. remaining estimating..."
    )
    # Operation-specific unit nouns pass through so shared formatting never forces
    # unrelated operations into acquisition's "record" wording.
    assert format_unit_timing_suffix(projected, unit_label="shard").startswith(" | shard 00:14")
