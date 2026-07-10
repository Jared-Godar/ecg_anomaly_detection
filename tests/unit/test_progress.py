"""Tests for observational-only stage progress reporting."""

from __future__ import annotations

import io

import pytest

from ecg_anomaly_detection.progress import ProgressReporter, format_elapsed_seconds


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
