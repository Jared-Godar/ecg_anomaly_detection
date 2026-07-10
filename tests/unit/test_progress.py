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
    """Verify that format elapsed seconds rounds to zero padded minutes and seconds.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        seconds: The seconds value supplied by the caller or surrounding test fixture.
        expected: The expected value supplied by the caller or surrounding test fixture.
    """

    assert format_elapsed_seconds(seconds) == expected


def test_reporter_with_no_stream_writes_nothing() -> None:
    """Verify that reporter with no stream writes nothing.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

    reporter = ProgressReporter(stream=None)

    reporter.header("run should not appear")
    # Scope `reporter.stage('acquisition', 1, 7)` here so the expected failure and fixture cleanup
    # stay scoped to this assertion.
    with reporter.stage("acquisition", 1, 7) as stage:
        stage.detail("irrelevant")


def test_stage_reports_start_and_completion_with_elapsed_and_detail() -> None:
    """Verify that stage reports start and completion with elapsed and detail.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

    clock = iter([0.0, 3.0])
    stream = io.StringIO()
    reporter = ProgressReporter(stream=stream, monotonic=lambda: next(clock))

    # Scope `reporter.stage('acquisition', 1, 7, detail='48 records')` here so the expected failure
    # and fixture cleanup stay scoped to this assertion.
    with reporter.stage("acquisition", 1, 7, detail="48 records") as stage:
        stage.detail("manifest written to artifacts/acquisition.json")

    lines = stream.getvalue().splitlines()
    assert lines == [
        "[1/7] acquisition: starting (48 records)",
        "[1/7] acquisition: complete in 00:03 (manifest written to artifacts/acquisition.json)",
    ]


def test_stage_without_detail_omits_parentheses() -> None:
    """Verify that stage without detail omits parentheses.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

    stream = io.StringIO()
    reporter = ProgressReporter(stream=stream, monotonic=lambda: 0.0)

    # Scope `reporter.stage('split', 4, 7)` here so the expected failure and fixture cleanup stay
    # scoped to this assertion.
    with reporter.stage("split", 4, 7):
        pass

    lines = stream.getvalue().splitlines()
    assert lines == [
        "[4/7] split: starting",
        "[4/7] split: complete in 00:00",
    ]


def test_stage_reports_failure_and_reraises_without_a_completion_detail() -> None:
    """Verify that stage reports failure and reraises without a completion detail.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

    stream = io.StringIO()
    reporter = ProgressReporter(stream=stream, monotonic=lambda: 0.0)

    # Scope `pytest.raises(ValueError, match='boom'), reporter.stage('training', 6, 7)` here so the
    # expected failure and fixture cleanup stay scoped to this assertion.
    with pytest.raises(ValueError, match="boom"), reporter.stage("training", 6, 7):
        raise ValueError("boom")

    lines = stream.getvalue().splitlines()
    assert lines == [
        "[6/7] training: starting",
        "[6/7] training: failed after 00:00",
    ]


def test_note_is_indented_beneath_stage_banners() -> None:
    """Verify that note is indented beneath stage banners.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

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
