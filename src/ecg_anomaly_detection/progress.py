"""Deterministic, observational-only progress reporting for long-running local stages.

Output produced here is a human-facing convenience. It never influences pipeline
control flow, artifact contents, or evidence schemas, and a reporter with no
stream attached is a silent no-op.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import TextIO


def format_elapsed_seconds(seconds: float) -> str:
    """Render a non-negative duration as zero-padded ``MM:SS``."""
    total_seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


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

    def header(self, message: str) -> None:
        """Emit one unindented line, for example a run identifier banner."""
        self._write(message)

    def note(self, message: str) -> None:
        """Emit one indented, free-form progress line inside a stage."""
        self._write(f"    {message}")

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
        print(line, file=self._stream)
        # Flush immediately: when this stream is the write end of a subprocess pipe
        # (as in the Step 0 notebook), Python fully block-buffers non-TTY stdout, so
        # without an explicit flush every banner would arrive in one batch at process
        # exit instead of live — defeating the point of reporting progress at all.
        self._stream.flush()
