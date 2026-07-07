#!/usr/bin/env python3
"""Quiet timing wrapper for picking_box_5.

This script runs the same picking logic and CLI arguments as picking_box_5.py,
but suppresses normal operator logs. It prints only elapsed wall-clock time per
stage, measured from the first print_stage() call for a stage until the next
stage starts.
"""

from __future__ import annotations

import contextlib
import sys
import time
from typing import Any, Callable, TextIO

import picking_box_5 as picking


class StageDurationPrinter:
    """Convert picking_box_5 stage transitions into duration-only output."""

    def __init__(
        self,
        *,
        output: TextIO = sys.__stdout__,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._output = output
        self._clock = clock
        self._current_stage: str | None = None
        self._current_start: float | None = None

    def mark(self, stage: str, _message: str) -> None:
        now = self._clock()
        if self._current_stage != stage:
            self.finish(now)
            self._current_stage = stage
            self._current_start = now

    def finish(self, now: float | None = None) -> None:
        if self._current_stage is None or self._current_start is None:
            return
        end = self._clock() if now is None else float(now)
        elapsed = max(0.0, end - self._current_start)
        print(
            f"[timing] {self._current_stage}: {elapsed:.3f}s",
            file=self._output,
            flush=True,
        )
        self._current_stage = None
        self._current_start = None


class _NullWriter:
    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        return None


def _quiet_print(*_args: Any, **_kwargs: Any) -> None:
    return None


def main(argv: list[str] | None = None) -> int:
    cli_args = list(sys.argv[1:] if argv is None else argv)
    if "-h" in cli_args or "--help" in cli_args:
        return picking.run_cli(cli_args)

    timer = StageDurationPrinter()
    original_print_stage = picking.print_stage
    had_print = hasattr(picking, "print")
    original_print = getattr(picking, "print", None)

    picking.print_stage = timer.mark
    picking.print = _quiet_print
    try:
        with contextlib.redirect_stdout(_NullWriter()):
            return picking.run_cli(cli_args)
    finally:
        timer.finish()
        picking.print_stage = original_print_stage
        if had_print:
            picking.print = original_print
        else:
            delattr(picking, "print")


if __name__ == "__main__":
    raise SystemExit(main())
