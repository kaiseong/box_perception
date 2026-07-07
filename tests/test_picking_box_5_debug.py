from __future__ import annotations

import io
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

sys.modules.setdefault("rby1_sdk", SimpleNamespace())

import picking_box_5_debug as debug


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


class PickingBox5DebugTests(unittest.TestCase):
    def test_stage_duration_printer_outputs_only_stage_seconds(self) -> None:
        output = io.StringIO()
        clock = FakeClock()
        timer = debug.StageDurationPrinter(output=output, clock=clock)

        timer.mark("2/7 live_vision", "capturing")
        clock.advance(0.25)
        timer.mark("2/7 live_vision", "done")
        clock.advance(0.75)
        timer.mark("3-4/7 mobile_base_se2_align", "start")
        clock.advance(1.5)
        timer.finish()

        self.assertEqual(
            output.getvalue().splitlines(),
            [
                "[timing] 2/7 live_vision: 1.000s",
                "[timing] 3-4/7 mobile_base_se2_align: 1.500s",
            ],
        )

    def test_debug_script_is_standalone(self) -> None:
        source = Path(debug.__file__).read_text()
        tree = ast.parse(source)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imports.append(node.module)

        self.assertNotIn("picking_box_5", imports)

    def test_print_stage_keeps_failures_visible_on_stderr(self) -> None:
        stderr = io.StringIO()
        with mock.patch.object(sys, "stderr", stderr):
            debug.print_stage("5/7 vision_pre_push", "FAILED; aborting")
        self.assertIn("[stage] 5/7 vision_pre_push | FAILED; aborting", stderr.getvalue())

    def test_print_stage_silences_normal_progress_messages(self) -> None:
        stderr = io.StringIO()
        with mock.patch.object(sys, "stderr", stderr):
            debug.print_stage("3-4/7 mobile_base_se2_align", "servoing error x=+1.0cm settled=0")
        self.assertEqual(stderr.getvalue(), "")

    def test_failure_keywords_cover_known_failure_messages(self) -> None:
        for message in (
            "FAILED: residual error exceeds safety band",
            "stop send failed (boom); calling robot.cancel_control()",
            "servo alignment aborted from the live view",
            "robot.cancel_control() requested",
        ):
            self.assertTrue(debug.stage_message_is_failure(message), message)
        for message in (
            "building target",
            "servo settled: error x=+0.3cm y=-0.2cm yaw=+0.5deg",
            "reached",
        ):
            self.assertFalse(debug.stage_message_is_failure(message), message)


if __name__ == "__main__":
    unittest.main()
