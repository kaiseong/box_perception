from __future__ import annotations

import io
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

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

    def test_mobile_servo_skips_stationary_confirm(self) -> None:
        source = Path(debug.__file__).read_text()
        servo_start = source.index("def run_mobile_base_visual_servo_alignment(")
        servo_end = source.index("def run_mobile_base_combined_alignment(", servo_start)
        servo_source = source[servo_start:servo_end]

        self.assertIn("skipping stationary confirm", servo_source)
        self.assertNotIn("time.sleep(0.3)", servo_source)

    def test_mobile_servo_handoff_keeps_stream_for_pre_push(self) -> None:
        source = Path(debug.__file__).read_text()
        servo_start = source.index("def run_mobile_base_visual_servo_alignment(")
        servo_end = source.index("def run_mobile_base_combined_alignment(", servo_start)
        servo_source = source[servo_start:servo_end]

        self.assertIn("def stop_thread", source)
        self.assertIn('"_handoff_stream"', servo_source)
        self.assertIn('"_handoff_live_view"', servo_source)
        self.assertIn("commander.stop_thread()", servo_source)

    def test_streamed_pre_push_uses_stream_and_live_vision_gate(self) -> None:
        source = Path(debug.__file__).read_text()
        builder_start = source.index("def build_streamed_vision_pre_push_command(")
        builder_end = source.index("def build_pose_command(", builder_start)
        builder_source = source[builder_start:builder_end]
        self.assertIn(".set_body_command(body)", builder_source)
        self.assertIn(".set_mobility_command(mobility)", builder_source)

        stage_start = source.index('print_stage("5/7 vision_pre_push", "building target")')
        stage_end = source.index("if not continue_pick:", stage_start)
        stage_source = source[stage_start:stage_end]
        handoff_start = stage_source.index("if handoff_stream is not None:")
        handoff_end = stage_source.index("else:", handoff_start)
        handoff_source = stage_source[handoff_start:handoff_end]

        self.assertIn("handoff_stream.send_command(command)", handoff_source)
        self.assertIn("wait_streamed_pre_push_live_vision", handoff_source)
        self.assertNotIn("send_stage", handoff_source)


if __name__ == "__main__":
    unittest.main()
