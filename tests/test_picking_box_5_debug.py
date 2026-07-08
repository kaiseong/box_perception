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

    def test_mobile_servo_settled_path_hands_off_live_stream(self) -> None:
        source = Path(debug.__file__).read_text()
        servo_start = source.index("def run_mobile_base_visual_servo_alignment(")
        servo_end = source.index("def run_mobile_base_combined_alignment(", servo_start)
        servo_source = source[servo_start:servo_end]

        self.assertIn("commander.stop_thread()", servo_source)
        self.assertIn("STREAM_HANDOFF_BRIDGE_HOLD_SEC", servo_source)
        self.assertIn('settled_measurement["_handoff_stream"] = stream', servo_source)
        self.assertIn("skipping stationary confirm", servo_source)

    def test_streamed_pre_push_rides_handoff_stream_with_fk_gate(self) -> None:
        source = Path(debug.__file__).read_text()
        stage_start = source.index('print_stage("5/7 vision_pre_push", "building target")')
        stage_end = source.index('print_stage("6/7 inward_push", "building ramped target stream")', stage_start)
        stage_source = source[stage_start:stage_end]
        handoff_start = stage_source.index("if handoff_stream is not None:")
        handoff_end = stage_source.index("if not streamed_pre_push_done:", handoff_start)
        handoff_source = stage_source[handoff_start:handoff_end]

        self.assertIn("build_streamed_vision_pre_push_command", handoff_source)
        self.assertIn("handoff_stream.send_command(streamed_command)", handoff_source)
        self.assertIn("wait_streamed_eef_arrival", handoff_source)
        self.assertNotIn("send_stage", handoff_source)
        # The composite must pair the arm targets with a held zero mobility.
        builder_start = source.index("def build_streamed_vision_pre_push_command(")
        builder_end = source.index("def build_pose_command(", builder_start)
        builder_source = source[builder_start:builder_end]
        self.assertIn(".set_body_command(body)", builder_source)
        self.assertIn(".set_mobility_command(mobility)", builder_source)

    def test_push_and_lift_reuse_handoff_stream_with_zero_mobility(self) -> None:
        source = Path(debug.__file__).read_text()
        self.assertIn("stream=handoff_stream,", source)

        push_start = source.index("def stream_impedance_push_stage(")
        push_end = source.index("def build_impedance_lift_command(", push_start)
        push_source = source[push_start:push_end]
        self.assertIn("zero_mobility_hold_sec=command_hold_time if shared_stream else None", push_source)
        self.assertIn("STREAMED_PUSH_FINAL_HOLD_SEC", push_source)

        lift_call_start = source.index('print_stage("7/7 lift", "building target")')
        lift_call_end = source.index("except UserAbortRequested", lift_call_start)
        lift_source = source[lift_call_start:lift_call_end]
        self.assertIn("zero_mobility_hold_sec=lift_stream_hold", lift_source)

    def test_handoff_stream_is_released_on_fallback_and_cleanup_paths(self) -> None:
        source = Path(debug.__file__).read_text()
        self.assertIn("def close_streamed_mobile_handoff(", source)
        # Discrete-align retry, yaw-gate abort, and main cleanup all release it.
        self.assertGreaterEqual(source.count("close_streamed_mobile_handoff("), 5)

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
