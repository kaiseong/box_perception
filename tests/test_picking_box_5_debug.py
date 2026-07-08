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

    def test_print_stage_traces_every_message_to_stderr(self) -> None:
        # Keyword-filtered visibility silenced the servo's exit reason twice on
        # hardware; every stage event must reach stderr with a timestamp.
        stderr = io.StringIO()
        with mock.patch.object(sys, "stderr", stderr):
            debug.print_stage("5/7 vision_pre_push", "FAILED; aborting")
            debug.print_stage("3-4/7 mobile_base_se2_align", "servoing error x=+1.0cm settled=0")
        lines = stderr.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("5/7 vision_pre_push | FAILED; aborting", lines[0])
        self.assertIn("servoing error x=+1.0cm settled=0", lines[1])
        for line in lines:
            self.assertRegex(line, r"^\[stage \+\s*\d+\.\d{3}s\]")

    def test_print_stage_keeps_stdout_clean_for_timing(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(sys, "stdout", stdout), mock.patch.object(sys, "stderr", stderr):
            debug.print_stage("2/7 live_vision", "capturing")
        self.assertEqual(stdout.getvalue(), "")

    def test_servo_no_vision_abort_reports_unusable_breakdown(self) -> None:
        source = Path(debug.__file__).read_text()
        servo_start = source.index("def run_mobile_base_visual_servo_alignment(")
        servo_end = source.index("def run_mobile_base_combined_alignment(", servo_start)
        servo_source = source[servo_start:servo_end]
        self.assertIn("unusable breakdown=", servo_source)
        self.assertIn("filter_rejected", servo_source)
        self.assertIn("no_yaw_from_axis", servo_source)

    def test_mobile_servo_settled_path_hands_off_live_stream(self) -> None:
        source = Path(debug.__file__).read_text()
        servo_start = source.index("def run_mobile_base_visual_servo_alignment(")
        servo_end = source.index("def run_mobile_base_combined_alignment(", servo_start)
        servo_source = source[servo_start:servo_end]

        self.assertIn("commander.stop_thread()", servo_source)
        self.assertIn("STREAM_HANDOFF_BRIDGE_HOLD_SEC", servo_source)
        self.assertIn('settled_measurement["_handoff_stream"] = stream', servo_source)
        self.assertIn("skipping stationary confirm", servo_source)

    def test_pre_push_uses_new_stream_first_command_with_fk_gate_and_fallback(self) -> None:
        # E-pattern (probe-verified): pre-push starts as a NEW stream's first
        # body command so it preempts the servo bridge with no idle gap; FK
        # gates arrival and any failure falls back to the proven non-stream
        # path instead of aborting.
        source = Path(debug.__file__).read_text()
        stage_start = source.index('print_stage("5/7 vision_pre_push", "building target")')
        stage_end = source.index('print_stage("6/7 inward_push", "building ramped target stream")', stage_start)
        stage_source = source[stage_start:stage_end]

        handoff_start = stage_source.index("if handoff_stream is not None:")
        handoff_end = stage_source.index("if not streamed_pre_push_done:", handoff_start)
        handoff_source = stage_source[handoff_start:handoff_end]
        self.assertIn("robot.create_command_stream", handoff_source)
        self.assertIn("wait_streamed_eef_arrival", handoff_source)
        self.assertIn("robot.cancel_control()", handoff_source)
        self.assertNotIn("return done", handoff_source)
        self.assertNotIn("send_stage", handoff_source)
        self.assertIn("send_stage", stage_source)

    def test_push_stage_fk_verifies_engagement_and_retries_once(self) -> None:
        source = Path(debug.__file__).read_text()
        push_start = source.index("def stream_impedance_push_stage(")
        push_end = source.index("def build_impedance_lift_command(", push_start)
        push_source = source[push_start:push_end]
        self.assertIn("hands_gap_m", push_source)
        self.assertIn("PUSH_ENGAGE_MIN_GAP_SHRINK_M", push_source)
        self.assertIn("PUSH_ENGAGE_ATTEMPTS", push_source)
        self.assertIn("STREAMED_PUSH_FINAL_HOLD_SEC", push_source)
        self.assertIn("robot.cancel_control()", push_source)

    def test_lift_streams_first_command_with_fk_check_and_fallback(self) -> None:
        source = Path(debug.__file__).read_text()
        lift_start = source.index('print_stage("7/7 lift", "building target")')
        lift_end = source.index("except UserAbortRequested", lift_start)
        lift_source = source[lift_start:lift_end]
        self.assertIn("robot.create_command_stream", lift_source)
        self.assertIn("LIFT_ENGAGE_MIN_RAISE_FRACTION", lift_source)
        self.assertIn("holding box on stream", lift_source)
        # Fallback path must still exist and use the FinishCode wait.
        self.assertIn("send_stage", lift_source)

    def test_synced_sample_includes_q_for_handoff_conversion(self) -> None:
        # synced_servo_sample_to_measurement requires "q"; without it the
        # settled handoff silently fell back to the stationary-confirm path.
        source = Path(debug.__file__).read_text()
        grab_start = source.index("def grab_synced_estimate(")
        grab_end = source.index("def collect_measurement(", grab_start)
        grab_source = source[grab_start:grab_end]
        self.assertIn('"q": q,', grab_source)

    def test_handoff_stream_is_released_on_fallback_and_cleanup_paths(self) -> None:
        source = Path(debug.__file__).read_text()
        self.assertIn("def close_streamed_mobile_handoff(", source)
        # Discrete-align retry, yaw-gate abort, and main cleanup all release it.
        self.assertGreaterEqual(source.count("close_streamed_mobile_handoff("), 5)

if __name__ == "__main__":
    unittest.main()
