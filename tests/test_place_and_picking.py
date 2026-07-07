from __future__ import annotations

import ast
import contextlib
import io
from pathlib import Path
import sys
import types
import unittest

import numpy as np


class _FinishCode:
    Ok = "Ok"


sys.modules.setdefault(
    "rby1_sdk",
    types.SimpleNamespace(
        RobotCommandFeedback=types.SimpleNamespace(FinishCode=_FinishCode),
    ),
)

import place_and_picking as pap


class PickingAndPlaceTests(unittest.TestCase):
    def test_script_is_independent_from_picking_box_5_imports(self) -> None:
        source = Path(pap.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)

        imported_modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)
        self.assertNotIn("picking_box_5", imported_modules)

    def test_default_cli_runs_place_regrasp_only_with_one_second_place_wait(self) -> None:
        args = pap.parse_args(["--address", "localhost:50051"])

        self.assertFalse(args.initial_pick)
        self.assertTrue(args.continue_pick)
        self.assertTrue(args.place_regrasp)
        self.assertTrue(args.transfer_confirm)
        self.assertAlmostEqual(args.place_wait_sec, 1.0)

    def test_initial_pick_cli_enables_live_vision_pick_path(self) -> None:
        args = pap.parse_args(["--address", "localhost:50051", "--initial-pick"])

        self.assertTrue(args.initial_pick)

    def test_push_stream_can_reverse_exact_push_distance(self) -> None:
        inward_targets: list[float] = []

        class FakeStream:
            def send_command(self, command) -> None:
                self.last_command = command

        class FakeRobot:
            def __init__(self) -> None:
                self.priority = None
                self.stream = FakeStream()

            def create_command_stream(self, *, priority: int):
                self.priority = priority
                return self.stream

        original_build = pap.build_impedance_push_command

        def fake_build(_dyn_model, _dyn_state, _q, *, inward: float, hold_time: float):
            inward_targets.append(float(inward))
            return {"inward": float(inward), "hold_time": float(hold_time)}

        robot = FakeRobot()
        pap.build_impedance_push_command = fake_build
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ok = pap.stream_impedance_push_stage(
                    robot,
                    object(),
                    object(),
                    np.zeros(1),
                    target_inward=-pap.PUSH_DISTANCE,
                    ramp_time_sec=0.0,
                    hold_time_sec=0.0,
                    stage="release",
                )
        finally:
            pap.build_impedance_push_command = original_build

        self.assertTrue(ok)
        self.assertEqual(robot.priority, pap.PUSH_STREAM_PRIORITY)
        self.assertEqual(inward_targets, [-pap.PUSH_DISTANCE])

    def test_place_regrasp_sequence_orders_lower_release_wait_regrasp_lift(self) -> None:
        calls: list[tuple[str, object]] = []

        class FakeState:
            position = np.zeros(1)

        class FakeRobot:
            def get_state(self) -> FakeState:
                return FakeState()

        original_send_stage = pap.send_stage
        original_stream_push = pap.stream_impedance_push_stage
        original_lower = pap.build_impedance_lower_command
        original_lift = pap.build_impedance_lift_command
        original_sleep = pap.time.sleep

        def fake_send_stage(_robot, builder, stage: str, *, timeout_sec: float) -> bool:
            calls.append(("send", stage, builder, float(timeout_sec)))
            return True

        def fake_stream_push(_robot, _dyn_model, _dyn_state, _q, **kwargs) -> bool:
            calls.append(("stream", kwargs["stage"], float(kwargs["target_inward"])))
            return True

        def fake_lower(_dyn_model, _dyn_state, _q):
            return "lower"

        def fake_lift(_dyn_model, _dyn_state, _q):
            return "lift"

        def fake_sleep(seconds: float) -> None:
            calls.append(("sleep", float(seconds)))

        pap.send_stage = fake_send_stage
        pap.stream_impedance_push_stage = fake_stream_push
        pap.build_impedance_lower_command = fake_lower
        pap.build_impedance_lift_command = fake_lift
        pap.time.sleep = fake_sleep
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ok = pap.perform_place_regrasp_sequence(
                    FakeRobot(),
                    object(),
                    object(),
                    push_ramp_time_sec=0.5,
                    place_wait_sec=1.0,
                    command_timeout_margin_sec=5.0,
                    min_command_timeout_sec=8.0,
                )
        finally:
            pap.send_stage = original_send_stage
            pap.stream_impedance_push_stage = original_stream_push
            pap.build_impedance_lower_command = original_lower
            pap.build_impedance_lift_command = original_lift
            pap.time.sleep = original_sleep

        self.assertTrue(ok)
        self.assertEqual(
            calls,
            [
                ("send", "9/13 place_lower", "lower", 105.5),
                ("stream", "10/13 release_push_reverse", -pap.PUSH_DISTANCE),
                ("sleep", 1.0),
                ("stream", "12/13 regrasp_push", pap.PUSH_DISTANCE),
                ("send", "13/13 regrasp_lift", "lift", 105.5),
            ],
        )

    def test_default_main_skips_live_vision_and_initial_pick_motion(self) -> None:
        calls: list[str] = []

        class FakeRobot:
            def connect(self) -> None:
                calls.append("connect")

            def is_connected(self) -> bool:
                return True

            def is_power_on(self, _power: str) -> bool:
                return True

            def is_servo_on(self, _pattern: str) -> bool:
                return True

            def reset_fault_control_manager(self) -> None:
                calls.append("reset_fault")

            def enable_control_manager(self) -> bool:
                return True

            def model(self) -> object:
                return types.SimpleNamespace(robot_joint_names=[])

            def get_dynamics(self) -> object:
                return types.SimpleNamespace(make_state=lambda _links, _joints: object())

        fake_rby = types.SimpleNamespace(create_robot=lambda _address, _model: FakeRobot())
        original_rby = pap.rby
        original_capture = pap.capture_live_box_measurement
        original_perform = pap.perform_place_regrasp_sequence
        original_send_stage = pap.send_stage

        def fail_capture(*_args, **_kwargs):
            raise AssertionError("default place/regrasp-only mode must not use live vision")

        def fake_perform(*_args, **_kwargs) -> bool:
            calls.append("perform_place_regrasp")
            return True

        def fail_send_stage(*_args, **_kwargs):
            raise AssertionError("default place/regrasp-only mode must not run initial pick joint motion")

        pap.rby = fake_rby
        pap.capture_live_box_measurement = fail_capture
        pap.perform_place_regrasp_sequence = fake_perform
        pap.send_stage = fail_send_stage
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ok = pap.main(
                    address="localhost:50051",
                    model="m",
                    power=".*",
                    initial_pick=False,
                    box_center_camera_m=None,
                    view_rotation="cw90",
                    approach_time=pap.VISION_APPROACH_MINIMUM_TIME,
                    hold_time=pap.VISION_APPROACH_HOLD_TIME,
                    max_reference_xy_shift_m=pap.VISION_APPROACH_MAX_REFERENCE_XY_SHIFT_M,
                    midpoint_offset_xy_m=(0.0, 0.0),
                    live_vision_frames_needed=pap.LIVE_VISION_FRAMES_NEEDED,
                    live_vision_timeout_sec=pap.LIVE_VISION_TIMEOUT_SEC,
                    live_center_spread_m=pap.LIVE_VISION_MAX_CENTER_SPREAD_M,
                    continue_pick=True,
                    place_regrasp=True,
                    transfer_confirm=True,
                    transfer_wait_sec=0.0,
                    place_wait_sec=1.0,
                    push_ramp_time_sec=pap.PUSH_RAMP_TIME,
                    visualize=False,
                    visualize_only=False,
                    discrete_align=False,
                    servo_kp_xy=pap.SERVO_KP_XY,
                    servo_kp_yaw=pap.SERVO_KP_YAW,
                    servo_settled_frames=pap.SERVO_SETTLED_FRAMES,
                    servo_timeout_sec=pap.SERVO_TOTAL_TIMEOUT_SEC,
                    servo_command_hold_time_sec=pap.SERVO_COMMAND_HOLD_TIME_SEC,
                    servo_command_stale_stop_sec=pap.SERVO_COMMAND_STALE_STOP_SEC,
                    servo_filter_window_frames=pap.SERVO_FILTER_WINDOW_FRAMES,
                    gripper_open=False,
                    mobile_base_yaw_align=pap.MOBILE_BASE_YAW_ALIGN_DEFAULT,
                    mobile_base_yaw_tolerance_deg=pap.MOBILE_BASE_YAW_TOLERANCE_DEG,
                    mobile_base_yaw_max_speed_radps=pap.MOBILE_BASE_YAW_MAX_SPEED_RADPS,
                    mobile_base_yaw_max_step_deg=pap.MOBILE_BASE_YAW_MAX_STEP_DEG,
                    mobile_base_yaw_max_iterations=pap.MOBILE_BASE_YAW_MAX_ITERATIONS,
                    mobile_base_yaw_total_timeout_sec=pap.MOBILE_BASE_YAW_TOTAL_TIMEOUT_SEC,
                    mobile_base_yaw_move_duration_sec=pap.MOBILE_BASE_YAW_MOVE_DURATION_SEC,
                    mobile_base_yaw_vision_frames_needed=pap.MOBILE_BASE_YAW_VISION_FRAMES_NEEDED,
                    mobile_base_yaw_vision_timeout_sec=pap.MOBILE_BASE_YAW_VISION_TIMEOUT_SEC,
                    mobile_base_combined_coarse_yaw_threshold_deg=pap.MOBILE_BASE_COMBINED_COARSE_YAW_THRESHOLD_DEG,
                    mobile_base_align=pap.MOBILE_BASE_ALIGN_DEFAULT,
                    mobile_base_target_x_m=pap.MOBILE_BASE_TARGET_X_M,
                    mobile_base_x_tolerance_m=pap.MOBILE_BASE_X_TOLERANCE_M,
                    mobile_base_y_tolerance_m=pap.MOBILE_BASE_Y_TOLERANCE_M,
                    mobile_base_max_speed_mps=pap.MOBILE_BASE_MAX_SPEED_MPS,
                    mobile_base_max_step_m=pap.MOBILE_BASE_MAX_STEP_M,
                    mobile_base_max_iterations=pap.MOBILE_BASE_MAX_ITERATIONS,
                    mobile_base_total_timeout_sec=pap.MOBILE_BASE_TOTAL_TIMEOUT_SEC,
                    mobile_base_move_duration_sec=pap.MOBILE_BASE_MOVE_DURATION_SEC,
                    mobile_base_vision_frames_needed=pap.MOBILE_BASE_VISION_FRAMES_NEEDED,
                    mobile_base_vision_timeout_sec=pap.MOBILE_BASE_VISION_TIMEOUT_SEC,
                    command_timeout_margin_sec=pap.COMMAND_TIMEOUT_MARGIN_SEC,
                    min_command_timeout_sec=pap.COMMAND_TIMEOUT_MIN_SEC,
                )
        finally:
            pap.rby = original_rby
            pap.capture_live_box_measurement = original_capture
            pap.perform_place_regrasp_sequence = original_perform
            pap.send_stage = original_send_stage

        self.assertTrue(ok)
        self.assertIn("perform_place_regrasp", calls)


if __name__ == "__main__":
    unittest.main()
