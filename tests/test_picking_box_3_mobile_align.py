from __future__ import annotations

import contextlib
import io
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

import picking_box_3 as pb3
from picking_box_3 import (
    MOBILE_BASE_MAX_SPEED_MPS,
    gripper_normalized_to_encoder_target,
    is_mobile_base_aligned,
    mobile_base_alignment_error_xy,
    mobile_base_alignment_step_xy,
    mobile_base_velocity_for_step_xy,
)


class PickingBox3MobileAlignTests(unittest.TestCase):
    def test_mobile_base_stream_matches_lerobot_command_shape(self) -> None:
        class FakeHeaderBuilder:
            def __init__(self) -> None:
                self.control_hold_time = None

            def set_control_hold_time(self, value: float):
                self.control_hold_time = float(value)
                return self

        class FakeSE2VelocityCommandBuilder:
            def __init__(self) -> None:
                self.header = None
                self.velocity = None
                self.angular = None
                self.minimum_time = None

            def set_command_header(self, header):
                self.header = header
                return self

            def set_velocity(self, velocity, angular):
                self.velocity = np.asarray(velocity, dtype=np.float64)
                self.angular = float(angular)
                return self

            def set_minimum_time(self, minimum_time: float):
                self.minimum_time = float(minimum_time)
                return self

        class FakeComponentBasedCommandBuilder:
            def __init__(self) -> None:
                self.mobility_command = None

            def set_mobility_command(self, command):
                self.mobility_command = command
                return self

        class FakeRobotCommandBuilder:
            def __init__(self) -> None:
                self.command = None

            def set_command(self, command):
                self.command = command
                return self

        class FakeStream:
            def __init__(self) -> None:
                self.sent = []

            def send_command(self, builder) -> None:
                self.sent.append(builder.command.mobility_command)

        class FakeRobot:
            def __init__(self) -> None:
                self.stream = FakeStream()
                self.priority = None

            def create_command_stream(self, *, priority: int):
                self.priority = priority
                return self.stream

        class FakeClock:
            def __init__(self) -> None:
                self.now = 0.0

            def monotonic(self) -> float:
                value = self.now
                self.now += 0.02
                return value

            def sleep(self, seconds: float) -> None:
                self.now += float(seconds)

        fake_rby = types.SimpleNamespace(
            CommandHeaderBuilder=FakeHeaderBuilder,
            SE2VelocityCommandBuilder=FakeSE2VelocityCommandBuilder,
            ComponentBasedCommandBuilder=FakeComponentBasedCommandBuilder,
            RobotCommandBuilder=FakeRobotCommandBuilder,
            RobotCommandFeedback=pb3.rby.RobotCommandFeedback,
        )
        robot = FakeRobot()
        clock = FakeClock()
        original_rby = pb3.rby
        original_monotonic = pb3.time.monotonic
        original_sleep = pb3.time.sleep
        pb3.rby = fake_rby
        pb3.time.monotonic = clock.monotonic
        pb3.time.sleep = clock.sleep
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ok, status = pb3.stream_mobile_base_velocity_stage(
                    robot,
                    [0.02, -0.01],
                    duration_sec=0.05,
                    stage="stage",
                    live_view=None,
                    dyn_model=object(),
                    dyn_state=object(),
                    view_rotation="cw90",
                    stream_period_sec=0.02,
                    control_hold_time_sec=0.25,
                )
        finally:
            pb3.rby = original_rby
            pb3.time.monotonic = original_monotonic
            pb3.time.sleep = original_sleep

        self.assertTrue(ok)
        self.assertEqual(status, "done")
        self.assertEqual(robot.priority, pb3.MOBILE_BASE_STREAM_PRIORITY)
        self.assertGreaterEqual(len(robot.stream.sent), 1 + pb3.MOBILE_BASE_STOP_REPEATS)

        first = robot.stream.sent[0]
        np.testing.assert_allclose(first.velocity, [0.02, -0.01])
        self.assertEqual(first.angular, 0.0)
        self.assertAlmostEqual(first.minimum_time, 0.02 * 1.01)
        self.assertAlmostEqual(first.header.control_hold_time, 0.25)

        for command in robot.stream.sent[-pb3.MOBILE_BASE_STOP_REPEATS :]:
            np.testing.assert_allclose(command.velocity, [0.0, 0.0])
            self.assertEqual(command.angular, 0.0)

    def test_mobile_base_stream_defaults_allow_d405_processing_slack(self) -> None:
        self.assertLessEqual(pb3.MOBILE_BASE_STREAM_PERIOD_SEC, 1.0 / 30.0 + 1e-9)
        self.assertGreaterEqual(pb3.MOBILE_BASE_COMMAND_HOLD_TIME_SEC, 2.0)

    def test_impedance_push_stream_ramps_target_to_full_inward_distance(self) -> None:
        class FakeHeaderBuilder:
            def __init__(self) -> None:
                self.control_hold_time = None

            def set_control_hold_time(self, value: float):
                self.control_hold_time = float(value)
                return self

        class FakeImpedanceControlCommandBuilder:
            def __init__(self) -> None:
                self.header = None
                self.reference_link_name = None
                self.link_name = None
                self.translation_weight = None
                self.rotation_weight = None
                self.transformation = None

            def set_command_header(self, header):
                self.header = header
                return self

            def set_reference_link_name(self, name: str):
                self.reference_link_name = name
                return self

            def set_link_name(self, name: str):
                self.link_name = name
                return self

            def set_translation_weight(self, weight):
                self.translation_weight = list(weight)
                return self

            def set_rotation_weight(self, weight):
                self.rotation_weight = list(weight)
                return self

            def set_transformation(self, transformation):
                self.transformation = np.asarray(transformation, dtype=np.float64)
                return self

        class FakeBodyComponentBasedCommandBuilder:
            def __init__(self) -> None:
                self.right_arm_command = None
                self.left_arm_command = None

            def set_right_arm_command(self, command):
                self.right_arm_command = command
                return self

            def set_left_arm_command(self, command):
                self.left_arm_command = command
                return self

        class FakeComponentBasedCommandBuilder:
            def __init__(self) -> None:
                self.body_command = None

            def set_body_command(self, command):
                self.body_command = command
                return self

        class FakeRobotCommandBuilder:
            def __init__(self) -> None:
                self.command = None

            def set_command(self, command):
                self.command = command
                return self

        class FakeStream:
            def __init__(self) -> None:
                self.sent = []

            def send_command(self, builder) -> None:
                self.sent.append(builder.command.body_command)

        class FakeRobot:
            def __init__(self) -> None:
                self.stream = FakeStream()
                self.priority = None

            def create_command_stream(self, *, priority: int):
                self.priority = priority
                return self.stream

        class FakeDynState:
            def set_q(self, q) -> None:
                self.q = np.asarray(q, dtype=np.float64)

        class FakeDynModel:
            def compute_forward_kinematics(self, _state) -> None:
                return None

            def compute_transformation(self, _state, _ref_index: int, ee_index: int):
                T = np.eye(4, dtype=np.float64)
                T[0, 3] = 0.4
                if ee_index == pb3.EE_RIGHT_INDEX:
                    T[1, 3] = -0.3
                elif ee_index == pb3.EE_LEFT_INDEX:
                    T[1, 3] = 0.3
                else:
                    raise AssertionError(f"unexpected ee_index={ee_index}")
                return T

        class FakeClock:
            def __init__(self) -> None:
                self.now = 0.0

            def monotonic(self) -> float:
                return self.now

            def sleep(self, seconds: float) -> None:
                self.now += max(0.0, float(seconds))

        fake_rby = types.SimpleNamespace(
            CommandHeaderBuilder=FakeHeaderBuilder,
            ImpedanceControlCommandBuilder=FakeImpedanceControlCommandBuilder,
            BodyComponentBasedCommandBuilder=FakeBodyComponentBasedCommandBuilder,
            ComponentBasedCommandBuilder=FakeComponentBasedCommandBuilder,
            RobotCommandBuilder=FakeRobotCommandBuilder,
            RobotCommandFeedback=pb3.rby.RobotCommandFeedback,
        )
        robot = FakeRobot()
        clock = FakeClock()
        original_rby = pb3.rby
        original_monotonic = pb3.time.monotonic
        original_sleep = pb3.time.sleep
        pb3.rby = fake_rby
        pb3.time.monotonic = clock.monotonic
        pb3.time.sleep = clock.sleep
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ok = pb3.stream_impedance_push_stage(
                    robot,
                    FakeDynModel(),
                    FakeDynState(),
                    np.zeros(1),
                    ramp_time_sec=0.06,
                    hold_time_sec=0.2,
                    stream_period_sec=0.02,
                    stage="stage",
                )
        finally:
            pb3.rby = original_rby
            pb3.time.monotonic = original_monotonic
            pb3.time.sleep = original_sleep

        self.assertTrue(ok)
        self.assertEqual(robot.priority, pb3.PUSH_STREAM_PRIORITY)
        self.assertGreaterEqual(len(robot.stream.sent), 4)

        first = robot.stream.sent[0]
        last = robot.stream.sent[-1]
        self.assertEqual(first.right_arm_command.reference_link_name, pb3.IMPEDANCE_REFERENCE_LINK)
        self.assertEqual(first.left_arm_command.reference_link_name, pb3.IMPEDANCE_REFERENCE_LINK)
        np.testing.assert_allclose(first.right_arm_command.transformation[1, 3], -0.3)
        np.testing.assert_allclose(first.left_arm_command.transformation[1, 3], 0.3)
        np.testing.assert_allclose(
            last.right_arm_command.transformation[1, 3],
            -0.3 + pb3.PUSH_DISTANCE,
        )
        np.testing.assert_allclose(
            last.left_arm_command.transformation[1, 3],
            0.3 - pb3.PUSH_DISTANCE,
        )
        self.assertAlmostEqual(
            first.right_arm_command.header.control_hold_time,
            pb3.PUSH_STREAM_COMMAND_HOLD_TIME,
        )
        self.assertAlmostEqual(last.right_arm_command.header.control_hold_time, 0.2)

    def test_gripper_stop_uses_bounded_join(self) -> None:
        class FakeThread:
            def __init__(self, alive: bool) -> None:
                self.alive = alive
                self.join_timeout = None

            def join(self, timeout=None) -> None:
                self.join_timeout = timeout

            def is_alive(self) -> bool:
                return self.alive

        stopped_thread = FakeThread(alive=False)
        gripper = object.__new__(pb3.MaxOpenGripper)
        gripper._running = True
        gripper._thread = stopped_thread
        gripper.stop()

        self.assertFalse(gripper._running)
        self.assertEqual(stopped_thread.join_timeout, pb3.GRIPPER_STOP_JOIN_TIMEOUT_SEC)
        self.assertIsNone(gripper._thread)

        stuck_thread = FakeThread(alive=True)
        gripper = object.__new__(pb3.MaxOpenGripper)
        gripper._running = True
        gripper._thread = stuck_thread
        with contextlib.redirect_stdout(io.StringIO()):
            gripper.stop()

        self.assertFalse(gripper._running)
        self.assertEqual(stuck_thread.join_timeout, pb3.GRIPPER_STOP_JOIN_TIMEOUT_SEC)
        self.assertIs(gripper._thread, stuck_thread)

    def test_gripper_homing_poll_interval_is_short(self) -> None:
        self.assertLessEqual(pb3.GRIPPER_HOMING_SLEEP_SEC, 0.05)

    def test_setup_gripper_retries_initialize_after_tool_voltage_settle(self) -> None:
        class FakeGripper:
            def __init__(self) -> None:
                self.initialize_calls: list[bool] = []
                self.homing_called = False
                self.open_target_called = False
                self.started = False
                self.min_q = np.array([0.0, 0.0])
                self.max_q = np.array([1.0, 1.0])

            def initialize(self, *, verbose: bool = True) -> bool:
                self.initialize_calls.append(verbose)
                return len(self.initialize_calls) >= 2

            def homing(self) -> bool:
                self.homing_called = True
                return True

            def set_open_target(self) -> bool:
                self.open_target_called = True
                return True

            def start(self) -> None:
                self.started = True

        original_gripper = pb3.MaxOpenGripper
        original_voltage = pb3.enable_gripper_tool_voltage
        original_sleep = pb3.time.sleep
        instances: list[FakeGripper] = []

        def make_gripper() -> FakeGripper:
            gripper = FakeGripper()
            instances.append(gripper)
            return gripper

        pb3.MaxOpenGripper = make_gripper
        pb3.enable_gripper_tool_voltage = lambda _robot: True
        pb3.time.sleep = lambda _seconds: None
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                gripper = pb3.setup_max_open_gripper(object())
        finally:
            pb3.MaxOpenGripper = original_gripper
            pb3.enable_gripper_tool_voltage = original_voltage
            pb3.time.sleep = original_sleep

        self.assertIs(gripper, instances[0])
        self.assertEqual(instances[0].initialize_calls, [True, True])
        self.assertTrue(instances[0].homing_called)
        self.assertTrue(instances[0].open_target_called)
        self.assertTrue(instances[0].started)

    def test_visual_abort_cancels_mobile_command_without_timeout_reason(self) -> None:
        class FakeCommand:
            def __init__(self) -> None:
                self.cancel_called = False

            def wait_for(self, _timeout_ms: int) -> bool:
                return False

            def cancel(self) -> None:
                self.cancel_called = True

        class FakeRobot:
            def __init__(self) -> None:
                self.command = FakeCommand()
                self.cancel_control_called = False

            def send_command(self, _builder):
                return self.command

            def get_state(self):
                return types.SimpleNamespace(position=np.zeros(1))

            def cancel_control(self) -> None:
                self.cancel_control_called = True

        class AbortLiveView:
            def process_next_frame(self, *, camera_to_base, status_lines=None):
                return False, "unused", None, np.eye(4), True

        original = pb3.compute_camera_to_base_for_view_rotation
        pb3.compute_camera_to_base_for_view_rotation = lambda *args, **kwargs: np.eye(4)
        robot = FakeRobot()
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                ok, status = pb3.send_mobile_stage_with_live_view(
                    robot,
                    object(),
                    "stage",
                    timeout_sec=1.0,
                    live_view=AbortLiveView(),
                    dyn_model=object(),
                    dyn_state=object(),
                    view_rotation="cw90",
                )
        finally:
            pb3.compute_camera_to_base_for_view_rotation = original

        self.assertFalse(ok)
        self.assertEqual(status, "visual_abort")
        self.assertTrue(robot.command.cancel_called)
        self.assertTrue(robot.cancel_control_called)
        text = output.getvalue()
        self.assertIn("visualization abort requested; canceling command", text)
        self.assertNotIn("TIMEOUT; canceling command", text)

    def test_collect_measurement_visual_abort_raises_operator_abort(self) -> None:
        class FakeRobot:
            def get_state(self):
                return types.SimpleNamespace(position=np.zeros(1))

        class AbortLiveView(pb3.ContinuousLiveBoxView):
            def __init__(self) -> None:
                pass

            def process_next_frame(self, *, camera_to_base, status_lines=None):
                return False, "unused", None, np.eye(4), True

        original = pb3.compute_camera_to_base_for_view_rotation
        pb3.compute_camera_to_base_for_view_rotation = lambda *args, **kwargs: np.eye(4)
        try:
            with self.assertRaises(pb3.UserAbortRequested):
                with contextlib.redirect_stdout(io.StringIO()):
                    AbortLiveView().collect_measurement(
                        FakeRobot(),
                        object(),
                        object(),
                        "cw90",
                        frames_needed=1,
                        timeout_sec=1.0,
                        max_center_spread_m=0.01,
                    )
        finally:
            pb3.compute_camera_to_base_for_view_rotation = original

    def test_gripper_open_normalized_maps_to_homed_open_endpoint(self) -> None:
        min_q = np.array([10.0, 20.0])
        max_q = np.array([110.0, 220.0])

        np.testing.assert_allclose(
            gripper_normalized_to_encoder_target([1.0, 1.0], min_q, max_q),
            min_q,
        )
        np.testing.assert_allclose(
            gripper_normalized_to_encoder_target([1.0, 1.0], min_q, max_q, direction=True),
            max_q,
        )
        np.testing.assert_allclose(
            gripper_normalized_to_encoder_target([2.0, -1.0], min_q, max_q),
            [10.0, 220.0],
        )

    def test_alignment_band_targets_45cm_x_and_zero_y(self) -> None:
        self.assertTrue(is_mobile_base_aligned([0.45, 0.0, 0.0]))
        self.assertTrue(is_mobile_base_aligned([0.44, -0.01, 0.0]))
        self.assertTrue(is_mobile_base_aligned([0.46, 0.01, 0.0]))
        self.assertFalse(is_mobile_base_aligned([0.461, 0.0, 0.0]))
        self.assertFalse(is_mobile_base_aligned([0.45, 0.011, 0.0]))

    def test_error_sign_matches_desired_base_motion_direction(self) -> None:
        np.testing.assert_allclose(mobile_base_alignment_error_xy([0.50, 0.02, 0.0]), [0.05, 0.02])
        np.testing.assert_allclose(mobile_base_alignment_error_xy([0.42, -0.03, 0.0]), [-0.03, -0.03])

    def test_step_is_zero_inside_tolerance_and_clipped_by_norm(self) -> None:
        np.testing.assert_allclose(mobile_base_alignment_step_xy([0.455, 0.005, 0.0]), [0.0, 0.0])

        step = mobile_base_alignment_step_xy([0.51, 0.04, 0.0], max_step_m=0.03)
        self.assertLessEqual(float(np.linalg.norm(step)), 0.0300001)
        self.assertGreater(step[0], 0.0)
        self.assertGreater(step[1], 0.0)

        np.testing.assert_allclose(
            mobile_base_alignment_step_xy([0.41, 0.0, 0.0], max_step_m=0.03),
            [-0.03, 0.0],
        )

    def test_velocity_respects_duration_and_speed_limit(self) -> None:
        self.assertEqual(MOBILE_BASE_MAX_SPEED_MPS, 0.02)
        np.testing.assert_allclose(
            mobile_base_velocity_for_step_xy([0.03, 0.0], duration_sec=1.0),
            [0.02, 0.0],
        )
        np.testing.assert_allclose(
            mobile_base_velocity_for_step_xy([0.03, 0.0], duration_sec=1.0, max_speed_mps=0.05),
            [0.03, 0.0],
        )

        velocity = mobile_base_velocity_for_step_xy([0.10, 0.0], duration_sec=1.0, max_speed_mps=0.05)
        np.testing.assert_allclose(velocity, [0.05, 0.0])

    def test_move_plan_displacement_equals_step_under_speed_cap(self) -> None:
        # A 3 cm step at the 0.02 m/s cap must extend the duration to 1.5 s
        # instead of truncating the displacement to 2 cm.
        velocity, duration = pb3.mobile_base_move_plan(
            [0.03, 0.0], base_duration_sec=1.0, max_speed_mps=0.02
        )
        self.assertAlmostEqual(duration, 1.5)
        np.testing.assert_allclose(velocity * duration, [0.03, 0.0])
        self.assertLessEqual(float(np.linalg.norm(velocity)), 0.02 + 1e-9)

        # Small steps keep the base duration.
        velocity, duration = pb3.mobile_base_move_plan(
            [0.01, 0.0], base_duration_sec=1.0, max_speed_mps=0.02
        )
        self.assertAlmostEqual(duration, 1.0)
        np.testing.assert_allclose(velocity * duration, [0.01, 0.0])

    def test_alignment_budget_covers_realistic_initial_error(self) -> None:
        # 30 s at 0.02 m/s leaves room for ~20+ cm of travel plus vision time;
        # the old 8 s budget only allowed ~8 cm and fell back to arm-only picks.
        self.assertGreaterEqual(pb3.MOBILE_BASE_TOTAL_TIMEOUT_SEC, 30.0)
        self.assertEqual(pb3.MOBILE_BASE_MAX_SPEED_MPS, 0.02)
        self.assertLessEqual(pb3.MOBILE_BASE_MAX_RESIDUAL_XY_M, 0.05)

    def test_user_abort_exception_exists_for_alignment(self) -> None:
        self.assertTrue(issubclass(pb3.UserAbortRequested, Exception))


if __name__ == "__main__":
    unittest.main()
