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

import picking_box_4 as pb4


def axis_from_yaw_deg(deg: float) -> np.ndarray:
    rad = np.deg2rad(float(deg))
    return np.array([np.cos(rad), np.sin(rad), 0.0], dtype=np.float64)


class PickingBox4YawAlignTests(unittest.TestCase):
    def test_yaw_error_targets_box_long_axis_to_base_y_mod_180(self) -> None:
        self.assertAlmostEqual(pb4.box_long_axis_base_yaw_error_deg(axis_from_yaw_deg(90.0)), 0.0)
        self.assertAlmostEqual(pb4.box_long_axis_base_yaw_error_deg(axis_from_yaw_deg(270.0)), 0.0)
        self.assertAlmostEqual(pb4.box_long_axis_base_yaw_error_deg(axis_from_yaw_deg(100.0)), 10.0)
        self.assertAlmostEqual(pb4.box_long_axis_base_yaw_error_deg(axis_from_yaw_deg(80.0)), -10.0)
        self.assertAlmostEqual(pb4.box_long_axis_base_yaw_error_deg(axis_from_yaw_deg(0.0)), 90.0)

    def test_measurement_yaw_error_uses_camera_to_base_rotation(self) -> None:
        camera_to_base = np.eye(4, dtype=np.float64)
        measurement = {
            "camera_to_base": camera_to_base,
            "long_axis_camera": axis_from_yaw_deg(87.0),
        }

        self.assertAlmostEqual(pb4.mobile_base_yaw_alignment_error_deg(measurement), -3.0)

    def test_yaw_step_applies_tolerance_and_step_limit(self) -> None:
        self.assertEqual(pb4.mobile_base_yaw_alignment_step_deg(2.5, tolerance_deg=3.0), 0.0)
        self.assertEqual(pb4.mobile_base_yaw_alignment_step_deg(12.0, max_step_deg=10.0), 10.0)
        self.assertEqual(pb4.mobile_base_yaw_alignment_step_deg(-12.0, max_step_deg=10.0), -10.0)

    def test_yaw_move_plan_preserves_signed_step_with_speed_cap(self) -> None:
        velocity, duration = pb4.mobile_base_yaw_move_plan(
            10.0,
            base_duration_sec=1.0,
            max_angular_speed_radps=0.1,
        )

        self.assertAlmostEqual(velocity, 0.1)
        self.assertAlmostEqual(velocity * duration, np.deg2rad(10.0))

        velocity, duration = pb4.mobile_base_yaw_move_plan(
            -5.0,
            base_duration_sec=2.0,
            max_angular_speed_radps=0.1,
        )
        self.assertLess(velocity, 0.0)
        self.assertAlmostEqual(duration, 2.0)
        self.assertAlmostEqual(velocity * duration, np.deg2rad(-5.0))

    def test_mobile_base_velocity_command_carries_angular_velocity(self) -> None:
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

        fake_rby = types.SimpleNamespace(
            CommandHeaderBuilder=FakeHeaderBuilder,
            SE2VelocityCommandBuilder=FakeSE2VelocityCommandBuilder,
            ComponentBasedCommandBuilder=FakeComponentBasedCommandBuilder,
            RobotCommandBuilder=FakeRobotCommandBuilder,
            RobotCommandFeedback=pb4.rby.RobotCommandFeedback,
        )
        original_rby = pb4.rby
        pb4.rby = fake_rby
        try:
            command = pb4.build_mobile_base_velocity_command(
                [0.01, -0.02],
                angular_velocity_radps=0.25,
                minimum_time=0.04,
                control_hold_time=0.5,
            )
        finally:
            pb4.rby = original_rby

        mobility = command.command.mobility_command
        np.testing.assert_allclose(mobility.velocity, [0.01, -0.02])
        self.assertAlmostEqual(mobility.angular, 0.25)
        self.assertAlmostEqual(mobility.minimum_time, 0.04)
        self.assertAlmostEqual(mobility.header.control_hold_time, 0.5)

    def test_yaw_alignment_returns_immediately_when_already_aligned(self) -> None:
        measurement = {
            "camera_to_base": np.eye(4, dtype=np.float64),
            "long_axis_camera": axis_from_yaw_deg(91.0),
        }

        with contextlib.redirect_stdout(io.StringIO()):
            result = pb4.run_mobile_base_yaw_alignment(
                object(),
                object(),
                object(),
                "cw90",
                measurement,
                visualize=False,
                tolerance_deg=3.0,
                max_angular_speed_radps=0.1,
                max_step_deg=10.0,
                max_iterations=2,
                total_timeout_sec=1.0,
                move_duration_sec=1.0,
                vision_frames_needed=1,
                vision_timeout_sec=0.1,
                max_center_spread_m=0.01,
            )

        self.assertIs(result, measurement)

    def test_yaw_alignment_streams_nonzero_angular_correction_then_recaptures(self) -> None:
        initial = {
            "camera_to_base": np.eye(4, dtype=np.float64),
            "long_axis_camera": axis_from_yaw_deg(100.0),
        }
        refreshed = {
            "camera_to_base": np.eye(4, dtype=np.float64),
            "long_axis_camera": axis_from_yaw_deg(91.0),
        }
        stream_calls: list[tuple[float, float]] = []

        original_stream = pb4.stream_mobile_base_velocity_stage
        original_capture = pb4.capture_live_box_measurement

        def fake_stream(_robot, velocity_xy, *, angular_velocity_radps, duration_sec, **_kwargs):
            np.testing.assert_allclose(velocity_xy, [0.0, 0.0])
            stream_calls.append((float(angular_velocity_radps), float(duration_sec)))
            return True, "ok"

        def fake_capture(*_args, **_kwargs):
            return refreshed

        pb4.stream_mobile_base_velocity_stage = fake_stream
        pb4.capture_live_box_measurement = fake_capture
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = pb4.run_mobile_base_yaw_alignment(
                    object(),
                    object(),
                    object(),
                    "cw90",
                    initial,
                    visualize=False,
                    tolerance_deg=3.0,
                    max_angular_speed_radps=0.1,
                    max_step_deg=10.0,
                    max_iterations=2,
                    total_timeout_sec=5.0,
                    move_duration_sec=1.0,
                    vision_frames_needed=1,
                    vision_timeout_sec=0.1,
                    max_center_spread_m=0.01,
                )
        finally:
            pb4.stream_mobile_base_velocity_stage = original_stream
            pb4.capture_live_box_measurement = original_capture

        self.assertIs(result, refreshed)
        self.assertEqual(len(stream_calls), 1)
        angular_velocity, duration = stream_calls[0]
        self.assertGreater(angular_velocity, 0.0)
        self.assertAlmostEqual(angular_velocity * duration, np.deg2rad(10.0))

    def test_yaw_alignment_fails_without_long_axis_before_contact(self) -> None:
        measurement = {
            "camera_to_base": np.eye(4, dtype=np.float64),
            "long_axis_camera": None,
        }

        with contextlib.redirect_stdout(io.StringIO()):
            result = pb4.run_mobile_base_yaw_alignment(
                object(),
                object(),
                object(),
                "cw90",
                measurement,
                visualize=False,
                tolerance_deg=3.0,
                max_angular_speed_radps=0.1,
                max_step_deg=10.0,
                max_iterations=2,
                total_timeout_sec=1.0,
                move_duration_sec=1.0,
                vision_frames_needed=1,
                vision_timeout_sec=0.1,
                max_center_spread_m=0.01,
            )

        self.assertIsNone(result)

    def test_pre_contact_yaw_gate_requires_measured_yaw_within_tolerance(self) -> None:
        aligned = {
            "camera_to_base": np.eye(4, dtype=np.float64),
            "long_axis_camera": axis_from_yaw_deg(91.0),
        }
        rotated = {
            "camera_to_base": np.eye(4, dtype=np.float64),
            "long_axis_camera": axis_from_yaw_deg(100.0),
        }
        missing = {
            "camera_to_base": np.eye(4, dtype=np.float64),
            "long_axis_camera": None,
        }

        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(pb4.verify_yaw_safe_before_contact(aligned, tolerance_deg=3.0))
            self.assertFalse(pb4.verify_yaw_safe_before_contact(rotated, tolerance_deg=3.0))
            self.assertFalse(pb4.verify_yaw_safe_before_contact(missing, tolerance_deg=3.0))

    def test_default_pre_contact_yaw_tolerance_is_four_degrees(self) -> None:
        inside_default = {
            "camera_to_base": np.eye(4, dtype=np.float64),
            "long_axis_camera": axis_from_yaw_deg(93.5),
        }
        outside_default = {
            "camera_to_base": np.eye(4, dtype=np.float64),
            "long_axis_camera": axis_from_yaw_deg(94.5),
        }

        self.assertAlmostEqual(pb4.MOBILE_BASE_YAW_TOLERANCE_DEG, 4.0)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(pb4.verify_yaw_safe_before_contact(inside_default))
            self.assertFalse(pb4.verify_yaw_safe_before_contact(outside_default))


if __name__ == "__main__":
    unittest.main()
