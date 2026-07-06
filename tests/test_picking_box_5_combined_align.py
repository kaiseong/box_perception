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

import picking_box_5 as pb5


def axis_from_yaw_deg(deg: float) -> np.ndarray:
    rad = np.deg2rad(float(deg))
    return np.array([np.cos(rad), np.sin(rad), 0.0], dtype=np.float64)


def measurement(*, center: tuple[float, float, float], yaw_deg: float) -> dict:
    return {
        "center_base_m": np.asarray(center, dtype=np.float64),
        "camera_to_base": np.eye(4, dtype=np.float64),
        "long_axis_camera": axis_from_yaw_deg(yaw_deg),
    }


class PickingBox5CombinedAlignTests(unittest.TestCase):
    def test_combined_plan_suppresses_translation_when_yaw_is_coarse(self) -> None:
        plan = pb5.mobile_base_combined_alignment_plan(
            measurement(center=(0.52, 0.03, 0.0), yaw_deg=105.0),
            target_x_m=0.45,
            x_tolerance_m=0.01,
            y_tolerance_m=0.01,
            yaw_tolerance_deg=4.0,
            coarse_yaw_threshold_deg=8.0,
            max_step_m=0.03,
            max_step_deg=10.0,
        )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertFalse(plan["translation_enabled"])
        np.testing.assert_allclose(plan["step_xy_m"], [0.0, 0.0])
        self.assertAlmostEqual(plan["step_yaw_deg"], 10.0)
        np.testing.assert_allclose(plan["velocity_xy_mps"], [0.0, 0.0])
        self.assertGreater(plan["angular_velocity_radps"], 0.0)

    def test_combined_plan_moves_xy_and_yaw_together_near_alignment(self) -> None:
        plan = pb5.mobile_base_combined_alignment_plan(
            measurement(center=(0.48, 0.02, 0.0), yaw_deg=96.0),
            target_x_m=0.45,
            x_tolerance_m=0.01,
            y_tolerance_m=0.01,
            yaw_tolerance_deg=4.0,
            coarse_yaw_threshold_deg=8.0,
            max_step_m=0.03,
            max_step_deg=10.0,
        )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertTrue(plan["translation_enabled"])
        self.assertGreater(float(np.linalg.norm(plan["step_xy_m"])), 0.0)
        self.assertAlmostEqual(plan["step_yaw_deg"], 6.0)
        self.assertGreater(float(np.linalg.norm(plan["velocity_xy_mps"])), 0.0)
        self.assertGreater(plan["angular_velocity_radps"], 0.0)

    def test_combined_plan_marks_measurement_aligned_only_when_xy_and_yaw_pass(self) -> None:
        plan = pb5.mobile_base_combined_alignment_plan(
            measurement(center=(0.455, 0.005, 0.0), yaw_deg=93.0),
            target_x_m=0.45,
            x_tolerance_m=0.01,
            y_tolerance_m=0.01,
            yaw_tolerance_deg=4.0,
            coarse_yaw_threshold_deg=8.0,
        )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertTrue(plan["aligned"])
        np.testing.assert_allclose(plan["step_xy_m"], [0.0, 0.0])
        self.assertEqual(plan["step_yaw_deg"], 0.0)

    def test_xy_residual_status_keeps_target_band_stricter_than_safety_band(self) -> None:
        status = pb5.mobile_base_xy_residual_status(
            np.array([0.475, 0.0, 0.0], dtype=np.float64),
            target_x_m=0.45,
            x_tolerance_m=0.01,
            y_tolerance_m=0.01,
            max_residual_xy_m=0.05,
        )

        np.testing.assert_allclose(status["residual_xy_m"], [0.025, 0.0])
        self.assertTrue(status["within_safety_band"])
        self.assertFalse(status["within_target_band"])

    def test_combined_alignment_streams_xy_and_yaw_in_one_command_then_recaptures(self) -> None:
        initial = measurement(center=(0.48, 0.02, 0.0), yaw_deg=96.0)
        refreshed = measurement(center=(0.451, 0.002, 0.0), yaw_deg=91.0)
        stream_calls: list[tuple[np.ndarray, float, float]] = []

        original_stream = pb5.stream_mobile_base_velocity_stage
        original_capture = pb5.capture_live_box_measurement

        def fake_stream(_robot, velocity_xy, *, angular_velocity_radps, duration_sec, **_kwargs):
            stream_calls.append(
                (
                    np.asarray(velocity_xy, dtype=np.float64),
                    float(angular_velocity_radps),
                    float(duration_sec),
                )
            )
            return True, "ok"

        def fake_capture(*_args, **_kwargs):
            return refreshed

        pb5.stream_mobile_base_velocity_stage = fake_stream
        pb5.capture_live_box_measurement = fake_capture
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = pb5.run_mobile_base_combined_alignment(
                    object(),
                    object(),
                    object(),
                    "cw90",
                    initial,
                    visualize=False,
                    target_x_m=0.45,
                    x_tolerance_m=0.01,
                    y_tolerance_m=0.01,
                    yaw_tolerance_deg=4.0,
                    coarse_yaw_threshold_deg=8.0,
                    max_speed_mps=0.04,
                    max_angular_speed_radps=0.1,
                    max_step_m=0.03,
                    max_step_deg=10.0,
                    max_iterations=2,
                    total_timeout_sec=5.0,
                    xy_move_duration_sec=1.0,
                    yaw_move_duration_sec=1.0,
                    vision_frames_needed=1,
                    vision_timeout_sec=0.1,
                    max_center_spread_m=0.01,
                )
        finally:
            pb5.stream_mobile_base_velocity_stage = original_stream
            pb5.capture_live_box_measurement = original_capture

        self.assertIs(result, refreshed)
        self.assertEqual(len(stream_calls), 1)
        velocity_xy, angular_velocity, _duration = stream_calls[0]
        self.assertGreater(float(np.linalg.norm(velocity_xy)), 0.0)
        self.assertGreater(angular_velocity, 0.0)


if __name__ == "__main__":
    unittest.main()
