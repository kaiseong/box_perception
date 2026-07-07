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
    def test_fast_default_timing_parameters_are_exposed_by_cli(self) -> None:
        args = pb5.parse_args(["--address", "localhost:50051"])

        self.assertEqual(args.live_vision_frames, 3)
        self.assertEqual(args.servo_settled_frames, 3)
        self.assertAlmostEqual(args.servo_kp_xy, 1.1)
        self.assertAlmostEqual(args.servo_kp_yaw, 1.3)
        self.assertAlmostEqual(args.mobile_base_max_speed_mps, 0.10)
        self.assertAlmostEqual(args.mobile_base_yaw_max_speed_radps, 0.4)
        self.assertAlmostEqual(args.vision_pre_push_linear_velocity_limit, 0.60)
        self.assertAlmostEqual(args.vision_pre_push_angular_velocity_limit, np.pi / 2)
        self.assertAlmostEqual(args.vision_pre_push_acceleration_limit_scaling, 0.80)

    def test_vision_pre_push_speed_limits_can_be_overridden(self) -> None:
        args = pb5.parse_args(
            [
                "--address",
                "localhost:50051",
                "--vision-pre-push-linear-velocity-limit",
                "0.45",
                "--vision-pre-push-angular-velocity-limit",
                "1.2",
                "--vision-pre-push-acceleration-limit-scaling",
                "0.6",
            ]
        )

        self.assertAlmostEqual(args.vision_pre_push_linear_velocity_limit, 0.45)
        self.assertAlmostEqual(args.vision_pre_push_angular_velocity_limit, 1.2)
        self.assertAlmostEqual(args.vision_pre_push_acceleration_limit_scaling, 0.6)

    def test_combined_plan_moves_xy_even_when_yaw_is_coarse_by_default(self) -> None:
        plan = pb5.mobile_base_combined_alignment_plan(
            measurement(center=(0.52, 0.03, 0.0), yaw_deg=105.0),
            target_x_m=0.45,
            x_tolerance_m=0.01,
            y_tolerance_m=0.01,
            yaw_tolerance_deg=4.0,
            max_step_m=0.03,
            max_step_deg=10.0,
        )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertTrue(plan["translation_enabled"])
        self.assertGreater(float(np.linalg.norm(plan["step_xy_m"])), 0.0)
        self.assertAlmostEqual(plan["step_yaw_deg"], 10.0)
        self.assertGreater(float(np.linalg.norm(plan["velocity_xy_mps"])), 0.0)
        self.assertGreater(plan["angular_velocity_radps"], 0.0)

    def test_combined_plan_can_suppress_translation_when_threshold_is_lowered(self) -> None:
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

    def test_se2_residual_status_reports_xy_fallback_need_when_yaw_is_ok(self) -> None:
        status = pb5.mobile_base_se2_residual_status(
            measurement(center=(0.475, 0.0, 0.0), yaw_deg=92.0),
            target_x_m=0.45,
            x_tolerance_m=0.01,
            y_tolerance_m=0.01,
            yaw_tolerance_deg=4.0,
        )

        self.assertTrue(status["yaw_within_target_band"])
        self.assertFalse(status["xy_status"]["within_target_band"])
        self.assertFalse(status["within_target_band"])
        np.testing.assert_allclose(status["xy_status"]["residual_xy_m"], [0.025, 0.0])

    def test_se2_residual_status_reports_yaw_fallback_need_when_xy_is_ok(self) -> None:
        status = pb5.mobile_base_se2_residual_status(
            measurement(center=(0.455, 0.0, 0.0), yaw_deg=99.0),
            target_x_m=0.45,
            x_tolerance_m=0.01,
            y_tolerance_m=0.01,
            yaw_tolerance_deg=4.0,
        )

        self.assertFalse(status["yaw_within_target_band"])
        self.assertTrue(status["xy_status"]["within_target_band"])
        self.assertFalse(status["within_target_band"])
        self.assertAlmostEqual(status["residual_yaw_deg"], 9.0)

    def test_live_center_cluster_ignores_single_outlier_candidate(self) -> None:
        centers = [
            [0.400, 0.000, 0.100],
            [0.404, 0.001, 0.101],
            [0.399, -0.001, 0.099],
            [0.520, 0.090, 0.100],
        ]
        axes = [axis_from_yaw_deg(91.0), axis_from_yaw_deg(92.0), None, axis_from_yaw_deg(20.0)]
        result, best_spread = pb5.select_stable_live_center_result(
            centers,
            axes,
            ["a", "b", "c", "outlier"],
            [False, False, True, False],
            frames_needed=3,
            max_center_spread_m=0.01,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertLessEqual(float(best_spread), 0.01)
        self.assertEqual(result["candidate_frames"], 4)
        self.assertEqual(result["frames_used"], 3)
        self.assertNotIn("outlier", result["modes"])
        np.testing.assert_allclose(result["center_camera_m"], [0.400, 0.000, 0.100], atol=0.002)
        self.assertTrue(result["long_axis_unconstrained"])

    def test_live_center_cluster_reports_unstable_candidates(self) -> None:
        result, best_spread = pb5.select_stable_live_center_result(
            [[0.0, 0.0, 0.0], [0.04, 0.0, 0.0], [0.08, 0.0, 0.0]],
            [None, None, None],
            ["a", "b", "c"],
            [False, False, False],
            frames_needed=3,
            max_center_spread_m=0.01,
        )

        self.assertIsNone(result)
        self.assertGreater(float(best_spread), 0.01)

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
