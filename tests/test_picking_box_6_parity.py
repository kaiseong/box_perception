from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

import picking_box_5 as pb5
import picking_box_6 as pb6


class PickingBox6ParityTests(unittest.TestCase):
    def test_default_cli_values_match_picking_box_5(self) -> None:
        args5 = pb5.parse_args(["--address", "localhost:50051"])
        args6 = pb6.parse_args(["--address", "localhost:50051"])

        for name in (
            "model",
            "power",
            "view_rotation",
            "approach_time",
            "hold_time",
            "vision_pre_push_linear_velocity_limit",
            "vision_pre_push_angular_velocity_limit",
            "vision_pre_push_acceleration_limit_scaling",
            "max_reference_xy_shift_m",
            "live_vision_frames",
            "live_vision_timeout_sec",
            "live_center_spread_m",
            "gripper_open",
            "mobile_base_yaw_align",
            "mobile_base_yaw_tolerance_deg",
            "mobile_base_yaw_max_speed_radps",
            "mobile_base_yaw_max_step_deg",
            "mobile_base_yaw_max_iterations",
            "mobile_base_yaw_timeout_sec",
            "mobile_base_yaw_move_duration_sec",
            "mobile_base_yaw_vision_frames",
            "mobile_base_yaw_vision_timeout_sec",
            "mobile_base_combined_coarse_yaw_threshold_deg",
            "mobile_base_align",
            "mobile_base_target_x_m",
            "mobile_base_x_tolerance_m",
            "mobile_base_y_tolerance_m",
            "mobile_base_max_speed_mps",
            "mobile_base_max_step_m",
            "mobile_base_max_iterations",
            "mobile_base_timeout_sec",
            "mobile_base_move_duration_sec",
            "mobile_base_vision_frames",
            "mobile_base_vision_timeout_sec",
            "command_timeout_margin_sec",
            "min_command_timeout_sec",
            "discrete_align",
            "servo_kp_xy",
            "servo_kp_yaw",
            "servo_settled_frames",
            "servo_timeout_sec",
            "servo_command_hold_time_sec",
            "servo_command_stale_stop_sec",
            "servo_filter_window_frames",
            "continue_pick",
            "push_ramp_time_sec",
            "lift_target_output",
        ):
            self.assertEqual(getattr(args6, name), getattr(args5, name), name)

    def test_combined_alignment_plan_matches_picking_box_5(self) -> None:
        yaw_deg = 100.0
        yaw_rad = np.deg2rad(yaw_deg)
        measurement = {
            "center_base_m": np.array([0.51, -0.035, 1.18], dtype=np.float64),
            "camera_to_base": np.eye(4, dtype=np.float64),
            "long_axis_camera": np.array(
                [np.cos(yaw_rad), np.sin(yaw_rad), 0.0],
                dtype=np.float64,
            ),
        }

        plan5 = pb5.mobile_base_combined_alignment_plan(
            measurement,
            target_x_m=pb5.MOBILE_BASE_TARGET_X_M,
            x_tolerance_m=pb5.MOBILE_BASE_X_TOLERANCE_M,
            y_tolerance_m=pb5.MOBILE_BASE_Y_TOLERANCE_M,
            yaw_tolerance_deg=pb5.MOBILE_BASE_YAW_TOLERANCE_DEG,
            coarse_yaw_threshold_deg=pb5.MOBILE_BASE_COMBINED_COARSE_YAW_THRESHOLD_DEG,
            max_speed_mps=pb5.MOBILE_BASE_MAX_SPEED_MPS,
            max_angular_speed_radps=pb5.MOBILE_BASE_YAW_MAX_SPEED_RADPS,
            max_step_m=pb5.MOBILE_BASE_MAX_STEP_M,
            max_step_deg=pb5.MOBILE_BASE_YAW_MAX_STEP_DEG,
            xy_move_duration_sec=pb5.MOBILE_BASE_MOVE_DURATION_SEC,
            yaw_move_duration_sec=pb5.MOBILE_BASE_YAW_MOVE_DURATION_SEC,
        )
        plan6 = pb6.mobile_base_combined_alignment_plan(
            measurement,
            target_x_m=pb6.MOBILE_BASE_TARGET_X_M,
            x_tolerance_m=pb6.MOBILE_BASE_X_TOLERANCE_M,
            y_tolerance_m=pb6.MOBILE_BASE_Y_TOLERANCE_M,
            yaw_tolerance_deg=pb6.MOBILE_BASE_YAW_TOLERANCE_DEG,
            coarse_yaw_threshold_deg=pb6.MOBILE_BASE_COMBINED_COARSE_YAW_THRESHOLD_DEG,
            max_speed_mps=pb6.MOBILE_BASE_MAX_SPEED_MPS,
            max_angular_speed_radps=pb6.MOBILE_BASE_YAW_MAX_SPEED_RADPS,
            max_step_m=pb6.MOBILE_BASE_MAX_STEP_M,
            max_step_deg=pb6.MOBILE_BASE_YAW_MAX_STEP_DEG,
            xy_move_duration_sec=pb6.MOBILE_BASE_MOVE_DURATION_SEC,
            yaw_move_duration_sec=pb6.MOBILE_BASE_YAW_MOVE_DURATION_SEC,
        )

        self.assertEqual(plan6["aligned"], plan5["aligned"])
        np.testing.assert_allclose(plan6["error_xy_m"], plan5["error_xy_m"])
        np.testing.assert_allclose(plan6["velocity_xy_mps"], plan5["velocity_xy_mps"])
        self.assertAlmostEqual(plan6["yaw_error_deg"], plan5["yaw_error_deg"])
        self.assertAlmostEqual(plan6["angular_velocity_radps"], plan5["angular_velocity_radps"])
        self.assertAlmostEqual(plan6["duration_sec"], plan5["duration_sec"])

    def test_cli_validation_errors_match_picking_box_5(self) -> None:
        cases = [
            ["--address", "localhost:50051", "--command-timeout-margin-sec", "-1"],
            ["--address", "localhost:50051", "--mobile-base-yaw-max-iterations", "-1"],
            ["--address", "localhost:50051", "--mobile-base-vision-frames", "0"],
            ["--address", "localhost:50051", "--servo-filter-window-frames", "0"],
            ["--address", "localhost:50051", "--push-ramp-time-sec", "-0.1"],
        ]

        for argv in cases:
            args5 = pb5.parse_args(argv)
            args6 = pb6.parse_args(argv)
            with self.subTest(argv=argv):
                with self.assertRaises(SystemExit) as cm5:
                    pb5.run_cli(argv)
                with self.assertRaises(SystemExit) as cm6:
                    pb6.validate_cli_args(args6)
                # pb5 exits before robot connection for these argument errors.
                self.assertEqual(str(cm6.exception), str(cm5.exception))
                self.assertEqual(args6.address, args5.address)

    def test_lift_target_record_format_stays_compatible(self) -> None:
        right = np.eye(4)
        left = np.eye(4)
        left[1, 3] = 0.5

        record = pb6.lift_target_record(right, left)

        self.assertEqual(record["format_version"], pb5.LIFT_TARGET_RECORD_VERSION)
        self.assertEqual(record["reference_link"], "base")
        self.assertEqual(record["source"], pb6.SCRIPT_NAME)
        np.testing.assert_allclose(record["right_target"], right)
        np.testing.assert_allclose(record["left_target"], left)

    def test_script_identity_was_updated(self) -> None:
        source = Path(pb6.__file__).read_text(encoding="utf-8")

        self.assertEqual(pb6.SCRIPT_NAME, "picking_box_6")
        self.assertNotIn("picking_box_5", source)


if __name__ == "__main__":
    unittest.main()
