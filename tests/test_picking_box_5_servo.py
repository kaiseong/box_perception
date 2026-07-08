from __future__ import annotations

import sys
import types
import unittest
import time
from pathlib import Path

import numpy as np


class _FinishCode:
    Ok = "Ok"


sys.modules.setdefault(
    "rby1_sdk",
    types.SimpleNamespace(
        RobotCommandFeedback=types.SimpleNamespace(FinishCode=_FinishCode),
    ),
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import picking_box_5 as pb5


def rot2(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


class ServoControlTests(unittest.TestCase):
    def test_orbit_feedforward_cancels_rotation_induced_drift(self) -> None:
        # Pure yaw error, box straight ahead: the command must strafe so the
        # box stays fixed in the base frame while rotating (v = omega x r).
        box = np.array([0.45, 0.0])
        velocity, omega = pb5.servo_control_velocities(
            [0.0, 0.0], 10.0, box, max_speed_mps=1.0, max_angular_speed_radps=1.0
        )
        self.assertGreater(omega, 0.0)
        np.testing.assert_allclose(velocity, omega * np.array([box[1], -box[0]]), atol=1e-12)

        # Integrate the apparent box motion: d(p_b)/dt = -omega x p_b - v.
        p = box.copy()
        dt = 0.001
        for _ in range(1000):
            dp = -omega * np.array([-p[1], p[0]]) - velocity
            p = p + dp * dt
        np.testing.assert_allclose(p, box, atol=1e-3)

    def test_speed_caps_scale_v_and_omega_together(self) -> None:
        box = np.array([0.45, 0.0])
        v_free, w_free = pb5.servo_control_velocities(
            [0.10, 0.0], 20.0, box, max_speed_mps=10.0, max_angular_speed_radps=10.0
        )
        v_capped, w_capped = pb5.servo_control_velocities(
            [0.10, 0.0], 20.0, box, max_speed_mps=0.04, max_angular_speed_radps=0.10
        )
        self.assertLessEqual(float(np.linalg.norm(v_capped)), 0.04 + 1e-9)
        self.assertLessEqual(abs(w_capped), 0.10 + 1e-9)
        # Joint scaling keeps the v/omega geometry (same direction ratios).
        np.testing.assert_allclose(
            v_capped / w_capped, v_free / w_free, atol=1e-9
        )

    def test_closed_loop_simulation_converges_from_offset_and_yaw(self) -> None:
        # World-fixed box; base SE(2) pose integrated from the servo commands.
        # This catches any self-inconsistent sign in the control law.
        box_world = np.array([0.55, 0.08])
        axis_world_deg = 70.0  # box long axis in world; target: 90 deg in base
        base_t = np.array([0.0, 0.0])
        base_theta = 0.0
        dt = 0.1

        for _ in range(600):
            p_b = rot2(-base_theta) @ (box_world - base_t)
            axis_b_deg = axis_world_deg - np.degrees(base_theta)
            error_xy = pb5.mobile_base_alignment_error_xy(
                [p_b[0], p_b[1], 0.0], target_x_m=0.45
            )
            yaw_error = pb5.signed_angle_error_mod_180_deg(axis_b_deg, 90.0)
            velocity, omega = pb5.servo_control_velocities(
                error_xy,
                yaw_error,
                p_b,
                max_speed_mps=0.04,
                max_angular_speed_radps=0.10,
            )
            base_t = base_t + rot2(base_theta) @ velocity * dt
            base_theta = base_theta + omega * dt

        p_b = rot2(-base_theta) @ (box_world - base_t)
        final_error = pb5.mobile_base_alignment_error_xy([p_b[0], p_b[1], 0.0], target_x_m=0.45)
        final_yaw = pb5.signed_angle_error_mod_180_deg(axis_world_deg - np.degrees(base_theta), 90.0)
        self.assertLess(abs(float(final_error[0])), 0.005)
        self.assertLess(abs(float(final_error[1])), 0.005)
        self.assertLess(abs(float(final_yaw)), 1.0)

    def test_divergence_guard_trips_on_growing_error_only(self) -> None:
        guard = pb5.ServoDivergenceGuard(window_sec=2.0, growth_ratio=1.3, grace_sec=1.0)
        # Shrinking error: never trips.
        for i in range(60):
            self.assertFalse(guard.update(i * 0.1, 10.0 / (1.0 + i * 0.1)))

        # Growing error (wrong-sign feedback look): trips after grace+window.
        guard = pb5.ServoDivergenceGuard(window_sec=2.0, growth_ratio=1.3, grace_sec=1.0)
        tripped_at = None
        for i in range(80):
            t = i * 0.1
            if guard.update(t, 2.0 + t * 2.0):
                tripped_at = t
                break
        self.assertIsNotNone(tripped_at)
        self.assertLess(tripped_at, 5.0)

    def test_divergence_guard_ignores_single_low_expired_noise_reference(self) -> None:
        guard = pb5.ServoDivergenceGuard(window_sec=1.0, growth_ratio=1.3, grace_sec=0.0)
        samples = [
            (0.0, 10.0),
            (0.1, 9.0),
            (0.2, 0.1),  # one-frame low outlier; must not become the sole reference
            (0.3, 8.5),
            (1.21, 6.0),
            (1.31, 5.8),
            (1.41, 5.5),
        ]
        for t_sec, error in samples:
            self.assertFalse(guard.update(t_sec, error))

    def test_normalized_error_uses_worst_component(self) -> None:
        value = pb5.servo_normalized_error(
            [0.02, 0.0], 1.0, x_tolerance_m=0.01, y_tolerance_m=0.01, yaw_tolerance_deg=4.0
        )
        self.assertAlmostEqual(value, 2.0)

    def test_measurement_filter_rejects_single_servo_outlier(self) -> None:
        filt = pb5.ServoMeasurementFilter(
            window_frames=3,
            center_outlier_m=0.03,
            yaw_outlier_deg=10.0,
        )
        self.assertIsNotNone(filt.update([0.50, 0.020, 0.0], 5.0, target_x_m=0.45))
        self.assertIsNotNone(filt.update([0.51, 0.021, 0.0], 4.0, target_x_m=0.45))
        self.assertIsNone(filt.update([0.70, 0.200, 0.0], 35.0, target_x_m=0.45))
        result = filt.update([0.49, 0.019, 0.0], 6.0, target_x_m=0.45)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result["center_base_m"][:2], [0.50, 0.020], atol=1e-12)
        np.testing.assert_allclose(result["error_xy_m"], [0.05, 0.020], atol=1e-12)
        self.assertAlmostEqual(float(result["yaw_error_deg"]), 5.0)

    def test_measurement_filter_recovers_from_base_motion_latch_up(self) -> None:
        # Hardware failure mode: the base moves during servoing, the median of
        # ACCEPTED samples goes stale, and every fresh (correct) sample gets
        # rejected forever. Sustained rejections must reset the window.
        filt = pb5.ServoMeasurementFilter(
            window_frames=3,
            center_outlier_m=0.03,
            yaw_outlier_deg=10.0,
            rejection_reset_frames=3,
        )
        self.assertIsNotNone(filt.update([0.60, 0.050, 0.0], 8.0, target_x_m=0.45))
        self.assertIsNotNone(filt.update([0.60, 0.049, 0.0], 8.0, target_x_m=0.45))
        # Base displaced >3cm relative to the stale window (reality moved).
        moved = [0.55, 0.010, 0.0]
        self.assertIsNone(filt.update(moved, 4.0, target_x_m=0.45))
        self.assertIsNone(filt.update(moved, 4.0, target_x_m=0.45))
        result = filt.update(moved, 4.0, target_x_m=0.45)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result["center_base_m"][:2], moved[:2], atol=1e-12)
        # After the reset the filter keeps tracking normally.
        self.assertIsNotNone(filt.update([0.549, 0.011, 0.0], 3.8, target_x_m=0.45))

    def test_measurement_filter_still_rejects_isolated_spikes_between_good_frames(self) -> None:
        filt = pb5.ServoMeasurementFilter(
            window_frames=3,
            center_outlier_m=0.03,
            yaw_outlier_deg=10.0,
            rejection_reset_frames=3,
        )
        self.assertIsNotNone(filt.update([0.50, 0.020, 0.0], 5.0, target_x_m=0.45))
        self.assertIsNotNone(filt.update([0.51, 0.021, 0.0], 4.0, target_x_m=0.45))
        # Isolated spikes separated by good frames never accumulate to a reset.
        for _ in range(4):
            self.assertIsNone(filt.update([0.90, 0.300, 0.0], 40.0, target_x_m=0.45))
            self.assertIsNotNone(filt.update([0.505, 0.020, 0.0], 4.5, target_x_m=0.45))

    def test_command_streamer_uses_short_hold_and_stale_zero(self) -> None:
        class FakeStream:
            def __init__(self) -> None:
                self.commands: list[dict[str, object]] = []

            def send_command(self, command: dict[str, object]) -> None:
                self.commands.append(command)

        old_builder = pb5.build_mobile_base_velocity_command

        def fake_builder(
            linear_velocity_xy_mps,
            *,
            angular_velocity_radps: float,
            minimum_time: float,
            control_hold_time: float,
        ) -> dict[str, object]:
            return {
                "linear": np.asarray(linear_velocity_xy_mps, dtype=np.float64).copy(),
                "angular": float(angular_velocity_radps),
                "minimum_time": float(minimum_time),
                "hold": float(control_hold_time),
            }

        pb5.build_mobile_base_velocity_command = fake_builder
        try:
            stream = FakeStream()
            sender = pb5.MobileBaseServoCommandStreamer(
                stream,
                period_sec=0.01,
                hold_time_sec=0.05,
                stale_stop_sec=0.025,
            )
            sender.start()
            sender.update([0.10, 0.0], 0.20)
            time.sleep(0.02)
            sender.update([0.10, 0.0], 0.20)
            time.sleep(0.06)
            sender.stop(zero_repeats=1)
        finally:
            pb5.build_mobile_base_velocity_command = old_builder

        self.assertGreaterEqual(len(stream.commands), 3)
        self.assertTrue(all(abs(float(cmd["hold"]) - 0.05) < 1e-12 for cmd in stream.commands))
        nonzero = [
            cmd
            for cmd in stream.commands
            if np.linalg.norm(cmd["linear"]) > 1e-9 or abs(float(cmd["angular"])) > 1e-9
        ]
        self.assertTrue(nonzero)
        last = stream.commands[-1]
        self.assertLessEqual(float(np.linalg.norm(last["linear"])), 1e-12)
        self.assertAlmostEqual(float(last["angular"]), 0.0)

    def test_command_streamer_raises_instead_of_sending_zero_when_thread_is_stuck(self) -> None:
        class FakeStream:
            def __init__(self) -> None:
                self.commands: list[object] = []

            def send_command(self, command: object) -> None:
                self.commands.append(command)

        class StuckThread:
            def join(self, timeout: float | None = None) -> None:
                self.timeout = timeout

            def is_alive(self) -> bool:
                return True

        stream = FakeStream()
        sender = pb5.MobileBaseServoCommandStreamer(stream, period_sec=0.01)
        sender._thread = StuckThread()

        with self.assertRaises(TimeoutError):
            sender.stop(zero_repeats=3)
        self.assertEqual(stream.commands, [])

    def test_default_stale_stop_exceeds_orin_frame_jitter_budget(self) -> None:
        self.assertGreaterEqual(pb5.SERVO_COMMAND_STALE_STOP_SEC, 0.70)

    def test_default_command_hold_survives_orin_vision_gil_stall(self) -> None:
        # A 0.25s hold expired the stream on hardware: the Jetson vision
        # estimator holds the GIL ~0.3s per frame, starving the 30Hz sender.
        self.assertGreaterEqual(pb5.SERVO_COMMAND_HOLD_TIME_SEC, 1.0)
        # The stale watchdog must still fire while the hold keeps the stream
        # alive, so zeroing stays the faster of the two safety nets.
        self.assertLess(pb5.SERVO_COMMAND_STALE_STOP_SEC, pb5.SERVO_COMMAND_HOLD_TIME_SEC)


if __name__ == "__main__":
    unittest.main()
