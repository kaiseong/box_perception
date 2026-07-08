from __future__ import annotations

import contextlib
import io
from pathlib import Path
import sys
import tempfile
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

import picking_box_5 as pick5
import placing_and_picking as pap


def transform(x: float, y: float, z: float) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = [x, y, z]
    return result


class PlacingAndPickingTests(unittest.TestCase):
    def test_default_cli_is_destination_place_regrasp_only(self) -> None:
        args = pap.parse_args(["--address", "localhost:50051"])

        self.assertEqual(args.lift_target_json, pap.DEFAULT_LIFT_TARGET_JSON)
        self.assertAlmostEqual(args.place_wait_sec, 1.0)
        self.assertAlmostEqual(args.place_lower_delta_m, 0.08)
        self.assertFalse(hasattr(args, "initial_pick"))
        self.assertFalse(hasattr(args, "gripper_open"))
        self.assertFalse(hasattr(args, "visualize"))
        self.assertFalse(hasattr(args, "mobile_base_align"))

    def test_target_chain_is_derived_from_lift_target_not_current_fk(self) -> None:
        lifted = pap.TargetPair(
            right=transform(0.45, -0.24, 1.12),
            left=transform(0.45, +0.24, 1.12),
        )

        targets = pap.build_place_regrasp_target_chain(lifted)

        np.testing.assert_allclose(targets.lowered.right[:3, 3], [0.45, -0.24, 1.04])
        np.testing.assert_allclose(targets.lowered.left[:3, 3], [0.45, +0.24, 1.04])
        np.testing.assert_allclose(targets.released.right[:3, 3], [0.45, -0.34, 1.04])
        np.testing.assert_allclose(targets.released.left[:3, 3], [0.45, +0.34, 1.04])
        np.testing.assert_allclose(targets.regrasped.right, targets.lowered.right)
        np.testing.assert_allclose(targets.regrasped.left, targets.lowered.left)
        np.testing.assert_allclose(targets.lifted.right, lifted.right)
        np.testing.assert_allclose(targets.lifted.left, lifted.left)

    def test_target_chain_handles_swapped_hand_y_order(self) -> None:
        lifted = pap.TargetPair(
            right=transform(0.45, +0.24, 1.12),
            left=transform(0.45, -0.24, 1.12),
        )

        targets = pap.build_place_regrasp_target_chain(lifted)

        np.testing.assert_allclose(targets.released.right[:3, 3], [0.45, +0.34, 1.04])
        np.testing.assert_allclose(targets.released.left[:3, 3], [0.45, -0.34, 1.04])

    def test_picking_box_5_writes_placing_target_record(self) -> None:
        right = transform(0.4, -0.2, 1.1)
        left = transform(0.4, +0.2, 1.1)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "latest_pick_lift_target.json"

            pick5.write_lift_target_record(path, right, left, source="unit-test")
            loaded = pap.load_lift_target_record(path)

        np.testing.assert_allclose(loaded.right, right)
        np.testing.assert_allclose(loaded.left, left)

    def test_perform_sequence_orders_target_ramps_and_wait(self) -> None:
        calls: list[tuple[str, object]] = []
        lifted = pap.TargetPair(
            right=transform(0.45, -0.24, 1.12),
            left=transform(0.45, +0.24, 1.12),
        )

        class FakeRobot:
            def cancel_control(self) -> None:
                calls.append(("cancel",))

        original_stream = pap.stream_target_ramp_stage
        original_wait = pap.wait_for_eef_targets
        original_gap_wait = pap.wait_for_gap_motion
        original_current_pair = pap.current_eef_pair
        original_sleep = pap.time.sleep

        def xyz(target: pap.TargetPair) -> tuple[float, float, float]:
            return tuple(round(float(v), 6) for v in target.right[:3, 3])

        def fake_stream(_robot, *, start, end, stage: str, ramp_time_sec: float, **kwargs):
            calls.append(
                (
                    "stream",
                    stage,
                    float(ramp_time_sec),
                    round(float(kwargs["final_hold_sec"]), 6),
                    xyz(end),
                )
            )
            return True

        def fake_wait(_robot, _dyn_model, _dyn_state, target, *, stage: str, **_kwargs):
            calls.append(("wait", stage, xyz(target)))
            return True

        def fake_gap_wait(
            _robot,
            _dyn_model,
            _dyn_state,
            *,
            initial_gap_m: float,
            target_gap_m: float,
            stage: str,
            **_kwargs,
        ) -> bool:
            calls.append(
                (
                    "gap_wait",
                    stage,
                    round(float(initial_gap_m), 6),
                    round(float(target_gap_m), 6),
                )
            )
            return True

        def fake_current_pair(_robot, _dyn_model, _dyn_state) -> pap.TargetPair:
            return lifted

        def fake_sleep(seconds: float) -> None:
            calls.append(("sleep", float(seconds)))

        pap.stream_target_ramp_stage = fake_stream
        pap.wait_for_eef_targets = fake_wait
        pap.wait_for_gap_motion = fake_gap_wait
        pap.current_eef_pair = fake_current_pair
        pap.time.sleep = fake_sleep
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ok = pap.perform_place_regrasp_sequence(
                    FakeRobot(),
                    object(),
                    object(),
                    lifted,
                    place_wait_sec=1.0,
                    lower_ramp_time_sec=1.1,
                    release_ramp_time_sec=0.5,
                    regrasp_ramp_time_sec=0.6,
                    lift_ramp_time_sec=1.2,
                )
        finally:
            pap.stream_target_ramp_stage = original_stream
            pap.wait_for_eef_targets = original_wait
            pap.wait_for_gap_motion = original_gap_wait
            pap.current_eef_pair = original_current_pair
            pap.time.sleep = original_sleep

        self.assertTrue(ok)
        self.assertEqual(
            calls,
            [
                ("cancel",),
                ("sleep", 0.3),
                ("stream", "1/5 place_lower", 1.1, 3.0, (0.45, -0.24, 1.04)),
                ("wait", "1/5 place_lower", (0.45, -0.24, 1.04)),
                ("cancel",),
                ("sleep", 0.3),
                ("stream", "2/5 release_open", 0.5, 3.0, (0.45, -0.34, 1.04)),
                ("gap_wait", "2/5 release_open", 0.48, 0.68),
                ("sleep", 1.0),
                ("cancel",),
                ("sleep", 0.3),
                ("stream", "4/5 regrasp_push", 0.6, 3.0, (0.45, -0.24, 1.04)),
                ("gap_wait", "4/5 regrasp_push", 0.48, 0.48),
                ("cancel",),
                ("sleep", 0.3),
                ("stream", "5/5 regrasp_lift", 1.2, 100.0, (0.45, -0.24, 1.12)),
                ("wait", "5/5 regrasp_lift", (0.45, -0.24, 1.12)),
            ],
        )

    def test_release_open_accepts_partial_target_when_gap_increases_enough(self) -> None:
        class FakeRobot:
            def __init__(self) -> None:
                self.positions = iter(
                    [
                        (-0.24, +0.24),
                        (-0.29, +0.29),
                    ]
                )

            def get_state(self) -> object:
                return types.SimpleNamespace(position=np.zeros(1))

        class FakeDynState:
            def set_q(self, _q) -> None:
                pass

        class FakeDynModel:
            def __init__(self, robot: FakeRobot) -> None:
                self.robot = robot
                self.current = (-0.24, +0.24)

            def compute_forward_kinematics(self, _state) -> None:
                try:
                    self.current = next(self.robot.positions)
                except StopIteration:
                    pass

            def compute_transformation(self, _state, _base_index, ee_index) -> np.ndarray:
                y = self.current[0] if ee_index == pap.EE_RIGHT_INDEX else self.current[1]
                return transform(0.45, y, 1.04)

        robot = FakeRobot()
        dyn_model = FakeDynModel(robot)

        original_sleep = pap.time.sleep
        pap.time.sleep = lambda _seconds: None
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ok = pap.wait_for_gap_motion(
                    robot,
                    dyn_model,
                    FakeDynState(),
                    initial_gap_m=0.48,
                    target_gap_m=0.68,
                    stage="2/5 release_open",
                    timeout_sec=0.1,
                )
        finally:
            pap.time.sleep = original_sleep

        self.assertTrue(ok)

    def test_target_stream_retries_first_expired_send_from_idle(self) -> None:
        calls: list[tuple[str, object]] = []
        lifted = pap.TargetPair(
            right=transform(0.45, -0.24, 1.12),
            left=transform(0.45, +0.24, 1.12),
        )

        class FakeStream:
            def __init__(self, stream_id: int) -> None:
                self.stream_id = stream_id

            def send_command(self, _command: object) -> None:
                calls.append(("send", self.stream_id))
                if self.stream_id == 1:
                    raise RuntimeError("This command stream is expired")

        class FakeRobot:
            def __init__(self) -> None:
                self.stream_id = 0

            def create_command_stream(self, *, priority: int) -> FakeStream:
                self.stream_id += 1
                calls.append(("create", priority, self.stream_id))
                return FakeStream(self.stream_id)

            def cancel_control(self) -> None:
                calls.append(("cancel",))

        original_build = pap.build_dual_arm_impedance_target_command
        original_sleep = pap.time.sleep
        pap.build_dual_arm_impedance_target_command = lambda *_args, **_kwargs: object()
        pap.time.sleep = lambda seconds: calls.append(("sleep", float(seconds)))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ok = pap.stream_target_ramp_stage(
                    FakeRobot(),
                    start=lifted,
                    end=lifted,
                    stage="unit",
                    ramp_time_sec=0.0,
                )
        finally:
            pap.build_dual_arm_impedance_target_command = original_build
            pap.time.sleep = original_sleep

        self.assertTrue(ok)
        self.assertEqual(
            calls,
            [
                ("create", pap.COMMAND_STREAM_PRIORITY, 1),
                ("send", 1),
                ("cancel",),
                ("sleep", pap.STREAM_RETRY_IDLE_SLEEP_SEC),
                ("create", pap.COMMAND_STREAM_PRIORITY, 2),
                ("send", 2),
                ("send", 2),
            ],
        )

    def test_default_main_does_not_call_gripper_vision_or_initial_pick_helpers(self) -> None:
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

        right = transform(0.4, -0.2, 1.1)
        left = transform(0.4, +0.2, 1.1)
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "target.json"
            pick5.write_lift_target_record(target_path, right, left)

            original_rby = pap.rby
            original_perform = pap.perform_place_regrasp_sequence
            pap.rby = types.SimpleNamespace(create_robot=lambda _address, _model: FakeRobot())

            def fake_perform(*_args, **_kwargs) -> bool:
                calls.append("perform_place_regrasp")
                return True

            pap.perform_place_regrasp_sequence = fake_perform
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    ok = pap.main(
                        address="localhost:50051",
                        model="m",
                        power=".*",
                        lift_target_json=target_path,
                        place_lower_delta_m=0.08,
                        place_wait_sec=1.0,
                        lower_ramp_time_sec=1.0,
                        release_ramp_time_sec=0.5,
                        regrasp_ramp_time_sec=0.5,
                        lift_ramp_time_sec=1.0,
                        eef_wait_timeout_sec=4.0,
                    )
            finally:
                pap.rby = original_rby
                pap.perform_place_regrasp_sequence = original_perform

        self.assertTrue(ok)
        self.assertEqual(calls, ["connect", "reset_fault", "perform_place_regrasp"])


if __name__ == "__main__":
    unittest.main()
