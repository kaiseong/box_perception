from __future__ import annotations

import contextlib
import io
import types
import unittest

import numpy as np


class _FinishCode:
    Ok = "Ok"


import sys

sys.modules.setdefault(
    "rby1_sdk",
    types.SimpleNamespace(
        RobotCommandFeedback=types.SimpleNamespace(FinishCode=_FinishCode),
    ),
)

import placing_and_picking as pap


class PlacingAndPickingTests(unittest.TestCase):
    def test_default_cli_is_current_fk_place_release_only(self) -> None:
        args = pap.parse_args(["--address", "localhost:50051"])

        self.assertAlmostEqual(args.push_ramp_time_sec, pap.PUSH_RAMP_TIME)
        self.assertFalse(hasattr(args, "lift_target_json"))
        self.assertFalse(hasattr(args, "place_wait_sec"))
        self.assertFalse(hasattr(args, "gripper_open"))
        self.assertFalse(hasattr(args, "visualize"))
        self.assertFalse(hasattr(args, "mobile_base_align"))

    def test_perform_sequence_lowers_then_releases_outward(self) -> None:
        calls: list[tuple[str, object]] = []

        class FakeRobot:
            def get_state(self) -> object:
                return types.SimpleNamespace(position=np.zeros(1))

        def fake_lower(_dyn_model, _dyn_state, q):
            calls.append(("build_lower", tuple(np.asarray(q).tolist())))
            return "lower_command"

        def fake_send(_robot, command, stage: str, **kwargs):
            calls.append(("send", command, stage, round(float(kwargs["timeout_sec"]), 3)))
            return True

        def fake_stream(_robot, _dyn_model, _dyn_state, q, **kwargs):
            calls.append(
                (
                    "stream",
                    kwargs["stage"],
                    round(float(kwargs["target_inward"]), 6),
                    round(float(kwargs["ramp_time_sec"]), 6),
                    round(float(kwargs["hold_time_sec"]), 6),
                    tuple(np.asarray(q).tolist()),
                )
            )
            return True

        original_lower = pap.build_impedance_lower_command
        original_send = pap.send_stage
        original_stream = pap.stream_impedance_push_stage
        pap.build_impedance_lower_command = fake_lower
        pap.send_stage = fake_send
        pap.stream_impedance_push_stage = fake_stream
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ok = pap.perform_place_release_sequence(
                    FakeRobot(),
                    object(),
                    object(),
                    push_ramp_time_sec=0.7,
                )
        finally:
            pap.build_impedance_lower_command = original_lower
            pap.send_stage = original_send
            pap.stream_impedance_push_stage = original_stream

        self.assertTrue(ok)
        self.assertEqual(calls[0], ("build_lower", (0.0,)))
        self.assertEqual(calls[1][0:3], ("send", "lower_command", "1/2 place_lower"))
        self.assertEqual(
            calls[2],
            (
                "stream",
                "2/2 release_open",
                -pap.PUSH_DISTANCE,
                0.7,
                pap.PUSH_HOLD_TIME,
                (0.0,),
            ),
        )

    def test_default_main_does_not_call_gripper_vision_or_regrasp_helpers(self) -> None:
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

        original_rby = pap.rby
        original_perform = pap.perform_place_release_sequence
        pap.rby = types.SimpleNamespace(create_robot=lambda _address, _model: FakeRobot())

        def fake_perform(*_args, **_kwargs) -> bool:
            calls.append("perform_place_release")
            return True

        pap.perform_place_release_sequence = fake_perform
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ok = pap.main(
                    address="localhost:50051",
                    model="m",
                    power=".*",
                    push_ramp_time_sec=0.5,
                    command_timeout_margin_sec=5.0,
                    min_command_timeout_sec=8.0,
                )
        finally:
            pap.rby = original_rby
            pap.perform_place_release_sequence = original_perform

        self.assertTrue(ok)
        self.assertEqual(calls, ["connect", "reset_fault", "perform_place_release"])


if __name__ == "__main__":
    unittest.main()
