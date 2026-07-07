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

import picking_and_place as pap


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

    def test_default_cli_runs_full_cycle_with_one_second_place_wait(self) -> None:
        args = pap.parse_args(["--address", "localhost:50051"])

        self.assertTrue(args.continue_pick)
        self.assertTrue(args.place_regrasp)
        self.assertTrue(args.transfer_confirm)
        self.assertAlmostEqual(args.place_wait_sec, 1.0)

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


if __name__ == "__main__":
    unittest.main()
