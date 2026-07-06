from __future__ import annotations

import sys
import types
import unittest


class _FinishCode:
    Ok = "Ok"


sys.modules.setdefault(
    "rby1_sdk",
    types.SimpleNamespace(
        RobotCommandFeedback=types.SimpleNamespace(FinishCode=_FinishCode),
    ),
)

from picking_box_2 import send_stage, stage_timeout_sec, wait_for_command_feedback


class _Feedback:
    finish_code = _FinishCode.Ok


class _FakeCommand:
    def __init__(self, *, completes: bool) -> None:
        self.completes = completes
        self.cancel_called = False
        self.wait_calls: list[int] = []

    def wait_for(self, timeout_ms: int) -> bool:
        self.wait_calls.append(timeout_ms)
        return self.completes

    def get(self) -> _Feedback:
        return _Feedback()

    def cancel(self) -> None:
        self.cancel_called = True


class _FakeRobot:
    def __init__(self, command: _FakeCommand) -> None:
        self.command = command
        self.cancel_control_called = False

    def send_command(self, builder: object) -> _FakeCommand:
        return self.command

    def cancel_control(self) -> None:
        self.cancel_control_called = True


class PickingBox2TimeoutTests(unittest.TestCase):
    def test_stage_timeout_uses_expected_duration_with_margin_and_minimum(self) -> None:
        self.assertEqual(stage_timeout_sec(1.0, 1.0, min_timeout_sec=8.0, margin_sec=5.0), 8.0)
        self.assertEqual(stage_timeout_sec(5.0, 100.0, min_timeout_sec=8.0, margin_sec=5.0), 110.0)

    def test_wait_for_command_feedback_returns_feedback_when_completed(self) -> None:
        command = _FakeCommand(completes=True)

        feedback = wait_for_command_feedback(command, "stage", timeout_sec=1.0)

        self.assertIsInstance(feedback, _Feedback)
        self.assertEqual(command.wait_calls, [100])

    def test_send_stage_cancels_timed_out_command(self) -> None:
        command = _FakeCommand(completes=False)
        robot = _FakeRobot(command)

        ok = send_stage(robot, object(), "stage", timeout_sec=0.001)

        self.assertFalse(ok)
        self.assertTrue(command.cancel_called)
        self.assertTrue(robot.cancel_control_called)


if __name__ == "__main__":
    unittest.main()
