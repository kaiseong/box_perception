from __future__ import annotations

import contextlib
import io
import types
import unittest

import place_and_picking as pap
import place_and_picking_sim as sim


class PlaceAndPickingSimTests(unittest.TestCase):
    def test_default_args_target_local_m_sim(self) -> None:
        args = sim.parse_args([])

        self.assertEqual(args.address, "localhost:50051")
        self.assertEqual(args.model, "m")
        self.assertAlmostEqual(args.place_wait_sec, pap.PLACE_WAIT_AFTER_RELEASE_SEC)
        self.assertAlmostEqual(args.push_ramp_time_sec, pap.PUSH_RAMP_TIME)

    def test_refuses_non_local_address_without_allow_real(self) -> None:
        args = sim.parse_args(["--address", "192.168.0.10:50051"])

        with self.assertRaises(SystemExit):
            sim.validate_args(args)

    def test_dry_run_needs_no_robot_sdk(self) -> None:
        args = sim.parse_args(["--dry-run"])
        original_rby = sim.rby

        sim.rby = None
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ok = sim.run(args)
        finally:
            sim.rby = original_rby

        self.assertTrue(ok)

    def test_run_invokes_destination_sequence_only(self) -> None:
        calls: list[tuple[str, object]] = []

        class FakeRobot:
            def model(self) -> object:
                return types.SimpleNamespace(robot_joint_names=["j0"])

            def get_dynamics(self) -> object:
                return types.SimpleNamespace(make_state=lambda links, joints: (tuple(links), tuple(joints)))

        args = sim.parse_args(["--skip-enable", "--place-wait-sec", "1.25"])
        original_connect = sim.connect_and_enable_robot
        original_perform = pap.perform_place_regrasp_sequence

        def fake_connect_and_enable_robot(**kwargs):
            calls.append(("connect", kwargs))
            return FakeRobot()

        def fake_perform(robot, dyn_model, dyn_state, **kwargs) -> bool:
            calls.append(("perform", kwargs, dyn_state))
            return True

        sim.connect_and_enable_robot = fake_connect_and_enable_robot
        pap.perform_place_regrasp_sequence = fake_perform
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ok = sim.run(args)
        finally:
            sim.connect_and_enable_robot = original_connect
            pap.perform_place_regrasp_sequence = original_perform

        self.assertTrue(ok)
        self.assertEqual(calls[0][0], "connect")
        self.assertEqual(calls[1][0], "perform")
        self.assertAlmostEqual(calls[1][1]["place_wait_sec"], 1.25)
        self.assertEqual(calls[1][2], (tuple(pap.DYN_LINK_NAMES), ("j0",)))


if __name__ == "__main__":
    unittest.main()
