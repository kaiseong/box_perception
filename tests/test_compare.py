from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

import compare


class CompareHarnessTests(unittest.TestCase):
    def test_candidate_mapping_uses_legacy_and_new_files(self) -> None:
        self.assertEqual(compare.CANDIDATES["1"].label, "legacy")
        self.assertEqual(compare.CANDIDATES["1"].module.__name__, "picking_box_legacy")
        self.assertEqual(compare.CANDIDATES["2"].label, "new")
        self.assertEqual(compare.CANDIDATES["2"].module.__name__, "picking_box_new")
        self.assertEqual(compare.PLACE_ONLY_KEY, "3")

    def test_build_candidate_argv_forces_no_gripper_and_warmed_center(self) -> None:
        args = argparse.Namespace(
            address="192.168.30.1:50051",
            model="m",
            power=".*",
            view_rotation="cw90",
        )
        argv = compare.build_candidate_argv(
            args,
            ["--servo-settled-frames", "1"],
            np.array([0.1, -0.2, 0.3], dtype=np.float64),
        )

        self.assertIn("--servo-settled-frames", argv)
        self.assertIn("--no-gripper-open", argv)
        self.assertEqual(argv[argv.index("--address") + 1], "192.168.30.1:50051")
        self.assertEqual(argv[argv.index("--view-rotation") + 1], "cw90")
        center_index = argv.index("--box-center-camera") + 1
        self.assertEqual(argv[center_index : center_index + 3], ["0.100000000", "-0.200000000", "0.300000000"])

    def test_append_result_writes_jsonl_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.jsonl"
            compare.append_result(path, {"candidate": "new", "elapsed_sec": 1.25})

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0]), {"candidate": "new", "elapsed_sec": 1.25})

    def test_place_only_sequence_lowers_releases_and_returns_ready_without_regrasp(self) -> None:
        target_pair = compare.placing_and_picking.TargetPair
        lifted_right = np.eye(4, dtype=np.float64)
        lifted_left = np.eye(4, dtype=np.float64)
        lifted_right[:3, 3] = [0.45, -0.05, 1.10]
        lifted_left[:3, 3] = [0.45, 0.05, 1.10]
        lifted = target_pair(right=lifted_right, left=lifted_left)

        calls: list[tuple[str, str]] = []
        observed: dict[str, float] = {}

        def stream_target_ramp_stage(*_args: object, **kwargs: object) -> bool:
            calls.append(("stream", str(kwargs["stage"])))
            return True

        def wait_for_eef_targets(*_args: object, **kwargs: object) -> bool:
            calls.append(("eef", str(kwargs["stage"])))
            target = kwargs["target"] if "target" in kwargs else _args[3]
            observed["lowered_z"] = float(target.right[2, 3])
            return True

        def wait_for_gap_motion(*_args: object, **kwargs: object) -> bool:
            calls.append(("gap", str(kwargs["stage"])))
            observed["initial_gap"] = float(kwargs["initial_gap_m"])
            observed["target_gap"] = float(kwargs["target_gap_m"])
            return True

        def cancel_control_for_next_stream(_robot: object, stage: str, **_kwargs: object) -> bool:
            calls.append(("cancel", stage))
            return True

        fake_module = SimpleNamespace(
            build_place_regrasp_target_chain=compare.placing_and_picking.build_place_regrasp_target_chain,
            print_stage=lambda *_args, **_kwargs: None,
            cancel_control_for_next_stream=cancel_control_for_next_stream,
            stream_target_ramp_stage=stream_target_ramp_stage,
            wait_for_eef_targets=wait_for_eef_targets,
            current_eef_pair=lambda *_args, **_kwargs: compare.placing_and_picking.offset_z(lifted, -0.08),
            hand_gap_m=compare.placing_and_picking.hand_gap_m,
            wait_for_gap_motion=wait_for_gap_motion,
        )

        with mock.patch.object(
            compare,
            "send_ready_after_place",
            side_effect=lambda _robot, *, ready_time_sec: calls.append(
                ("ready", f"{ready_time_sec:.1f}")
            )
            or True,
        ):
            ok = compare.perform_place_only_sequence(
                fake_module,
                robot=object(),
                dyn_model=object(),
                dyn_state=object(),
                lifted=lifted,
                place_lower_delta_m=0.08,
                place_release_distance_m=0.10,
                lower_ramp_time_sec=1.0,
                release_ramp_time_sec=0.5,
                eef_wait_timeout_sec=4.0,
                ready_time_sec=2.0,
            )

        self.assertTrue(ok)
        self.assertEqual(
            [call for call in calls if call[0] == "stream"],
            [
                ("stream", "place_only 1/2 place_lower"),
                ("stream", "place_only 2/2 release_open"),
            ],
        )
        self.assertNotIn(("stream", "4/5 regrasp_push"), calls)
        self.assertIn(("ready", "2.0"), calls)
        self.assertAlmostEqual(observed["lowered_z"], 1.02)
        self.assertAlmostEqual(observed["initial_gap"], 0.10)
        self.assertAlmostEqual(observed["target_gap"], 0.30)


if __name__ == "__main__":
    unittest.main()
