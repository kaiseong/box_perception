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

    def test_place_only_sequence_reuses_current_fk_place_release_and_returns_ready(self) -> None:
        calls: list[tuple[str, str]] = []

        def perform_place_release_sequence(*_args: object, **kwargs: object) -> bool:
            calls.append(("place_release", f"{float(kwargs['push_ramp_time_sec']):.1f}"))
            return True

        fake_module = SimpleNamespace(
            perform_place_release_sequence=perform_place_release_sequence,
            COMMAND_TIMEOUT_MARGIN_SEC=5.0,
            COMMAND_TIMEOUT_MIN_SEC=8.0,
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
                release_ramp_time_sec=0.5,
                ready_time_sec=2.0,
            )

        self.assertTrue(ok)
        self.assertEqual(
            calls,
            [
                ("place_release", "0.5"),
                ("ready", "2.0"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
