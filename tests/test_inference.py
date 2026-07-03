from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference import build_pose_record, config_from_args, parse_args, plane_prior_status


class InferenceTests(unittest.TestCase):
    def test_help_does_not_require_realsense_runtime(self) -> None:
        root = Path(__file__).resolve().parents[1]

        result = subprocess.run(
            [sys.executable, str(root / "inference.py"), "--help"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("--init-frames", result.stdout)
        self.assertIn("--view-rotation", result.stdout)

    def test_config_defaults_match_live_d405_inference(self) -> None:
        config = config_from_args(parse_args([]))

        self.assertEqual(config.width, 1280)
        self.assertEqual(config.height, 720)
        self.assertEqual(config.fps, 30)
        self.assertEqual(config.view_rotation, "cw90")
        self.assertEqual(config.init_frames, 15)
        self.assertTrue(config.preview)
        self.assertTrue(config.align_depth_to_color)
        self.assertTrue(config.image_fallback)

    def test_config_rejects_invalid_init_frames(self) -> None:
        with self.assertRaisesRegex(ValueError, "--init-frames"):
            config_from_args(parse_args(["--init-frames", "-1"]))

    def test_plane_prior_status_reports_fallback_mode(self) -> None:
        self.assertEqual(
            plane_prior_status(None),
            {
                "available": False,
                "mode": "per_frame_discovery",
                "frames_used": 0,
                "frames_discovered": 0,
                "offset_spread_m": None,
            },
        )

    def test_plane_prior_status_reports_startup_burst(self) -> None:
        status = plane_prior_status({"frames_used": 4, "frames_discovered": 6, "offset_spread_m": 0.002})

        self.assertEqual(status["available"], True)
        self.assertEqual(status["mode"], "startup_burst")
        self.assertEqual(status["frames_used"], 4)
        self.assertEqual(status["frames_discovered"], 6)
        self.assertEqual(status["offset_spread_m"], 0.002)

    def test_build_pose_record_promotes_required_ok_fields(self) -> None:
        record = build_pose_record(
            frame_id=7,
            timestamp_ms=123.5,
            view_rotation="cw90",
            known_size={
                "method": "plane",
                "center_top_camera_m": [0.01, 0.02, 0.4],
                "yaw_mod_180": 12.5,
                "center_image": [640.0, 360.0],
                "confidence": {"ok": True, "score": 0.9, "reasons": []},
                "failure_reasons": [],
                "support": {
                    "long_axis_camera": [1.0, 0.0, 0.0],
                    "short_axis_camera": [0.0, -0.7, 0.7],
                },
            },
            plane_prior={"frames_used": 3, "frames_discovered": 3, "offset_spread_m": 0.001},
        )

        self.assertTrue(record["ok"])
        self.assertEqual(record["frame_id"], 7)
        self.assertEqual(record["center_top_camera_m"], [0.01, 0.02, 0.4])
        self.assertEqual(record["yaw_mod_180"], 12.5)
        self.assertEqual(record["long_axis_camera"], [1.0, 0.0, 0.0])
        self.assertEqual(record["short_axis_camera"], [0.0, -0.7, 0.7])
        self.assertEqual(record["plane_prior"]["mode"], "startup_burst")

    def test_build_pose_record_nulls_pose_when_rejected(self) -> None:
        record = build_pose_record(
            frame_id=8,
            timestamp_ms=None,
            view_rotation="cw90",
            known_size={
                "method": "plane",
                "center_top_camera_m": [0.01, 0.02, 0.4],
                "yaw_mod_180": 12.5,
                "center_image": [640.0, 360.0],
                "confidence": {"ok": False, "score": 0.2, "reasons": ["long_axis_center_underconstrained"]},
                "failure_reasons": ["long_axis_center_underconstrained"],
                "support": {
                    "long_axis_camera": [1.0, 0.0, 0.0],
                    "short_axis_camera": [0.0, -0.7, 0.7],
                },
            },
            plane_prior=None,
        )

        self.assertFalse(record["ok"])
        self.assertIsNone(record["center_top_camera_m"])
        self.assertIsNone(record["yaw_mod_180"])
        self.assertIsNone(record["long_axis_camera"])
        self.assertIsNone(record["short_axis_camera"])
        self.assertEqual(record["failure_reasons"], ["long_axis_center_underconstrained"])
        self.assertEqual(record["plane_prior"]["mode"], "per_frame_discovery")


if __name__ == "__main__":
    unittest.main()
