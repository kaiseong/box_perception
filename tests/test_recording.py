from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recording import (
    FORMAT_VERSION,
    RecordingConfig,
    build_manifest,
    config_from_args,
    default_session_name,
    depth_statistics,
    prepare_session,
    save_frame,
    should_continue,
    zed_image_to_bgr,
)


class RecordingTests(unittest.TestCase):
    def test_default_session_name_uses_utc_zed_prefix(self) -> None:
        name = default_session_name()

        self.assertTrue(name.startswith("zed_"))
        self.assertTrue(name.endswith("Z"))

    def test_config_requires_stop_condition(self) -> None:
        args = type(
            "Args",
            (),
            {
                "output_root": "recordings",
                "session_name": None,
                "fps": 15,
                "resolution": "HD720",
                "depth_mode": "ULTRA",
                "max_frames": None,
                "duration_sec": None,
                "warmup_frames": 10,
                "rgb_format": "jpg",
                "depth_format": "npz",
                "jpeg_quality": 95,
                "preview": False,
                "serial_number": None,
            },
        )()

        with self.assertRaises(ValueError):
            config_from_args(args)

    def test_config_rejects_unsupported_camera_fps(self) -> None:
        args = type(
            "Args",
            (),
            {
                "output_root": "recordings",
                "session_name": None,
                "fps": 10,
                "resolution": "HD720",
                "depth_mode": "ULTRA",
                "max_frames": 1,
                "duration_sec": None,
                "warmup_frames": 10,
                "rgb_format": "npy",
                "depth_format": "npz",
                "jpeg_quality": 95,
                "preview": False,
                "serial_number": None,
            },
        )()

        with self.assertRaisesRegex(ValueError, "--fps must be one of"):
            config_from_args(args)

    def test_prepare_session_creates_expected_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = prepare_session(tmp, "session_a")

            self.assertTrue(paths.session_dir.exists())
            self.assertTrue(paths.rgb_dir.exists())
            self.assertTrue(paths.depth_dir.exists())
            self.assertEqual(paths.index_path, Path(tmp) / "session_a" / "index.jsonl")
            self.assertEqual(paths.manifest_path, Path(tmp) / "session_a" / "manifest.json")

            with self.assertRaises(FileExistsError):
                prepare_session(tmp, "session_a")

    def test_build_manifest_captures_config_intrinsics_and_layout(self) -> None:
        config = sample_config()
        manifest = build_manifest(
            config,
            camera_info={"serial_number": 1234, "camera_model": "ZED-X"},
            intrinsics={"fx": 100.0, "fy": 101.0, "cx": 50.0, "cy": 51.0},
            started_at="2026-07-01T00:00:00.000Z",
        )

        self.assertEqual(manifest["format_version"], FORMAT_VERSION)
        self.assertEqual(manifest["config"]["session_name"], "unit")
        self.assertEqual(manifest["camera"]["serial_number"], 1234)
        self.assertEqual(manifest["intrinsics"]["fx"], 100.0)
        self.assertEqual(manifest["data_layout"]["depth_units"], "meter")
        self.assertEqual(manifest["recording"]["frame_count"], 0)

    def test_save_frame_writes_rgb_depth_and_index_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = prepare_session(tmp, "session_b")
            bgr = np.zeros((4, 5, 3), dtype=np.uint8)
            bgr[:, :, 1] = 200
            depth = np.array(
                [
                    [0.0, 1.0, 1.1, np.nan, 2.0],
                    [1.2, 1.3, 0.0, 1.4, 1.5],
                    [1.6, 1.7, 1.8, 1.9, 2.1],
                    [0.0, 0.0, 2.2, 2.3, 2.4],
                ],
                dtype=np.float32,
            )

            record = save_frame(
                paths,
                frame_id=0,
                bgr=bgr,
                depth_m=depth,
                rgb_format="png",
                depth_format="npz",
                jpeg_quality=95,
                wall_time="2026-07-01T00:00:00.000Z",
                monotonic_time_sec=12.5,
            )

            rgb_path = paths.session_dir / record["rgb_path"]
            depth_path = paths.session_dir / record["depth_path"]
            self.assertTrue(rgb_path.exists())
            self.assertTrue(depth_path.exists())
            self.assertEqual(cv2.imread(str(rgb_path), cv2.IMREAD_COLOR).shape, (4, 5, 3))
            np.testing.assert_allclose(np.load(depth_path)["depth_m"], depth, equal_nan=True)

            lines = paths.index_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            index_record = json.loads(lines[0])
            self.assertEqual(index_record["frame_id"], 0)
            self.assertEqual(index_record["rgb_path"], "rgb/frame_000000.png")
            self.assertEqual(index_record["depth_path"], "depth/frame_000000.depth.npz")
            self.assertEqual(index_record["depth_stats"]["valid_count"], 15)

    def test_save_frame_supports_rgb_npy_for_pyzed_numpy2_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = prepare_session(tmp, "session_rgb_npy")
            bgr = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
            depth = np.ones((2, 4), dtype=np.float32)

            record = save_frame(
                paths,
                frame_id=7,
                bgr=bgr,
                depth_m=depth,
                rgb_format="npy",
                depth_format="npz",
                jpeg_quality=95,
                wall_time="2026-07-01T00:00:00.000Z",
                monotonic_time_sec=99.0,
            )

            rgb_path = paths.session_dir / record["rgb_path"]
            self.assertEqual(record["rgb_path"], "rgb/frame_000007.npy")
            np.testing.assert_array_equal(np.load(rgb_path), bgr)

    def test_zed_bgra_image_conversion_drops_alpha_without_opencv(self) -> None:
        bgra = np.zeros((2, 3, 4), dtype=np.uint8)
        bgra[:, :, 0] = 10
        bgra[:, :, 1] = 20
        bgra[:, :, 2] = 30
        bgra[:, :, 3] = 255

        bgr = zed_image_to_bgr(bgra)

        self.assertEqual(bgr.shape, (2, 3, 3))
        self.assertTrue(np.all(bgr[:, :, 0] == 10))
        self.assertTrue(np.all(bgr[:, :, 1] == 20))
        self.assertTrue(np.all(bgr[:, :, 2] == 30))

    def test_depth_statistics_handles_empty_depth(self) -> None:
        stats = depth_statistics(np.zeros((3, 4), dtype=np.float32))

        self.assertEqual(stats["valid_count"], 0)
        self.assertIsNone(stats["min_m"])
        self.assertIsNone(stats["max_m"])
        self.assertIsNone(stats["mean_m"])

    def test_should_continue_honors_frame_and_duration_limits(self) -> None:
        config = sample_config(max_frames=3, duration_sec=10.0)

        self.assertTrue(should_continue(2, 9.0, config))
        self.assertFalse(should_continue(3, 9.0, config))
        self.assertFalse(should_continue(2, 10.0, config))


def sample_config(max_frames: int | None = 10, duration_sec: float | None = None) -> RecordingConfig:
    return RecordingConfig(
        output_root="recordings",
        session_name="unit",
        fps=15,
        resolution="HD720",
        depth_mode="ULTRA",
        max_frames=max_frames,
        duration_sec=duration_sec,
        warmup_frames=0,
        rgb_format="jpg",
        depth_format="npz",
        jpeg_quality=95,
        preview=False,
        serial_number=None,
    )


if __name__ == "__main__":
    unittest.main()
