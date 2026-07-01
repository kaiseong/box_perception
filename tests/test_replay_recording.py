from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recording import RecordingConfig, build_manifest, prepare_session, save_frame
from replay_recording import (
    angular_distance_mod_180,
    iter_index_records,
    load_depth_frame,
    load_manifest,
    load_rgb_frame,
    summarize_results,
    yaw_summary,
)


class ReplayRecordingTests(unittest.TestCase):
    def test_loads_recording_manifest_index_rgb_and_depth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = prepare_session(tmp, "session_replay")
            manifest = build_manifest(
                sample_config(),
                camera_info={"serial_number": 15466, "camera_model": "ZED"},
                intrinsics={"fx": 667.0, "fy": 667.0, "cx": 669.0, "cy": 371.0},
                started_at="2026-07-01T00:00:00.000Z",
            )
            paths.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            bgr = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
            depth = np.array([[1.0, np.inf, -np.inf], [0.0, 1.2, 1.3]], dtype=np.float32)
            save_frame(
                paths,
                frame_id=0,
                bgr=bgr,
                depth_m=depth,
                rgb_format="npy",
                depth_format="npy",
                jpeg_quality=95,
                wall_time="2026-07-01T00:00:00.000Z",
                monotonic_time_sec=1.0,
            )

            session = Path(tmp) / "session_replay"
            loaded_manifest = load_manifest(session)
            records = iter_index_records(session)

            self.assertEqual(loaded_manifest["camera"]["serial_number"], 15466)
            self.assertEqual(len(records), 1)
            np.testing.assert_array_equal(load_rgb_frame(session, records[0], cv2_module=None), bgr)
            np.testing.assert_array_equal(load_depth_frame(session, records[0]), depth)

    def test_iter_index_records_honors_stride_and_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            with (session / "index.jsonl").open("w", encoding="utf-8") as f:
                for frame_id in range(6):
                    f.write(json.dumps({"frame_id": frame_id}) + "\n")

            records = iter_index_records(session, stride=2, max_frames=2)

            self.assertEqual([record["frame_id"] for record in records], [0, 2])

    def test_yaw_summary_wraps_modulo_180(self) -> None:
        self.assertAlmostEqual(angular_distance_mod_180(179.0, 1.0), 2.0)

        summary = yaw_summary([179.0, 1.0])

        self.assertIsNotNone(summary)
        self.assertAlmostEqual(summary["spread_deg"], 2.0)

    def test_summarize_results_counts_pixel_and_metric_success(self) -> None:
        results = [
            sample_result(frame_id=0, pixel_ok=True, metric_ok=True, yaw=1.0),
            sample_result(frame_id=1, pixel_ok=True, metric_ok=True, yaw=2.0),
            sample_result(frame_id=2, pixel_ok=False, metric_ok=False, yaw=float("nan")),
        ]

        summary = summarize_results(results)

        self.assertEqual(summary["frames_analyzed"], 3)
        self.assertEqual(summary["pixel_ok_frames"], 2)
        self.assertEqual(summary["metric_ok_frames"], 2)
        self.assertAlmostEqual(summary["pixel_ok_fraction"], 2 / 3)


def sample_config() -> RecordingConfig:
    return RecordingConfig(
        output_root="recordings",
        session_name="unit",
        fps=15,
        resolution="HD720",
        depth_mode="ULTRA",
        max_frames=None,
        duration_sec=20.0,
        warmup_frames=10,
        rgb_format="npy",
        depth_format="npy",
        jpeg_quality=95,
        preview=False,
        serial_number=None,
    )


def sample_result(*, frame_id: int, pixel_ok: bool, metric_ok: bool, yaw: float) -> dict:
    return {
        "frame_id": frame_id,
        "pixel": {
            "confidence": {"ok": pixel_ok, "score": 1.0 if pixel_ok else 0.0, "reasons": []},
            "yaw_mod_180": yaw,
        },
        "metric": {
            "confidence": {"ok": metric_ok, "score": 1.0 if metric_ok else 0.0, "reasons": []},
            "yaw_mod_180": yaw,
            "center_camera_m": [float(frame_id), 0.0, 1.0] if metric_ok else None,
        },
    }


if __name__ == "__main__":
    unittest.main()
