#!/usr/bin/env python3
"""Real-time box pose inference with the stored rim-plane calibration.

Live mode (D405 attached):

    python inference_2.py --config config_2/rim_plane.json

Replay mode (no camera, for testing the same code path offline):

    python inference_2.py --replay recordings/d405_center_visible

By default only confident estimates are drawn: frames whose confidence is not
ok show the failure reasons and keep the last confident pose as a gray ghost
so the operator can see what the robot would still be relying on. Use
--show-unreliable to also draw the current low-confidence fit in red.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

cv2.setNumThreads(1)  # the pixel loops gain nothing from the thread pool

from box_pose import CameraIntrinsics, estimate_plane_box, segment_yellow_box
from box_pose.visualization import draw_known_size_estimate
from replay_recording import (
    image_size_from_manifest_or_record,
    iter_index_records,
    load_depth_frame,
    load_manifest,
    load_rgb_frame,
    rotate_array_for_view,
    rotate_intrinsics_for_view,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live/replay box pose inference and visualization.")
    parser.add_argument("--config", default="config_2/rim_plane.json", help="Stored rim plane calibration JSON.")
    parser.add_argument("--replay", help="Recording session directory to replay instead of a live camera.")
    parser.add_argument("--stride", type=int, default=1, help="Replay frame stride.")
    parser.add_argument("--serial-number", help="RealSense serial number when several cameras are attached.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--view-rotation", default="cw90", choices=("none", "cw90", "ccw90", "180"))
    parser.add_argument("--box-long-m", type=float, default=0.505)
    parser.add_argument("--box-short-m", type=float, default=0.335)
    parser.add_argument("--box-height-m", type=float, default=0.195)
    parser.add_argument("--min-score", type=float, default=0.0, help="Extra score threshold on top of confidence ok.")
    parser.add_argument(
        "--show-unreliable",
        action="store_true",
        help="Also draw low-confidence estimates (in red) instead of only the reasons.",
    )
    parser.add_argument("--no-window", action="store_true", help="Do not open a display window.")
    parser.add_argument("--save-video", help="Write the visualization to this mp4 path.")
    parser.add_argument("--print-json", action="store_true", help="Print one JSON line per frame to stdout.")
    parser.add_argument("--max-frames", type=int, help="Stop after this many frames.")
    return parser.parse_args()


def load_rim_plane(path: str) -> tuple[list[float], list[float]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["normal"], data["point"]


def iter_replay_frames(args: argparse.Namespace):
    session_dir = Path(args.replay)
    manifest = load_manifest(session_dir)
    records = iter_index_records(session_dir, stride=args.stride, max_frames=args.max_frames)
    width, height = image_size_from_manifest_or_record(manifest, records[0])
    raw_intrinsics = CameraIntrinsics.from_mapping(manifest["intrinsics"])
    intrinsics = rotate_intrinsics_for_view(
        raw_intrinsics,
        width=width,
        height=height,
        rotation=args.view_rotation,
        intrinsics_cls=CameraIntrinsics,
    )
    for record in records:
        image = rotate_array_for_view(load_rgb_frame(session_dir, record, cv2), args.view_rotation)
        depth = rotate_array_for_view(load_depth_frame(session_dir, record), args.view_rotation)
        yield int(record["frame_id"]), image, depth, intrinsics


def iter_live_frames(
    args: argparse.Namespace | None = None,
    *,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    serial_number: str | None = None,
    view_rotation: str = "cw90",
):
    """Yield (frame_id, image, depth_m, intrinsics) from the live D405.

    The single shared D405 open/align/rotate path: inference_2.py's live mode
    and picking_box_2.py's capture both consume this generator, so camera
    settings only exist here. Closing the generator stops the pipeline.
    """
    if args is not None:
        width, height, fps = args.width, args.height, args.fps
        serial_number = args.serial_number
        view_rotation = args.view_rotation

    try:
        import pyrealsense2 as rs
    except ImportError as exc:  # pragma: no cover - camera runtime only
        raise SystemExit("pyrealsense2 is required for live mode. Use --replay for offline testing.") from exc

    pipeline = rs.pipeline()
    rs_config = rs.config()
    if serial_number:
        rs_config.enable_device(serial_number)
    rs_config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    rs_config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(rs_config)
    align = rs.align(rs.stream.color)
    depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    raw = color_profile.get_intrinsics()
    raw_intrinsics = CameraIntrinsics(fx=float(raw.fx), fy=float(raw.fy), cx=float(raw.ppx), cy=float(raw.ppy))
    intrinsics = rotate_intrinsics_for_view(
        raw_intrinsics,
        width=width,
        height=height,
        rotation=view_rotation,
        intrinsics_cls=CameraIntrinsics,
    )

    frame_id = 0
    try:
        while True:
            frames = align.process(pipeline.wait_for_frames())
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                continue
            image = rotate_array_for_view(np.asanyarray(color.get_data()), view_rotation)
            depth_m = rotate_array_for_view(
                np.asanyarray(depth.get_data()).astype(np.float32) * depth_scale, view_rotation
            )
            yield frame_id, image, depth_m, intrinsics
            frame_id += 1
    finally:
        pipeline.stop()


def draw_status(image: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    for index, (text, color) in enumerate(lines):
        y = 30 + 30 * index
        cv2.putText(image, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(image, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.66, color, 1, cv2.LINE_AA)


def draw_ghost(image: np.ndarray, corners: np.ndarray, center: np.ndarray, age_sec: float) -> None:
    cv2.polylines(image, [corners.astype(np.int32).reshape(-1, 1, 2)], True, (160, 160, 160), 2)
    cv2.circle(image, tuple(np.rint(center).astype(int)), 5, (160, 160, 160), -1)
    cv2.putText(
        image,
        f"last confident pose ({age_sec:.1f}s ago)",
        (12, image.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (160, 160, 160),
        2,
        cv2.LINE_AA,
    )


def main() -> int:
    args = parse_args()
    rim_plane = load_rim_plane(args.config)
    frames = iter_replay_frames(args) if args.replay else iter_live_frames(args)

    writer = None
    last_good: dict[str, Any] | None = None
    processed = 0
    confident = 0
    started = time.perf_counter()

    for frame_id, image, depth, intrinsics in frames:
        mask, _ = segment_yellow_box(image, keep_largest_component=False)
        estimate = estimate_plane_box(
            mask,
            depth,
            intrinsics,
            box_long_m=args.box_long_m,
            box_short_m=args.box_short_m,
            box_height_m=args.box_height_m,
            rim_plane=rim_plane,
        )
        processed += 1
        reliable = (
            estimate.center_top_camera_m is not None
            and estimate.confidence.ok
            and estimate.confidence.score >= args.min_score
        )

        vis = image.copy()
        if reliable:
            confident += 1
            vis = draw_known_size_estimate(vis, estimate)
            center = estimate.center_top_camera_m
            draw_status(
                vis,
                [
                    (
                        "center=(%.3f, %.3f, %.3f) m  yaw=%.1f deg  score=%.2f"
                        % (center[0], center[1], center[2], estimate.yaw_mod_180, estimate.confidence.score),
                        (30, 160, 30),
                    )
                ],
            )
            last_good = {
                "time": time.perf_counter(),
                "corners": np.asarray(estimate.model_corners, dtype=np.float64),
                "center": np.asarray(estimate.center_image, dtype=np.float64),
            }
        else:
            reasons = ",".join(estimate.failure_reasons[:3]) or "below_min_score"
            draw_status(vis, [("UNRELIABLE: " + reasons, (40, 40, 220))])
            if args.show_unreliable and estimate.model_corners is not None:
                corners = np.asarray(estimate.model_corners, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis, [corners], True, (40, 40, 220), 2)
            if last_good is not None:
                draw_ghost(vis, last_good["corners"], last_good["center"], time.perf_counter() - last_good["time"])

        if args.print_json:
            print(
                json.dumps(
                    {
                        "frame_id": frame_id,
                        "reliable": bool(reliable),
                        "center_top_camera_m": estimate.center_top_camera_m,
                        "yaw_mod_180": estimate.yaw_mod_180,
                        "score": estimate.confidence.score,
                        "reasons": list(estimate.failure_reasons),
                    }
                ),
                flush=True,
            )

        if args.save_video:
            if writer is None:
                writer = cv2.VideoWriter(
                    args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), 10, (vis.shape[1], vis.shape[0])
                )
            writer.write(vis)
        if not args.no_window:
            cv2.imshow("box pose inference", vis)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
        if args.max_frames is not None and processed >= args.max_frames:
            break

    if writer is not None:
        writer.release()
    if not args.no_window:
        cv2.destroyAllWindows()
    elapsed = time.perf_counter() - started
    print(
        f"frames={processed} confident={confident} ({0 if not processed else confident / processed:.0%}) "
        f"avg={0 if not processed else elapsed / processed * 1000:.1f} ms/frame"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
