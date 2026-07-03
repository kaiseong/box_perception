#!/usr/bin/env python3
"""Run live D405 box pose inference and RGB overlay visualization."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import sys
from typing import Any

import numpy as np

from recording import (
    configure_depth_sensor,
    import_realsense_sdk,
    open_realsense_pipeline,
    realsense_manifest_metadata,
    require_bgr,
)
from replay_recording import rotate_array_for_view, rotate_intrinsics_for_view


VIEW_ROTATIONS = ("none", "cw90", "ccw90", "180")


@dataclass(frozen=True)
class InferenceConfig:
    width: int
    height: int
    fps: int
    serial_number: str | None
    align_depth_to_color: bool
    enable_emitter: bool
    laser_power: float | None
    view_rotation: str
    init_frames: int
    min_init_planes: int
    max_frames: int | None
    preview: bool
    json_indent: int | None
    box_long_m: float
    box_short_m: float
    box_height_m: float
    image_fallback: bool


@dataclass(frozen=True)
class LiveFrame:
    bgr: np.ndarray
    depth_m: np.ndarray
    timestamp_ms: float | None


@dataclass(frozen=True)
class AnalysisDependencies:
    cv2: Any
    camera_intrinsics_cls: Any
    average_rim_planes: Any
    discover_rim_plane: Any
    estimate_known_size_box: Any
    estimate_plane_box: Any
    estimate_pixel_box: Any
    segment_yellow_box: Any
    draw_known_size_estimate: Any
    draw_pixel_estimate: Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run live Intel RealSense D405 box pose inference. The script initializes a "
            "short rim-plane prior, then streams RGB overlay visualization and stdout JSONL."
        )
    )
    parser.add_argument("--width", type=int, default=1280, help="Color/depth stream width.")
    parser.add_argument("--height", type=int, default=720, help="Color/depth stream height.")
    parser.add_argument("--fps", type=int, default=30, help="Requested RealSense stream FPS.")
    parser.add_argument("--serial-number", help="Optional RealSense serial number when multiple cameras exist.")
    parser.add_argument(
        "--no-align-depth",
        action="store_true",
        help="Do not align depth to color. Leave this off for normal box-pose inference.",
    )
    parser.add_argument("--disable-emitter", action="store_true", help="Disable the RealSense IR emitter.")
    parser.add_argument("--laser-power", type=float, help="Optional RealSense laser power value.")
    parser.add_argument(
        "--view-rotation",
        choices=VIEW_ROTATIONS,
        default="cw90",
        help="Rotate raw frames into the analysis orientation. Current D405 mount normally uses cw90.",
    )
    parser.add_argument(
        "--init-frames",
        type=int,
        default=15,
        help="Number of startup frames used to discover a rim-plane prior before continuous inference.",
    )
    parser.add_argument(
        "--min-init-planes",
        type=int,
        default=1,
        help="Minimum accepted rim-plane discoveries required to use the startup prior.",
    )
    parser.add_argument("--max-frames", type=int, help="Stop after this many analyzed inference frames.")
    parser.add_argument("--preview", dest="preview", action="store_true", default=True, help="Show RGB overlay window.")
    parser.add_argument("--no-preview", dest="preview", action="store_false", help="Disable RGB overlay window.")
    parser.add_argument(
        "--json-indent",
        type=int,
        help="Pretty-print JSON with this indent. Default is compact one-object-per-line JSONL.",
    )
    parser.add_argument("--box-long-m", type=float, default=0.505, help="Known box long-side length in meters.")
    parser.add_argument("--box-short-m", type=float, default=0.335, help="Known box short-side length in meters.")
    parser.add_argument("--box-height-m", type=float, default=0.195, help="Known box height in meters.")
    parser.add_argument(
        "--no-image-fallback",
        action="store_true",
        help="Do not fall back to the previous image-space known-size estimator when plane fitting fails.",
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> InferenceConfig:
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive.")
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    if args.init_frames < 0:
        raise ValueError("--init-frames must be non-negative.")
    if args.min_init_planes < 0:
        raise ValueError("--min-init-planes must be non-negative.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive when set.")
    if args.json_indent is not None and args.json_indent < 0:
        raise ValueError("--json-indent must be non-negative.")
    if args.box_long_m <= 0.0 or args.box_short_m <= 0.0 or args.box_height_m <= 0.0:
        raise ValueError("Box dimensions must be positive.")

    return InferenceConfig(
        width=int(args.width),
        height=int(args.height),
        fps=int(args.fps),
        serial_number=None if args.serial_number is None else str(args.serial_number),
        align_depth_to_color=not bool(args.no_align_depth),
        enable_emitter=not bool(args.disable_emitter),
        laser_power=None if args.laser_power is None else float(args.laser_power),
        view_rotation=str(args.view_rotation),
        init_frames=int(args.init_frames),
        min_init_planes=int(args.min_init_planes),
        max_frames=None if args.max_frames is None else int(args.max_frames),
        preview=bool(args.preview),
        json_indent=None if args.json_indent is None else int(args.json_indent),
        box_long_m=float(args.box_long_m),
        box_short_m=float(args.box_short_m),
        box_height_m=float(args.box_height_m),
        image_fallback=not bool(args.no_image_fallback),
    )


def load_analysis_dependencies() -> AnalysisDependencies:
    try:
        import cv2  # type: ignore[import-not-found]

        from box_pose import (
            CameraIntrinsics,
            average_rim_planes,
            discover_rim_plane,
            estimate_known_size_box,
            estimate_plane_box,
            estimate_pixel_box,
            segment_yellow_box,
        )
        from box_pose.visualization import draw_known_size_estimate, draw_pixel_estimate
    except Exception as exc:
        raise SystemExit(
            "Failed to import live inference dependencies. Install OpenCV and NumPy in the active "
            "environment, then run from the repository root."
        ) from exc

    return AnalysisDependencies(
        cv2=cv2,
        camera_intrinsics_cls=CameraIntrinsics,
        average_rim_planes=average_rim_planes,
        discover_rim_plane=discover_rim_plane,
        estimate_known_size_box=estimate_known_size_box,
        estimate_plane_box=estimate_plane_box,
        estimate_pixel_box=estimate_pixel_box,
        segment_yellow_box=segment_yellow_box,
        draw_known_size_estimate=draw_known_size_estimate,
        draw_pixel_estimate=draw_pixel_estimate,
    )


def read_live_frame(pipeline: Any, align: Any | None, depth_scale_m_per_unit: float) -> LiveFrame | None:
    frames = pipeline.wait_for_frames()
    if align is not None:
        frames = align.process(frames)

    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if not color_frame or not depth_frame:
        return None

    bgr = require_bgr(np.asanyarray(color_frame.get_data())).copy()
    depth_raw = np.asanyarray(depth_frame.get_data())
    depth_m = depth_raw.astype(np.float32) * float(depth_scale_m_per_unit)
    timestamp_ms = None
    try:
        timestamp_ms = float(color_frame.get_timestamp())
    except RuntimeError:
        timestamp_ms = None
    return LiveFrame(bgr=bgr, depth_m=depth_m, timestamp_ms=timestamp_ms)


def initialize_rim_plane(
    deps: AnalysisDependencies,
    *,
    rs: Any,
    pipeline: Any,
    align: Any | None,
    depth_scale_m_per_unit: float,
    intrinsics: Any,
    config: InferenceConfig,
) -> dict[str, Any] | None:
    if config.init_frames <= 0:
        return None

    accepted: list[dict[str, Any]] = []
    for burst_index in range(config.init_frames):
        live = read_live_frame(pipeline, align, depth_scale_m_per_unit)
        if live is None:
            continue
        image = rotate_array_for_view(live.bgr, config.view_rotation)
        depth = rotate_array_for_view(live.depth_m, config.view_rotation)
        evidence_mask, _ = deps.segment_yellow_box(image, keep_largest_component=False)
        plane = deps.discover_rim_plane(evidence_mask, depth, intrinsics, box_short_m=config.box_short_m)
        if plane is not None:
            plane["burst_index"] = burst_index
            accepted.append(plane)

    if len(accepted) < config.min_init_planes:
        return None
    calibration = deps.average_rim_planes(accepted)
    calibration["accepted_burst_indices"] = [plane["burst_index"] for plane in accepted]
    return calibration


def analyze_live_frame(
    deps: AnalysisDependencies,
    *,
    frame_id: int,
    live: LiveFrame,
    intrinsics: Any,
    config: InferenceConfig,
    rim_plane: tuple[Any, Any] | None,
    plane_prior: dict[str, Any] | None,
) -> tuple[dict[str, Any], np.ndarray]:
    image = rotate_array_for_view(live.bgr, config.view_rotation)
    depth = rotate_array_for_view(live.depth_m, config.view_rotation)
    mask, mask_stats = deps.segment_yellow_box(image)
    evidence_mask, evidence_stats = deps.segment_yellow_box(image, keep_largest_component=False)
    pixel_estimate = deps.estimate_pixel_box(mask, mask_stats)

    estimate = deps.estimate_plane_box(
        evidence_mask,
        depth,
        intrinsics,
        box_long_m=config.box_long_m,
        box_short_m=config.box_short_m,
        box_height_m=config.box_height_m,
        rim_plane=rim_plane,
    )
    method = "plane"
    plane_failure_reasons: list[str] = []
    if estimate.center_top_camera_m is None and config.image_fallback:
        plane_failure_reasons = list(estimate.failure_reasons)
        estimate = deps.estimate_known_size_box(
            evidence_mask,
            depth,
            intrinsics,
            box_long_m=config.box_long_m,
            box_short_m=config.box_short_m,
            box_height_m=config.box_height_m,
        )
        method = "image_fallback"

    known_size = estimate.to_dict()
    known_size["method"] = method
    known_size["mask_stats"] = evidence_stats.to_dict()
    if plane_failure_reasons:
        known_size["plane_failure_reasons"] = plane_failure_reasons

    record = build_pose_record(
        frame_id=frame_id,
        timestamp_ms=live.timestamp_ms,
        view_rotation=config.view_rotation,
        known_size=known_size,
        plane_prior=plane_prior,
    )
    overlay = deps.draw_pixel_estimate(image, pixel_estimate)
    overlay = deps.draw_known_size_estimate(overlay, estimate)
    draw_live_status(deps.cv2, overlay, record)
    return record, overlay


def build_pose_record(
    *,
    frame_id: int,
    timestamp_ms: float | None,
    view_rotation: str,
    known_size: dict[str, Any],
    plane_prior: dict[str, Any] | None,
) -> dict[str, Any]:
    confidence = known_size.get("confidence") or {}
    support = known_size.get("support") or {}
    ok = bool(confidence.get("ok")) and known_size.get("center_top_camera_m") is not None
    return json_safe(
        {
            "frame_id": int(frame_id),
            "timestamp_ms": timestamp_ms,
            "view_rotation": view_rotation,
            "ok": ok,
            "method": known_size.get("method"),
            "center_top_camera_m": known_size.get("center_top_camera_m") if ok else None,
            "yaw_mod_180": known_size.get("yaw_mod_180") if ok else None,
            "long_axis_camera": support.get("long_axis_camera") if ok else None,
            "short_axis_camera": support.get("short_axis_camera") if ok else None,
            "center_image": known_size.get("center_image") if ok else None,
            "confidence": confidence,
            "failure_reasons": known_size.get("failure_reasons") or confidence.get("reasons") or [],
            "plane_failure_reasons": known_size.get("plane_failure_reasons") or [],
            "plane_prior": plane_prior_status(plane_prior),
        }
    )


def plane_prior_status(plane_prior: dict[str, Any] | None) -> dict[str, Any]:
    if plane_prior is None:
        return {
            "available": False,
            "mode": "per_frame_discovery",
            "frames_used": 0,
            "frames_discovered": 0,
            "offset_spread_m": None,
        }
    return {
        "available": True,
        "mode": "startup_burst",
        "frames_used": int(plane_prior.get("frames_used", 0)),
        "frames_discovered": int(plane_prior.get("frames_discovered", 0)),
        "offset_spread_m": plane_prior.get("offset_spread_m"),
    }


def draw_live_status(cv2_module: Any, overlay: np.ndarray, record: dict[str, Any]) -> None:
    ok = bool(record["ok"])
    color = (0, 220, 0) if ok else (0, 0, 255)
    if ok:
        center = record["center_top_camera_m"]
        text = (
            f"OK yaw={record['yaw_mod_180']:.1f} "
            f"center=({center[0]:.3f},{center[1]:.3f},{center[2]:.3f})m"
        )
    else:
        reasons = ",".join(str(item) for item in record.get("failure_reasons", [])[:3])
        text = f"NO POSE {reasons}"
    cv2_module.putText(overlay, text, (20, 170), cv2_module.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 3, cv2_module.LINE_AA)
    cv2_module.putText(overlay, text, (20, 170), cv2_module.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2_module.LINE_AA)

    prior = record["plane_prior"]
    prior_text = (
        f"plane_prior={prior['mode']} "
        f"frames={prior['frames_used']}/{prior['frames_discovered']}"
    )
    cv2_module.putText(
        overlay,
        prior_text,
        (20, 200),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        3,
        cv2_module.LINE_AA,
    )
    cv2_module.putText(
        overlay,
        prior_text,
        (20, 200),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 0),
        1,
        cv2_module.LINE_AA,
    )


def print_pose_record(record: dict[str, Any], *, indent: int | None) -> None:
    print(json.dumps(record, sort_keys=True, indent=indent), flush=True)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def run_live_inference(config: InferenceConfig) -> int:
    deps = load_analysis_dependencies()
    rs = import_realsense_sdk()
    pipeline = None

    try:
        pipeline, profile = open_realsense_pipeline(rs, config)
        depth_settings = configure_depth_sensor(rs, profile, config)
        align = rs.align(rs.stream.color) if config.align_depth_to_color else None

        _, intrinsics_mapping, _, _, _ = realsense_manifest_metadata(rs, profile, depth_settings)
        raw_intrinsics = deps.camera_intrinsics_cls.from_mapping(intrinsics_mapping)
        intrinsics = rotate_intrinsics_for_view(
            raw_intrinsics,
            width=int(intrinsics_mapping.get("width", config.width)),
            height=int(intrinsics_mapping.get("height", config.height)),
            rotation=config.view_rotation,
            intrinsics_cls=deps.camera_intrinsics_cls,
        )

        print(
            f"Initializing rim plane from {config.init_frames} frame(s)...",
            file=sys.stderr,
            flush=True,
        )
        plane_prior = initialize_rim_plane(
            deps,
            rs=rs,
            pipeline=pipeline,
            align=align,
            depth_scale_m_per_unit=float(depth_settings["depth_scale_m_per_unit"]),
            intrinsics=intrinsics,
            config=config,
        )
        rim_plane = None
        if plane_prior is None:
            print(
                "WARNING: rim plane startup initialization failed; continuing with per-frame discovery.",
                file=sys.stderr,
                flush=True,
            )
        else:
            rim_plane = (plane_prior["normal"], plane_prior["point"])
            print(
                "Rim plane initialized from "
                f"{plane_prior['frames_used']} frame(s), offset spread "
                f"{plane_prior['offset_spread_m'] * 1000.0:.1f} mm.",
                file=sys.stderr,
                flush=True,
            )

        frame_id = 0
        while config.max_frames is None or frame_id < config.max_frames:
            live = read_live_frame(pipeline, align, float(depth_settings["depth_scale_m_per_unit"]))
            if live is None:
                continue
            record, overlay = analyze_live_frame(
                deps,
                frame_id=frame_id,
                live=live,
                intrinsics=intrinsics,
                config=config,
                rim_plane=rim_plane,
                plane_prior=plane_prior,
            )
            print_pose_record(record, indent=config.json_indent)

            if config.preview:
                deps.cv2.imshow("KETI D405 box inference", overlay)
                if deps.cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_id += 1
    except KeyboardInterrupt:
        print("Inference stopped by user.", file=sys.stderr)
    finally:
        if pipeline is not None:
            pipeline.stop()
        if config.preview:
            deps.cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    config = config_from_args(parse_args(argv))
    return run_live_inference(config)


if __name__ == "__main__":
    raise SystemExit(main())
