#!/usr/bin/env python3
"""Replay a D405 recording session and estimate box yaw frame by frame."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


VIEW_ROTATIONS = ("none", "cw90", "ccw90", "180")


@dataclass(frozen=True)
class AnalysisDependencies:
    cv2: Any
    camera_intrinsics_cls: Any
    estimate_metric_box: Any
    estimate_pixel_box: Any
    evaluate_still_frame_spread: Any
    segment_yellow_box: Any
    draw_pixel_estimate: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a recording.py session and write per-frame box yaw/center estimates."
    )
    parser.add_argument(
        "session",
        nargs="?",
        default="recordings/d405_box_static_001",
        help="Recording session directory containing manifest.json and index.jsonl.",
    )
    parser.add_argument("--output-dir", help="Analysis output directory. Defaults to <session>/analysis.")
    parser.add_argument("--max-frames", type=int, help="Maximum number of indexed frames to analyze.")
    parser.add_argument("--stride", type=int, default=1, help="Analyze every Nth indexed frame.")
    parser.add_argument(
        "--debug-every",
        type=int,
        default=10,
        help="Save one debug overlay every N analyzed frames. Use 0 to disable debug images.",
    )
    parser.add_argument("--save-mask", action="store_true", help="Save binary masks next to debug overlays.")
    parser.add_argument("--no-metric", action="store_true", help="Skip depth/intrinsics metric estimation.")
    parser.add_argument(
        "--view-rotation",
        choices=("auto",) + VIEW_ROTATIONS,
        default="auto",
        help=(
            "Rotate raw RGB/depth frames into the analysis image orientation. "
            "auto uses manifest config.view_rotation when present. Use cw90 for the current "
            "rotated D405 mount."
        ),
    )
    return parser.parse_args()


def load_analysis_dependencies() -> AnalysisDependencies:
    try:
        import cv2  # type: ignore[import-not-found]

        from box_pose import (
            CameraIntrinsics,
            estimate_metric_box,
            estimate_pixel_box,
            evaluate_still_frame_spread,
            segment_yellow_box,
        )
        from box_pose.visualization import draw_pixel_estimate
    except Exception as exc:
        raise SystemExit(
            "Failed to import OpenCV-based analysis dependencies. "
            "Run replay_recording.py in the offline analysis environment with OpenCV and NumPy installed. "
            "The RealSense recording environment can be reused if cv2 imports cleanly there."
        ) from exc

    return AnalysisDependencies(
        cv2=cv2,
        camera_intrinsics_cls=CameraIntrinsics,
        estimate_metric_box=estimate_metric_box,
        estimate_pixel_box=estimate_pixel_box,
        evaluate_still_frame_spread=evaluate_still_frame_spread,
        segment_yellow_box=segment_yellow_box,
        draw_pixel_estimate=draw_pixel_estimate,
    )


def load_manifest(session_dir: Path) -> dict[str, Any]:
    path = session_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing recording manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def iter_index_records(session_dir: Path, *, stride: int = 1, max_frames: int | None = None) -> list[dict[str, Any]]:
    if stride <= 0:
        raise ValueError("--stride must be positive.")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("--max-frames must be positive when set.")

    index_path = session_dir / "index.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing recording index: {index_path}")

    selected: list[dict[str, Any]] = []
    for index, line in enumerate(index_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        if index % stride != 0:
            continue
        selected.append(json.loads(line))
        if max_frames is not None and len(selected) >= max_frames:
            break
    return selected


def load_rgb_frame(session_dir: Path, record: dict[str, Any], cv2_module: Any) -> np.ndarray:
    path = session_dir / record["rgb_path"]
    if path.suffix == ".npy":
        image = np.load(path)
    else:
        image = cv2_module.imread(str(path), cv2_module.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read RGB image: {path}")
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"RGB frame must have shape HxWx3, got {image.shape} from {path}")
    return image


def load_depth_frame(session_dir: Path, record: dict[str, Any]) -> np.ndarray:
    path = session_dir / record["depth_path"]
    if path.suffix == ".npz":
        with np.load(path) as data:
            depth = data["depth_m"]
    else:
        depth = np.load(path)
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Depth frame must be 2D, got {depth.shape} from {path}")
    return depth


def resolve_view_rotation(manifest: dict[str, Any], requested: str) -> str:
    if requested != "auto":
        return normalize_view_rotation(requested)
    config_rotation = manifest.get("config", {}).get("view_rotation")
    layout_rotation = manifest.get("data_layout", {}).get("view_rotation_from_raw_to_analysis")
    return normalize_view_rotation(config_rotation or layout_rotation or "none")


def normalize_view_rotation(rotation: str | None) -> str:
    value = "none" if rotation is None else str(rotation).lower()
    if value not in VIEW_ROTATIONS:
        raise ValueError(f"Unsupported view rotation: {rotation!r}")
    return value


def rotate_array_for_view(array: np.ndarray, rotation: str) -> np.ndarray:
    normalized = normalize_view_rotation(rotation)
    if normalized == "none":
        return array
    if normalized == "cw90":
        return np.rot90(array, k=3).copy()
    if normalized == "ccw90":
        return np.rot90(array, k=1).copy()
    if normalized == "180":
        return np.rot90(array, k=2).copy()
    raise AssertionError(f"unreachable rotation: {normalized}")


def image_size_from_manifest_or_record(manifest: dict[str, Any], record: dict[str, Any]) -> tuple[int, int]:
    intrinsics = manifest.get("intrinsics", {})
    if "width" in intrinsics and "height" in intrinsics:
        return int(intrinsics["width"]), int(intrinsics["height"])
    config = manifest.get("config", {})
    if "width" in config and "height" in config:
        return int(config["width"]), int(config["height"])
    image_shape = record.get("image_shape")
    if isinstance(image_shape, list) and len(image_shape) >= 2:
        return int(image_shape[1]), int(image_shape[0])
    raise ValueError("Cannot determine raw image size for rotated intrinsics.")


def rotate_intrinsics_for_view(intrinsics: Any, *, width: int, height: int, rotation: str, intrinsics_cls: Any) -> Any:
    normalized = normalize_view_rotation(rotation)
    if normalized == "none":
        return intrinsics
    if normalized == "cw90":
        return intrinsics_cls(fx=intrinsics.fy, fy=intrinsics.fx, cx=(height - 1.0) - intrinsics.cy, cy=intrinsics.cx)
    if normalized == "ccw90":
        return intrinsics_cls(fx=intrinsics.fy, fy=intrinsics.fx, cx=intrinsics.cy, cy=(width - 1.0) - intrinsics.cx)
    if normalized == "180":
        return intrinsics_cls(fx=intrinsics.fx, fy=intrinsics.fy, cx=(width - 1.0) - intrinsics.cx, cy=(height - 1.0) - intrinsics.cy)
    raise AssertionError(f"unreachable rotation: {normalized}")


def analyze_frame(
    deps: AnalysisDependencies,
    *,
    session_dir: Path,
    record: dict[str, Any],
    intrinsics: Any | None,
    output_dir: Path,
    save_debug: bool,
    save_mask: bool,
    run_metric: bool,
    view_rotation: str,
) -> dict[str, Any]:
    image = rotate_array_for_view(load_rgb_frame(session_dir, record, deps.cv2), view_rotation)
    mask, mask_stats = deps.segment_yellow_box(image)
    pixel_estimate = deps.estimate_pixel_box(mask, mask_stats)

    result: dict[str, Any] = {
        "frame_id": int(record["frame_id"]),
        "wall_time": record.get("wall_time"),
        "rgb_path": record.get("rgb_path"),
        "depth_path": record.get("depth_path"),
        "view_rotation": view_rotation,
        "analysis_image_shape": list(image.shape),
        "pixel": pixel_estimate.to_dict(),
    }

    if run_metric and intrinsics is not None:
        depth = rotate_array_for_view(load_depth_frame(session_dir, record), view_rotation)
        metric_estimate = deps.estimate_metric_box(mask, depth, intrinsics)
        metric = metric_estimate.to_dict()
        metric["long_length_m"] = metric_estimate.long_length_m
        metric["short_length_m"] = metric_estimate.short_length_m
        metric["center_camera_m"] = (
            None if metric_estimate.camera_T_box is None else metric_estimate.camera_T_box[:3, 3].tolist()
        )
        result["metric"] = metric

    if save_debug:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        overlay = deps.draw_pixel_estimate(image, pixel_estimate)
        deps.cv2.imwrite(str(debug_dir / f"frame_{int(record['frame_id']):06d}_debug.png"), overlay)

    if save_mask:
        mask_dir = output_dir / "mask"
        mask_dir.mkdir(parents=True, exist_ok=True)
        deps.cv2.imwrite(str(mask_dir / f"frame_{int(record['frame_id']):06d}_mask.png"), mask)

    return result


def summarize_results(results: list[dict[str, Any]], evaluate_still_frame_spread: Any | None = None) -> dict[str, Any]:
    pixel_ok = [item for item in results if item["pixel"]["confidence"]["ok"]]
    metric_ok = [
        item
        for item in results
        if item.get("metric", {}).get("confidence", {}).get("ok") and item["metric"].get("center_camera_m") is not None
    ]

    summary: dict[str, Any] = {
        "frames_analyzed": len(results),
        "view_rotation": None if not results else results[0].get("view_rotation"),
        "pixel_ok_frames": len(pixel_ok),
        "pixel_ok_fraction": 0.0 if not results else len(pixel_ok) / len(results),
        "metric_ok_frames": len(metric_ok),
        "metric_ok_fraction": 0.0 if not results else len(metric_ok) / len(results),
        "pixel_yaw_mod_180": yaw_summary([item["pixel"]["yaw_mod_180"] for item in pixel_ok]),
    }

    if metric_ok:
        centers = np.asarray([item["metric"]["center_camera_m"] for item in metric_ok], dtype=np.float64)
        yaws = np.asarray([item["metric"]["yaw_mod_180"] for item in metric_ok], dtype=np.float64)
        summary["metric_center_camera_m_mean"] = centers.mean(axis=0).tolist()
        summary["metric_yaw_mod_180"] = yaw_summary(yaws.tolist())
        if evaluate_still_frame_spread is not None and centers.shape[0] >= 2:
            spread_conf = evaluate_still_frame_spread(centers, yaws)
            summary["metric_still_frame_spread"] = spread_conf.to_dict()
    return summary


def yaw_summary(yaws_deg: list[float]) -> dict[str, Any] | None:
    values = np.asarray([value for value in yaws_deg if np.isfinite(value)], dtype=np.float64)
    if values.size == 0:
        return None
    mean = circular_mean_mod_180(values)
    return {
        "mean_deg": mean,
        "min_deg": float(values.min()),
        "max_deg": float(values.max()),
        "spread_deg": angular_spread_mod_180(values),
    }


def circular_mean_mod_180(values_deg: np.ndarray) -> float:
    doubled = np.deg2rad(values_deg * 2.0)
    mean = np.rad2deg(np.arctan2(np.sin(doubled).mean(), np.cos(doubled).mean())) / 2.0
    return float(mean % 180.0)


def angular_spread_mod_180(values_deg: np.ndarray) -> float:
    if values_deg.size < 2:
        return 0.0
    spread = 0.0
    for i, first in enumerate(values_deg):
        for second in values_deg[i + 1 :]:
            spread = max(spread, angular_distance_mod_180(float(first), float(second)))
    return float(spread)


def angular_distance_mod_180(first: float, second: float) -> float:
    return abs(((first - second + 90.0) % 180.0) - 90.0)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    deps = load_analysis_dependencies()

    session_dir = Path(args.session)
    output_dir = Path(args.output_dir) if args.output_dir else session_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(session_dir)
    records = iter_index_records(session_dir, stride=args.stride, max_frames=args.max_frames)
    if not records:
        raise SystemExit(f"No frames selected from {session_dir / 'index.jsonl'}")

    view_rotation = resolve_view_rotation(manifest, args.view_rotation)
    intrinsics = None
    if not args.no_metric:
        raw_width, raw_height = image_size_from_manifest_or_record(manifest, records[0])
        raw_intrinsics = deps.camera_intrinsics_cls.from_mapping(manifest["intrinsics"])
        intrinsics = rotate_intrinsics_for_view(
            raw_intrinsics,
            width=raw_width,
            height=raw_height,
            rotation=view_rotation,
            intrinsics_cls=deps.camera_intrinsics_cls,
        )

    frame_records_path = output_dir / "frames.jsonl"
    results: list[dict[str, Any]] = []
    with frame_records_path.open("w", encoding="utf-8") as frame_file:
        for analyzed_index, record in enumerate(records):
            save_debug = args.debug_every > 0 and analyzed_index % args.debug_every == 0
            result = analyze_frame(
                deps,
                session_dir=session_dir,
                record=record,
                intrinsics=intrinsics,
                output_dir=output_dir,
                save_debug=save_debug,
                save_mask=save_debug and args.save_mask,
                run_metric=not args.no_metric,
                view_rotation=view_rotation,
            )
            results.append(result)
            frame_file.write(json.dumps(result, sort_keys=True) + "\n")

    summary = summarize_results(results, deps.evaluate_still_frame_spread)
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {frame_records_path}")
    return 0 if summary["pixel_ok_frames"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
