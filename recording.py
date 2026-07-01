#!/usr/bin/env python3
"""Record synchronized ZED RGB/depth frames for offline box-pose analysis."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np


FORMAT_VERSION = "box-perception-recording-v1"


@dataclass(frozen=True)
class RecordingConfig:
    output_root: str
    session_name: str
    fps: int
    resolution: str
    depth_mode: str
    max_frames: int | None
    duration_sec: float | None
    warmup_frames: int
    rgb_format: str
    depth_format: str
    jpeg_quality: int
    preview: bool
    serial_number: int | None


@dataclass(frozen=True)
class SessionPaths:
    session_dir: Path
    rgb_dir: Path
    depth_dir: Path
    index_path: Path
    manifest_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record ZED left RGB frames, depth maps, intrinsics, and timestamps for offline box-pose tests."
    )
    parser.add_argument("--output-root", default="KETI/recordings", help="Directory where recording sessions are stored.")
    parser.add_argument("--session-name", help="Session directory name. Defaults to zed_YYYYMMDDTHHMMSSZ.")
    parser.add_argument("--fps", type=int, default=10, help="Requested ZED camera FPS.")
    parser.add_argument(
        "--resolution",
        choices=("HD2K", "HD1080", "HD720", "VGA"),
        default="HD720",
        help="Requested ZED camera resolution.",
    )
    parser.add_argument(
        "--depth-mode",
        choices=("ULTRA", "QUALITY", "PERFORMANCE", "NEURAL"),
        default="ULTRA",
        help="Requested ZED depth mode. ULTRA avoids NEURAL/TensorRT dependency surprises.",
    )
    parser.add_argument("--max-frames", type=int, help="Stop after this many saved frames.")
    parser.add_argument("--duration-sec", type=float, help="Stop after this many seconds.")
    parser.add_argument("--warmup-frames", type=int, default=10, help="ZED frames to grab before saving.")
    parser.add_argument("--rgb-format", choices=("jpg", "png"), default="jpg", help="RGB frame image format.")
    parser.add_argument("--depth-format", choices=("npz", "npy"), default="npz", help="Depth map storage format.")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality when --rgb-format jpg is used.")
    parser.add_argument("--preview", action="store_true", help="Show a live preview window. Press q to stop.")
    parser.add_argument("--serial-number", type=int, help="Optional ZED serial number when multiple cameras exist.")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> RecordingConfig:
    if args.max_frames is None and args.duration_sec is None:
        raise ValueError("Set at least one stop condition: --max-frames or --duration-sec.")
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive.")
    if args.duration_sec is not None and args.duration_sec <= 0.0:
        raise ValueError("--duration-sec must be positive.")
    if args.warmup_frames < 0:
        raise ValueError("--warmup-frames must be non-negative.")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be in [1, 100].")

    return RecordingConfig(
        output_root=str(args.output_root),
        session_name=args.session_name or default_session_name(),
        fps=int(args.fps),
        resolution=str(args.resolution),
        depth_mode=str(args.depth_mode),
        max_frames=None if args.max_frames is None else int(args.max_frames),
        duration_sec=None if args.duration_sec is None else float(args.duration_sec),
        warmup_frames=int(args.warmup_frames),
        rgb_format=str(args.rgb_format),
        depth_format=str(args.depth_format),
        jpeg_quality=int(args.jpeg_quality),
        preview=bool(args.preview),
        serial_number=None if args.serial_number is None else int(args.serial_number),
    )


def default_session_name(now: datetime | None = None) -> str:
    stamp = now or datetime.now(timezone.utc)
    return stamp.astimezone(timezone.utc).strftime("zed_%Y%m%dT%H%M%SZ")


def prepare_session(output_root: str | Path, session_name: str) -> SessionPaths:
    root = Path(output_root)
    session_dir = root / session_name
    if session_dir.exists():
        raise FileExistsError(f"Recording session already exists: {session_dir}")

    rgb_dir = session_dir / "rgb"
    depth_dir = session_dir / "depth"
    rgb_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)
    return SessionPaths(
        session_dir=session_dir,
        rgb_dir=rgb_dir,
        depth_dir=depth_dir,
        index_path=session_dir / "index.jsonl",
        manifest_path=session_dir / "manifest.json",
    )


def build_manifest(
    config: RecordingConfig,
    *,
    camera_info: dict[str, Any],
    intrinsics: dict[str, float],
    started_at: str,
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "created_at": started_at,
        "recording": {
            "frame_count": 0,
            "finished_at": None,
            "duration_sec_actual": None,
        },
        "config": asdict(config),
        "camera": camera_info,
        "intrinsics": intrinsics,
        "data_layout": {
            "index": "index.jsonl",
            "rgb_dir": "rgb",
            "depth_dir": "depth",
            "rgb_frame_pattern": "rgb/frame_000000.<jpg|png>",
            "depth_frame_pattern": "depth/frame_000000.depth.<npz|npy>",
            "depth_units": "meter",
            "rgb_color_order": "BGR as saved by OpenCV",
        },
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_frame(
    paths: SessionPaths,
    *,
    frame_id: int,
    bgr: np.ndarray,
    depth_m: np.ndarray,
    rgb_format: str,
    depth_format: str,
    jpeg_quality: int,
    wall_time: str,
    monotonic_time_sec: float,
) -> dict[str, Any]:
    rgb_name = f"frame_{frame_id:06d}.{rgb_format}"
    depth_name = f"frame_{frame_id:06d}.depth.{depth_format}"
    rgb_path = paths.rgb_dir / rgb_name
    depth_path = paths.depth_dir / depth_name

    image = require_bgr(bgr)
    depth = require_depth(depth_m)

    if rgb_format == "jpg":
        ok = cv2.imwrite(str(rgb_path), image, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
    elif rgb_format == "png":
        ok = cv2.imwrite(str(rgb_path), image)
    else:
        raise ValueError(f"Unsupported rgb_format: {rgb_format}")
    if not ok:
        raise OSError(f"Failed to write RGB frame: {rgb_path}")

    if depth_format == "npz":
        np.savez_compressed(depth_path, depth_m=depth.astype(np.float32, copy=False))
    elif depth_format == "npy":
        np.save(depth_path, depth.astype(np.float32, copy=False))
    else:
        raise ValueError(f"Unsupported depth_format: {depth_format}")

    record = {
        "frame_id": int(frame_id),
        "wall_time": wall_time,
        "monotonic_time_sec": float(monotonic_time_sec),
        "rgb_path": str(Path("rgb") / rgb_name),
        "depth_path": str(Path("depth") / depth_name),
        "image_shape": list(image.shape),
        "depth_shape": list(depth.shape),
        "depth_stats": depth_statistics(depth),
    }
    append_index(paths.index_path, record)
    return record


def append_index(index_path: Path, record: dict[str, Any]) -> None:
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def depth_statistics(depth_m: np.ndarray) -> dict[str, int | float | None]:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        return {"valid_count": 0, "min_m": None, "max_m": None, "mean_m": None}
    values = depth[valid]
    return {
        "valid_count": valid_count,
        "min_m": float(np.min(values)),
        "max_m": float(np.max(values)),
        "mean_m": float(np.mean(values)),
    }


def require_bgr(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"bgr image must have shape HxWx3, got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def require_depth(depth_m: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth_m must have shape HxW, got {depth.shape}")
    return depth


def zed_image_to_bgr(image_data: np.ndarray) -> np.ndarray:
    image = np.asarray(image_data)
    if image.ndim != 3:
        raise ValueError(f"ZED image data must have shape HxWxC, got {image.shape}")
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.shape[2] == 3:
        return image.copy()
    raise ValueError(f"Unsupported ZED image channel count: {image.shape[2]}")


def import_zed_sdk() -> Any:
    try:
        import pyzed.sl as sl  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "Failed to import pyzed.sl. Install the Stereolabs ZED SDK Python API on the Jetson, "
            "or run only the offline/demo/test scripts that do not need ZED."
        ) from exc
    return sl


def enum_value(enum_container: Any, name: str) -> Any:
    if not hasattr(enum_container, name):
        options = [item for item in dir(enum_container) if item.isupper()]
        raise ValueError(f"Unsupported ZED enum value {name}. Available values: {options}")
    return getattr(enum_container, name)


def open_zed_camera(sl: Any, config: RecordingConfig) -> Any:
    init_params = sl.InitParameters()
    init_params.camera_resolution = enum_value(sl.RESOLUTION, config.resolution)
    init_params.camera_fps = int(config.fps)
    init_params.depth_mode = enum_value(sl.DEPTH_MODE, config.depth_mode)
    init_params.coordinate_units = sl.UNIT.METER
    if config.serial_number is not None:
        init_params.set_from_serial_number(int(config.serial_number))

    camera = sl.Camera()
    status = camera.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(f"Failed to open ZED camera: {status}")
    return camera


def zed_camera_metadata(camera: Any) -> tuple[dict[str, Any], dict[str, float]]:
    info = camera.get_camera_information()
    config = info.camera_configuration
    left_cam = config.calibration_parameters.left_cam
    intrinsics = {
        "fx": float(left_cam.fx),
        "fy": float(left_cam.fy),
        "cx": float(left_cam.cx),
        "cy": float(left_cam.cy),
    }
    camera_info = {
        "serial_number": safe_attr(info, "serial_number"),
        "camera_model": str(safe_attr(info, "camera_model")),
        "camera_firmware_version": safe_attr(info, "camera_firmware_version"),
        "resolution": {
            "width": int(config.resolution.width),
            "height": int(config.resolution.height),
        },
        "fps": safe_attr(config, "fps"),
    }
    return camera_info, intrinsics


def safe_attr(obj: Any, name: str) -> Any:
    value = getattr(obj, name, None)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def record_zed(config: RecordingConfig) -> Path:
    sl = import_zed_sdk()
    paths = prepare_session(config.output_root, config.session_name)
    camera = open_zed_camera(sl, config)
    started_at = utc_now_iso()
    start_monotonic = time.monotonic()
    frame_count = 0

    try:
        camera_info, intrinsics = zed_camera_metadata(camera)
        manifest = build_manifest(config, camera_info=camera_info, intrinsics=intrinsics, started_at=started_at)
        write_json(paths.manifest_path, manifest)

        runtime = sl.RuntimeParameters()
        image_mat = sl.Mat()
        depth_mat = sl.Mat()

        for _ in range(config.warmup_frames):
            camera.grab(runtime)

        print(f"Recording session: {paths.session_dir}")
        print("Press Ctrl-C to stop." + (" Press q in preview window to stop." if config.preview else ""))

        while should_continue(frame_count, time.monotonic() - start_monotonic, config):
            status = camera.grab(runtime)
            if status != sl.ERROR_CODE.SUCCESS:
                continue

            camera.retrieve_image(image_mat, sl.VIEW.LEFT)
            camera.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)

            bgr = zed_image_to_bgr(np.array(image_mat.get_data(), copy=True))
            depth_m = np.array(depth_mat.get_data(), dtype=np.float32, copy=True)
            now_monotonic = time.monotonic()
            save_frame(
                paths,
                frame_id=frame_count,
                bgr=bgr,
                depth_m=depth_m,
                rgb_format=config.rgb_format,
                depth_format=config.depth_format,
                jpeg_quality=config.jpeg_quality,
                wall_time=utc_now_iso(),
                monotonic_time_sec=now_monotonic,
            )

            frame_count += 1
            if frame_count == 1 or frame_count % 10 == 0:
                print(f"Saved {frame_count} frames -> {paths.session_dir}")

            if config.preview:
                cv2.imshow("KETI ZED recording", bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
    finally:
        duration_sec = time.monotonic() - start_monotonic
        try:
            camera.close()
        finally:
            if config.preview:
                cv2.destroyAllWindows()

        if paths.manifest_path.exists():
            manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
            manifest["recording"]["frame_count"] = int(frame_count)
            manifest["recording"]["finished_at"] = utc_now_iso()
            manifest["recording"]["duration_sec_actual"] = float(duration_sec)
            write_json(paths.manifest_path, manifest)

    print(f"Done. Saved {frame_count} frames in {paths.session_dir}")
    return paths.session_dir


def should_continue(frame_count: int, elapsed_sec: float, config: RecordingConfig) -> bool:
    if config.max_frames is not None and frame_count >= config.max_frames:
        return False
    if config.duration_sec is not None and elapsed_sec >= config.duration_sec:
        return False
    return True


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def main() -> int:
    config = config_from_args(parse_args())
    record_zed(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
