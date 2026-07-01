"""Geometry primitives for box yaw and center estimation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import cv2
import numpy as np

from .segmentation import MaskStats


FORBIDDEN_OUTPUT_FIELDS = {
    "address",
    "command_recommended",
    "power",
    "servo",
    "target_t5_T_ee",
}


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "CameraIntrinsics":
        if "camera_matrix" in data:
            matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
            return cls(float(matrix[0, 0]), float(matrix[1, 1]), float(matrix[0, 2]), float(matrix[1, 2]))
        return cls(float(data["fx"]), float(data["fy"]), float(data["cx"]), float(data["cy"]))


@dataclass(frozen=True)
class Confidence:
    ok: bool
    score: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "score": self.score, "reasons": list(self.reasons)}


@dataclass(frozen=True)
class PixelBoxEstimate:
    pixel_obb: dict[str, Any]
    mask_stats: MaskStats
    confidence: Confidence
    yaw_mod_180: float
    long_axis_image: tuple[float, float]
    grasp_axis_image: tuple[float, float]
    failure_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return safe_output_dict(
            {
                "pixel_obb": self.pixel_obb,
                "mask_stats": self.mask_stats.to_dict(),
                "confidence": self.confidence.to_dict(),
                "yaw_mod_180": self.yaw_mod_180,
                "yaw_frame": "image",
                "long_axis_image": list(self.long_axis_image),
                "grasp_axis_image": list(self.grasp_axis_image),
                "failure_reasons": list(self.failure_reasons),
            }
        )


@dataclass(frozen=True)
class MetricBoxEstimate:
    camera_T_box: np.ndarray | None
    t5_T_box: np.ndarray | None
    long_axis_camera: tuple[float, float, float] | None
    grasp_axis_camera: tuple[float, float, float] | None
    long_axis_t5: tuple[float, float, float] | None
    grasp_axis_t5: tuple[float, float, float] | None
    yaw_mod_180: float | None
    yaw_frame: str
    long_length_m: float | None
    short_length_m: float | None
    confidence: Confidence
    failure_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return safe_output_dict(
            {
                "camera_T_box": None if self.camera_T_box is None else self.camera_T_box.tolist(),
                "t5_T_box": None if self.t5_T_box is None else self.t5_T_box.tolist(),
                "long_axis_camera": None if self.long_axis_camera is None else list(self.long_axis_camera),
                "grasp_axis_camera": None if self.grasp_axis_camera is None else list(self.grasp_axis_camera),
                "long_axis_t5": None if self.long_axis_t5 is None else list(self.long_axis_t5),
                "grasp_axis_t5": None if self.grasp_axis_t5 is None else list(self.grasp_axis_t5),
                "yaw_mod_180": self.yaw_mod_180,
                "yaw_frame": self.yaw_frame,
                "long_length_m": self.long_length_m,
                "short_length_m": self.short_length_m,
                "confidence": self.confidence.to_dict(),
                "failure_reasons": list(self.failure_reasons),
            }
        )


def estimate_pixel_box(
    mask: np.ndarray,
    mask_stats: MaskStats,
    *,
    min_area_fraction: float = 0.03,
    min_dominant_fraction: float = 0.70,
    min_aspect_ratio: float = 1.25,
    min_fill_ratio: float = 0.45,
    max_significant_components: int = 3,
) -> PixelBoxEstimate:
    """Estimate image-frame box center and yaw from a binary mask."""

    mask_u8 = _require_mask(mask)
    contour = largest_contour(mask_u8)
    if contour is None:
        confidence = Confidence(False, 0.0, ("no_contour",))
        return PixelBoxEstimate({}, mask_stats, confidence, math.nan, (math.nan, math.nan), (math.nan, math.nan), confidence.reasons)

    rect = cv2.minAreaRect(contour)
    corners = cv2.boxPoints(rect).astype(np.float64)
    center, long_axis, short_axis, long_len, short_len = _axes_from_corners(corners)
    yaw = yaw_mod_180_from_axis(long_axis)
    obb_area = max(float(long_len * short_len), 1e-9)
    contour_area = float(cv2.contourArea(contour))
    support_area = float(min(contour_area, mask_stats.dominant_area))
    fill_ratio = support_area / obb_area
    aspect_ratio = float(long_len / max(short_len, 1e-9))

    reasons: list[str] = []
    if mask_stats.image_area <= 0 or mask_stats.dominant_area / mask_stats.image_area < min_area_fraction:
        reasons.append("mask_area_too_small")
    if mask_stats.dominant_fraction < min_dominant_fraction:
        reasons.append("dominant_component_too_weak")
    if mask_stats.significant_components > max_significant_components:
        reasons.append("mask_fragmented")
    if aspect_ratio < min_aspect_ratio:
        reasons.append("aspect_ratio_ambiguous")
    if fill_ratio < min_fill_ratio:
        reasons.append("partial_or_sparse_contour")

    score = confidence_score(reasons, aspect_ratio=aspect_ratio, fill_ratio=fill_ratio)
    confidence = Confidence(not reasons, score, tuple(reasons))
    pixel_obb = {
        "center": center.tolist(),
        "corners": corners.tolist(),
        "long_length_px": float(long_len),
        "short_length_px": float(short_len),
        "aspect_ratio": aspect_ratio,
        "fill_ratio": fill_ratio,
    }
    return PixelBoxEstimate(
        pixel_obb=pixel_obb,
        mask_stats=mask_stats,
        confidence=confidence,
        yaw_mod_180=float(yaw),
        long_axis_image=tuple(float(v) for v in long_axis),
        grasp_axis_image=tuple(float(v) for v in short_axis),
        failure_reasons=confidence.reasons,
    )


def estimate_metric_box(
    mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics | dict[str, Any],
    *,
    t5_T_camera: np.ndarray | None = None,
    table_plane: tuple[np.ndarray, np.ndarray] | None = None,
    min_boundary_depth_points: int = 30,
    min_aspect_ratio: float = 1.25,
    min_short_extent_m: float = 0.02,
    max_plane_depth_error_m: float = 0.03,
) -> MetricBoxEstimate:
    """Estimate a metric pose from boundary depth support.

    This function intentionally returns perception-only pose data. It never
    emits robot command targets.
    """

    mask_u8 = _require_mask(mask)
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2 or depth.shape != mask_u8.shape:
        raise ValueError(f"depth_m must be 2D and match mask shape, got {depth.shape} vs {mask_u8.shape}")
    intr = intrinsics if isinstance(intrinsics, CameraIntrinsics) else CameraIntrinsics.from_mapping(intrinsics)

    boundary = boundary_pixels(mask_u8)
    if boundary.shape[0] == 0:
        return _metric_failure("camera_table_plane", "no_boundary_pixels")

    points = backproject_pixels_with_depth(boundary, depth, intr)
    if points.shape[0] < min_boundary_depth_points:
        if table_plane is None:
            return _metric_failure("camera_table_plane", "insufficient_boundary_depth")
        if points.shape[0] > 0 and not points_match_plane(points, table_plane, max_error_m=max_plane_depth_error_m):
            return _metric_failure("camera_table_plane", "depth_plane_inconsistent")
        points = intersect_pixels_with_plane(boundary, intr, table_plane)
        if points.shape[0] < min_boundary_depth_points:
            return _metric_failure("camera_table_plane", "invalid_table_plane_fallback")

    if table_plane is None:
        plane_normal, plane_origin = fit_plane(points)
    else:
        normal, point = table_plane
        plane_normal = normalize(np.asarray(normal, dtype=np.float64).reshape(3))
        plane_origin = np.asarray(point, dtype=np.float64).reshape(3)
        if not points_match_plane(points, table_plane, max_error_m=max_plane_depth_error_m):
            return _metric_failure("camera_table_plane", "depth_plane_inconsistent")
    u_axis, v_axis = plane_basis(plane_normal)
    points_2d = np.column_stack(((points - plane_origin) @ u_axis, (points - plane_origin) @ v_axis)).astype(np.float32)
    rect = cv2.minAreaRect(points_2d)
    corners_2d = cv2.boxPoints(rect).astype(np.float64)
    center_2d, long_2d, short_2d, long_len, short_len = _axes_from_corners(corners_2d)
    aspect_ratio = float(long_len / max(short_len, 1e-9))
    reasons: list[str] = []
    if aspect_ratio < min_aspect_ratio:
        reasons.append("metric_aspect_ratio_ambiguous")
    if short_len < min_short_extent_m:
        reasons.append("boundary_support_biased")

    center_3d = plane_origin + center_2d[0] * u_axis + center_2d[1] * v_axis
    long_axis = normalize(long_2d[0] * u_axis + long_2d[1] * v_axis)
    short_axis = normalize(short_2d[0] * u_axis + short_2d[1] * v_axis)
    camera_T_box = construct_box_transform(center_3d, long_axis, short_axis, plane_normal)

    t5_T_box = None
    long_t5 = None
    short_t5 = None
    if t5_T_camera is not None:
        t5 = require_transform(t5_T_camera, "t5_T_camera")
        t5_T_box = t5 @ camera_T_box
        long_t5 = tuple(float(v) for v in normalize(t5_T_box[:3, 1]))
        short_t5 = tuple(float(v) for v in normalize(t5_T_box[:3, 0]))
        yaw = yaw_mod_180_from_axis(np.array([long_t5[0], long_t5[1]], dtype=np.float64))
        yaw_frame = "t5"
    else:
        yaw = yaw_mod_180_from_axis(long_2d)
        yaw_frame = "camera_table_plane"

    confidence = Confidence(not reasons, confidence_score(reasons, aspect_ratio=aspect_ratio, fill_ratio=1.0), tuple(reasons))
    return MetricBoxEstimate(
        camera_T_box=camera_T_box,
        t5_T_box=t5_T_box,
        long_axis_camera=tuple(float(v) for v in long_axis),
        grasp_axis_camera=tuple(float(v) for v in short_axis),
        long_axis_t5=long_t5,
        grasp_axis_t5=short_t5,
        yaw_mod_180=float(yaw),
        yaw_frame=yaw_frame,
        long_length_m=float(long_len),
        short_length_m=float(short_len),
        confidence=confidence,
        failure_reasons=confidence.reasons,
    )


def evaluate_still_frame_spread(
    centers_m: np.ndarray,
    yaws_deg: np.ndarray,
    *,
    max_center_spread_m: float = 0.01,
    max_yaw_spread_deg: float = 5.0,
) -> Confidence:
    """Return confidence for a short still-frame center/yaw sample."""

    centers = np.asarray(centers_m, dtype=np.float64)
    yaws = np.asarray(yaws_deg, dtype=np.float64).reshape(-1)
    reasons: list[str] = []
    if centers.ndim != 2 or centers.shape[1] != 3 or centers.shape[0] == 0:
        reasons.append("invalid_center_samples")
    if yaws.shape[0] != centers.shape[0]:
        reasons.append("yaw_center_sample_count_mismatch")
    if reasons:
        return Confidence(False, 0.0, tuple(reasons))

    if max_pairwise_distance(centers) > max_center_spread_m:
        reasons.append("center_spread_too_large")
    if yaw_spread_mod_180(yaws) > max_yaw_spread_deg:
        reasons.append("yaw_spread_too_large")
    return Confidence(not reasons, 0.0 if reasons else 1.0, tuple(reasons))


def largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(_require_mask(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def boundary_pixels(mask: np.ndarray) -> np.ndarray:
    contour = largest_contour(mask)
    if contour is None:
        return np.empty((0, 2), dtype=np.float64)
    hull = cv2.convexHull(contour)
    canvas = np.zeros_like(_require_mask(mask), dtype=np.uint8)
    cv2.drawContours(canvas, [hull], -1, 255, thickness=1)
    rows, cols = np.nonzero(canvas > 0)
    return np.column_stack((cols, rows)).astype(np.float64)


def backproject_pixels_with_depth(
    pixels_xy: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    pixels = np.asarray(pixels_xy, dtype=np.float64)
    cols = np.rint(pixels[:, 0]).astype(int)
    rows = np.rint(pixels[:, 1]).astype(int)
    h, w = depth_m.shape
    inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    rows = rows[inside]
    cols = cols[inside]
    z = np.asarray(depth_m, dtype=np.float64)[rows, cols]
    valid = np.isfinite(z) & (z > 0.0)
    rows = rows[valid]
    cols = cols[valid]
    z = z[valid]
    x = (cols.astype(np.float64) - intrinsics.cx) * z / intrinsics.fx
    y = (rows.astype(np.float64) - intrinsics.cy) * z / intrinsics.fy
    return np.column_stack((x, y, z)).astype(np.float64)


def points_match_plane(
    points: np.ndarray,
    table_plane: tuple[np.ndarray, np.ndarray],
    *,
    max_error_m: float,
) -> bool:
    if points.shape[0] == 0:
        return True
    normal, point = table_plane
    n = normalize(np.asarray(normal, dtype=np.float64).reshape(3))
    p0 = np.asarray(point, dtype=np.float64).reshape(3)
    distances = np.abs((np.asarray(points, dtype=np.float64) - p0) @ n)
    return bool(float(np.max(distances)) <= max_error_m)


def intersect_pixels_with_plane(
    pixels_xy: np.ndarray,
    intrinsics: CameraIntrinsics,
    table_plane: tuple[np.ndarray, np.ndarray],
    *,
    eps: float = 1e-9,
) -> np.ndarray:
    """Intersect camera rays with a plane described by normal and point."""

    normal, point = table_plane
    n = normalize(np.asarray(normal, dtype=np.float64).reshape(3))
    p0 = np.asarray(point, dtype=np.float64).reshape(3)
    pixels = np.asarray(pixels_xy, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError(f"pixels_xy must have shape Nx2, got {pixels.shape}")

    x = (pixels[:, 0] - intrinsics.cx) / intrinsics.fx
    y = (pixels[:, 1] - intrinsics.cy) / intrinsics.fy
    rays = np.column_stack((x, y, np.ones_like(x))).astype(np.float64)
    denom = rays @ n
    numer = float(np.dot(p0, n))
    valid = np.abs(denom) > eps
    t = np.zeros_like(denom)
    t[valid] = numer / denom[valid]
    valid &= t > eps
    return (rays[valid] * t[valid, None]).astype(np.float64)


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 3:
        raise ValueError("Need at least three 3D points to fit a plane.")
    origin = np.median(pts, axis=0)
    centered = pts - origin
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = normalize(vh[-1])
    if normal[2] < 0.0:
        normal = -normal
    return normal, origin


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = normalize(normal)
    candidate = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(candidate, n))) > 0.9:
        candidate = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u_axis = normalize(candidate - np.dot(candidate, n) * n)
    v_axis = normalize(np.cross(n, u_axis))
    return u_axis, v_axis


def construct_box_transform(
    center: np.ndarray,
    long_axis: np.ndarray,
    short_axis: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    y_axis = normalize(long_axis)
    x_axis = normalize(short_axis - np.dot(short_axis, y_axis) * y_axis)
    z_axis = normalize(np.cross(x_axis, y_axis))
    if float(np.dot(z_axis, normal)) < 0.0:
        x_axis = -x_axis
        z_axis = normalize(np.cross(x_axis, y_axis))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 0] = x_axis
    transform[:3, 1] = y_axis
    transform[:3, 2] = z_axis
    transform[:3, 3] = np.asarray(center, dtype=np.float64).reshape(3)
    return transform


def yaw_mod_180_from_axis(axis_xy: np.ndarray) -> float:
    axis = normalize(np.asarray(axis_xy, dtype=np.float64).reshape(-1)[:2])
    angle = math.degrees(math.atan2(float(axis[1]), float(axis[0])))
    return angle % 180.0


def confidence_score(reasons: list[str] | tuple[str, ...], *, aspect_ratio: float, fill_ratio: float) -> float:
    if reasons:
        return 0.0
    aspect_term = min(max((aspect_ratio - 1.0) / 1.0, 0.0), 1.0)
    fill_term = min(max(fill_ratio, 0.0), 1.0)
    return float(round(0.5 + 0.25 * aspect_term + 0.25 * fill_term, 3))


def max_pairwise_distance(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] <= 1:
        return 0.0
    deltas = pts[:, None, :] - pts[None, :, :]
    return float(np.max(np.linalg.norm(deltas, axis=2)))


def yaw_spread_mod_180(yaws_deg: np.ndarray) -> float:
    yaws = np.asarray(yaws_deg, dtype=np.float64).reshape(-1) % 180.0
    if yaws.shape[0] <= 1:
        return 0.0
    doubled = np.deg2rad(yaws * 2.0)
    mean_angle = math.atan2(float(np.mean(np.sin(doubled))), float(np.mean(np.cos(doubled)))) / 2.0
    center = math.degrees(mean_angle) % 180.0
    return float(max(angle_distance_mod_180(yaw, center) for yaw in yaws))


def angle_distance_mod_180(a: float, b: float) -> float:
    delta = abs((float(a) - float(b)) % 180.0)
    return min(delta, 180.0 - delta)


def safe_output_dict(data: dict[str, Any]) -> dict[str, Any]:
    forbidden = sorted(_find_forbidden_fields(data))
    if forbidden:
        raise ValueError(f"Unsafe robot command fields are forbidden: {sorted(forbidden)}")
    return data


def _find_forbidden_fields(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN_OUTPUT_FIELDS:
                found.add(str(key))
            found.update(_find_forbidden_fields(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.update(_find_forbidden_fields(item))
    return found


def require_transform(transform: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def normalize(vector: np.ndarray, *, eps: float = 1e-9) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm < eps:
        raise ValueError("Cannot normalize a near-zero vector.")
    return arr / norm


def _axes_from_corners(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    center = np.mean(pts, axis=0)
    edges = [(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
    lengths = np.array([np.linalg.norm(edge) for edge in edges], dtype=np.float64)
    long_index = int(np.argmax(lengths))
    long_axis = normalize(edges[long_index])
    short_candidates = [
        edge
        for edge, length in zip(edges, lengths)
        if 1e-9 < length < lengths[long_index] * 0.95
    ]
    if short_candidates:
        short_axis = normalize(short_candidates[0])
    else:
        short_axis = np.array([-long_axis[1], long_axis[0]], dtype=np.float64)
    short_axis = short_axis - np.dot(short_axis, long_axis) * long_axis
    if np.linalg.norm(short_axis) < 1e-9:
        short_axis = np.array([-long_axis[1], long_axis[0]], dtype=np.float64)
    short_axis = normalize(short_axis)
    long_len = float(lengths[long_index])
    short_len = float(np.min(lengths))
    return center, long_axis, short_axis, max(long_len, short_len), min(long_len, short_len)


def _require_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"mask must be 2D, got {arr.shape}")
    return np.where(arr > 0, 255, 0).astype(np.uint8)


def _metric_failure(yaw_frame: str, reason: str) -> MetricBoxEstimate:
    confidence = Confidence(False, 0.0, (reason,))
    return MetricBoxEstimate(
        camera_T_box=None,
        t5_T_box=None,
        long_axis_camera=None,
        grasp_axis_camera=None,
        long_axis_t5=None,
        grasp_axis_t5=None,
        yaw_mod_180=None,
        yaw_frame=yaw_frame,
        long_length_m=None,
        short_length_m=None,
        confidence=confidence,
        failure_reasons=confidence.reasons,
    )
