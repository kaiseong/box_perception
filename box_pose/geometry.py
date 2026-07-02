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
    short_axis_image: tuple[float, float]
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
                "short_axis_image": list(self.short_axis_image),
                "failure_reasons": list(self.failure_reasons),
            }
        )


@dataclass(frozen=True)
class MetricBoxEstimate:
    camera_T_box: np.ndarray | None
    t5_T_box: np.ndarray | None
    long_axis_camera: tuple[float, float, float] | None
    short_axis_camera: tuple[float, float, float] | None
    long_axis_t5: tuple[float, float, float] | None
    short_axis_t5: tuple[float, float, float] | None
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
                "short_axis_camera": None if self.short_axis_camera is None else list(self.short_axis_camera),
                "long_axis_t5": None if self.long_axis_t5 is None else list(self.long_axis_t5),
                "short_axis_t5": None if self.short_axis_t5 is None else list(self.short_axis_t5),
                "yaw_mod_180": self.yaw_mod_180,
                "yaw_frame": self.yaw_frame,
                "long_length_m": self.long_length_m,
                "short_length_m": self.short_length_m,
                "confidence": self.confidence.to_dict(),
                "failure_reasons": list(self.failure_reasons),
            }
        )


@dataclass(frozen=True)
class KnownSizeBoxEstimate:
    center_image: tuple[float, float] | None
    center_top_camera_m: tuple[float, float, float] | None
    yaw_mod_180: float | None
    long_axis_image: tuple[float, float] | None
    short_axis_image: tuple[float, float] | None
    model_corners: tuple[tuple[float, float], ...] | None
    model_long_length_px: float | None
    model_short_length_px: float | None
    depth_reference_m: float | None
    box_long_m: float
    box_short_m: float
    box_height_m: float
    projection_plane: str
    support: dict[str, Any]
    confidence: Confidence
    failure_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return safe_output_dict(
            {
                "center_image": None if self.center_image is None else list(self.center_image),
                "center_top_camera_m": None
                if self.center_top_camera_m is None
                else list(self.center_top_camera_m),
                "yaw_mod_180": self.yaw_mod_180,
                "yaw_frame": "image",
                "long_axis_image": None if self.long_axis_image is None else list(self.long_axis_image),
                "short_axis_image": None if self.short_axis_image is None else list(self.short_axis_image),
                "model_corners": None
                if self.model_corners is None
                else [list(corner) for corner in self.model_corners],
                "model_long_length_px": self.model_long_length_px,
                "model_short_length_px": self.model_short_length_px,
                "depth_reference_m": self.depth_reference_m,
                "box_size_m": {
                    "long": self.box_long_m,
                    "short": self.box_short_m,
                    "height": self.box_height_m,
                },
                "projection_plane": self.projection_plane,
                "support": self.support,
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
        short_axis_image=tuple(float(v) for v in short_axis),
        failure_reasons=confidence.reasons,
    )


def estimate_known_size_box(
    mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics | dict[str, Any],
    *,
    box_long_m: float = 0.505,
    box_short_m: float = 0.335,
    box_height_m: float = 0.195,
    projection_plane: str = "box_top_plane",
    min_evidence_pixels: int = 500,
    min_depth_pixels: int = 200,
    depth_band_below_m: float = 0.25,
    depth_band_above_m: float = 0.18,
    yaw_cluster_deg: float = 18.0,
    short_pair_min_ratio: float = 0.45,
    short_pair_max_ratio: float = 1.45,
) -> KnownSizeBoxEstimate:
    """Fit a fixed-size box footprint to partial yellow/rim evidence.

    The D405 crop case often invalidates a plain minAreaRect: the visible
    yellow mask can be only one wall, one rim, or a clipped footprint. This
    estimator uses Hough line orientation for yaw, then combines observed long
    rim lines with the known box size and depth-derived pixel scale to infer a
    best-effort center in the analysis image frame.
    """

    mask_u8 = _require_mask(mask)
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2 or depth.shape != mask_u8.shape:
        raise ValueError(f"depth_m must be 2D and match mask shape, got {depth.shape} vs {mask_u8.shape}")
    intr = intrinsics if isinstance(intrinsics, CameraIntrinsics) else CameraIntrinsics.from_mapping(intrinsics)

    rows, cols = np.nonzero(mask_u8 > 0)
    if cols.size < min_evidence_pixels:
        return _known_size_failure(
            "insufficient_yellow_evidence",
            box_long_m=box_long_m,
            box_short_m=box_short_m,
            box_height_m=box_height_m,
            projection_plane=projection_plane,
            support={"evidence_pixels": int(cols.size)},
        )

    reasons: list[str] = []
    support_notes: dict[str, Any] = {"raw_evidence_pixels": int(cols.size)}
    raw_depth_values = depth[rows, cols]
    raw_valid_depth = raw_depth_values[np.isfinite(raw_depth_values) & (raw_depth_values > 0.0)]
    if raw_valid_depth.size >= min_depth_pixels:
        preliminary_depth = float(np.median(raw_valid_depth))
        lower_depth = max(0.0, preliminary_depth - float(depth_band_below_m))
        upper_depth = preliminary_depth + float(depth_band_above_m)
        depth_band = np.isfinite(depth) & (depth > 0.0) & (depth >= lower_depth) & (depth <= upper_depth)
        filtered_mask = np.where((mask_u8 > 0) & depth_band, 255, 0).astype(np.uint8)
        filtered_count = int(np.count_nonzero(filtered_mask))
        support_notes["depth_filter"] = {
            "preliminary_median_m": preliminary_depth,
            "lower_m": lower_depth,
            "upper_m": upper_depth,
            "filtered_evidence_pixels": filtered_count,
        }
        if filtered_count >= min_evidence_pixels:
            mask_u8 = filtered_mask
            rows, cols = np.nonzero(mask_u8 > 0)
        else:
            reasons.append("depth_filter_too_sparse")

    points = np.column_stack((cols, rows)).astype(np.float64)
    depth_values = depth[rows, cols]
    valid_depth = depth_values[np.isfinite(depth_values) & (depth_values > 0.0)]
    depth_reference = None
    expected_long_px = None
    expected_short_px = None
    if valid_depth.size < min_depth_pixels:
        reasons.append("insufficient_depth_support")
    else:
        q10, q90 = np.quantile(valid_depth, [0.10, 0.90])
        trimmed = valid_depth[(valid_depth >= q10) & (valid_depth <= q90)]
        depth_reference = float(np.median(trimmed if trimmed.size else valid_depth))
        focal = float((intr.fx + intr.fy) * 0.5)
        expected_long_px = focal * float(box_long_m) / depth_reference
        expected_short_px = focal * float(box_short_m) / depth_reference

    edge = yellow_edge_mask(mask_u8)
    lines = hough_line_segments(edge)
    if not lines:
        pixel_fallback = estimate_pixel_box(mask_u8, MaskStats(mask_u8.size, int(cols.size), int(cols.size), 1.0, 1))
        if not np.isfinite(pixel_fallback.yaw_mod_180):
            return _known_size_failure(
                "insufficient_line_support",
                box_long_m=box_long_m,
                box_short_m=box_short_m,
                box_height_m=box_height_m,
                projection_plane=projection_plane,
                support={"evidence_pixels": int(cols.size), "line_count": 0},
            )
        yaw = float(pixel_fallback.yaw_mod_180)
        yaw_lines: list[dict[str, Any]] = []
        reasons.append("pixel_yaw_fallback")
    else:
        yaw, yaw_lines, line_support_fraction = dominant_line_yaw(lines, cluster_deg=yaw_cluster_deg)
        if line_support_fraction < 0.35:
            reasons.append("weak_yaw_line_consensus")

    u_axis = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw))], dtype=np.float64)
    if u_axis[0] < 0.0:
        u_axis = -u_axis
    v_axis = np.array([-u_axis[1], u_axis[0]], dtype=np.float64)

    image_h, image_w = mask_u8.shape
    long_len_px = choose_model_long_length(points, u_axis, expected_long_px)
    short_pair = choose_short_axis_pair(
        yaw_lines,
        v_axis,
        expected_short_px,
        min_ratio=short_pair_min_ratio,
        max_ratio=short_pair_max_ratio,
    )
    if short_pair is None:
        reasons.append("missing_parallel_short_extent")
        short_len_px = choose_fallback_short_length(points, v_axis, expected_short_px)
        center_v = choose_fallback_short_center(points, v_axis, short_len_px)
    else:
        center_v, short_len_px, pair_support = short_pair

    center_u = choose_long_center(points, u_axis, long_len_px)
    center = center_u * u_axis + center_v * v_axis
    corners = rectangle_corners(center, u_axis, v_axis, long_len_px, short_len_px)

    perimeter_support = score_rectangle_perimeter(edge, mask_u8, corners)
    if perimeter_support["visible_samples"] < 20:
        reasons.append("too_little_visible_model_perimeter")
    if perimeter_support["edge_support_fraction"] < 0.08:
        reasons.append("low_perimeter_edge_support")

    if not (0.0 <= center[0] < image_w and 0.0 <= center[1] < image_h):
        reasons.append("center_outside_image")

    center_camera = None
    if depth_reference is not None and np.all(np.isfinite(center)):
        center_camera = backproject_center(center, depth_reference, intr)

    line_support_fraction = 0.0 if not lines else float(sum(line["weight"] for line in yaw_lines) / sum(line["weight"] for line in lines))
    support: dict[str, Any] = {
        "evidence_pixels": int(cols.size),
        "depth_pixels": int(valid_depth.size),
        "line_count": len(lines),
        "yaw_line_count": len(yaw_lines),
        "yaw_line_support_fraction": line_support_fraction,
        "perimeter": perimeter_support,
        "expected_long_length_px": expected_long_px,
        "expected_short_length_px": expected_short_px,
    }
    support.update(support_notes)
    if short_pair is not None:
        support["short_axis_pair_support"] = pair_support

    score = known_size_confidence_score(reasons, line_support_fraction, perimeter_support)
    confidence = Confidence(not reasons, score, tuple(reasons))
    return KnownSizeBoxEstimate(
        center_image=tuple(float(v) for v in center),
        center_top_camera_m=None if center_camera is None else tuple(float(v) for v in center_camera),
        yaw_mod_180=float(yaw % 180.0),
        long_axis_image=tuple(float(v) for v in u_axis),
        short_axis_image=tuple(float(v) for v in v_axis),
        model_corners=tuple(tuple(float(v) for v in corner) for corner in corners),
        model_long_length_px=float(long_len_px),
        model_short_length_px=float(short_len_px),
        depth_reference_m=depth_reference,
        box_long_m=float(box_long_m),
        box_short_m=float(box_short_m),
        box_height_m=float(box_height_m),
        projection_plane=projection_plane,
        support=support,
        confidence=confidence,
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
        short_axis_camera=tuple(float(v) for v in short_axis),
        long_axis_t5=long_t5,
        short_axis_t5=short_t5,
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


def yellow_edge_mask(mask: np.ndarray) -> np.ndarray:
    mask_u8 = _require_mask(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edge = cv2.morphologyEx(mask_u8, cv2.MORPH_GRADIENT, kernel)
    return np.where(edge > 0, 255, 0).astype(np.uint8)


def hough_line_segments(edge: np.ndarray) -> list[dict[str, Any]]:
    edge_u8 = _require_mask(edge)
    h, w = edge_u8.shape
    min_line_length = max(35, int(round(min(h, w) * 0.06)))
    threshold = max(18, int(round(min_line_length * 0.55)))
    max_line_gap = max(12, int(round(min_line_length * 0.45)))
    raw_lines = cv2.HoughLinesP(
        edge_u8,
        1,
        np.pi / 180.0,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if raw_lines is None:
        return []

    lines: list[dict[str, Any]] = []
    for x1, y1, x2, y2 in raw_lines[:, 0, :]:
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = float(math.hypot(dx, dy))
        if length < min_line_length:
            continue
        angle = math.degrees(math.atan2(dy, dx)) % 180.0
        midpoint = np.array([(float(x1) + float(x2)) * 0.5, (float(y1) + float(y2)) * 0.5], dtype=np.float64)
        lines.append(
            {
                "p1": (float(x1), float(y1)),
                "p2": (float(x2), float(y2)),
                "midpoint": midpoint,
                "angle_deg": float(angle),
                "length_px": length,
                "weight": float(length * length),
            }
        )
    return lines


def dominant_line_yaw(lines: list[dict[str, Any]], *, cluster_deg: float) -> tuple[float, list[dict[str, Any]], float]:
    total_weight = float(sum(line["weight"] for line in lines))
    best_group: list[dict[str, Any]] = []
    best_weight = -1.0
    for candidate in lines:
        group = [
            line
            for line in lines
            if angle_distance_mod_180(float(candidate["angle_deg"]), float(line["angle_deg"])) <= cluster_deg
        ]
        weight = float(sum(line["weight"] for line in group))
        if weight > best_weight:
            best_weight = weight
            best_group = group

    yaw = weighted_circular_mean_mod_180(
        np.array([line["angle_deg"] for line in best_group], dtype=np.float64),
        np.array([line["weight"] for line in best_group], dtype=np.float64),
    )
    return float(yaw), best_group, 0.0 if total_weight <= 0.0 else float(best_weight / total_weight)


def weighted_circular_mean_mod_180(values_deg: np.ndarray, weights: np.ndarray) -> float:
    if values_deg.size == 0:
        return math.nan
    doubled = np.deg2rad((values_deg % 180.0) * 2.0)
    w = np.asarray(weights, dtype=np.float64)
    if not np.any(w > 0.0):
        w = np.ones_like(values_deg, dtype=np.float64)
    sin_mean = float(np.average(np.sin(doubled), weights=w))
    cos_mean = float(np.average(np.cos(doubled), weights=w))
    return float((math.degrees(math.atan2(sin_mean, cos_mean)) * 0.5) % 180.0)


def choose_model_long_length(points_xy: np.ndarray, u_axis: np.ndarray, expected_long_px: float | None) -> float:
    projection = np.asarray(points_xy, dtype=np.float64) @ u_axis
    q01, q99 = np.quantile(projection, [0.01, 0.99])
    observed_span = float(max(q99 - q01, 1.0))
    if expected_long_px is None or not np.isfinite(expected_long_px) or expected_long_px <= 0.0:
        return max(observed_span, 20.0)
    return max(float(expected_long_px), 20.0)


def choose_long_center(points_xy: np.ndarray, u_axis: np.ndarray, long_len_px: float) -> float:
    projection = np.asarray(points_xy, dtype=np.float64) @ u_axis
    q01, q99 = np.quantile(projection, [0.01, 0.99])
    low = float(q99 - long_len_px * 0.5)
    high = float(q01 + long_len_px * 0.5)
    if low <= high:
        return float((low + high) * 0.5)
    return float((q01 + q99) * 0.5)


def choose_short_axis_pair(
    yaw_lines: list[dict[str, Any]],
    v_axis: np.ndarray,
    expected_short_px: float | None,
    *,
    min_ratio: float = 0.45,
    max_ratio: float = 1.45,
) -> tuple[float, float, dict[str, Any]] | None:
    if len(yaw_lines) < 2:
        return None

    raw_positions = []
    for line in yaw_lines:
        position = float(np.asarray(line["midpoint"], dtype=np.float64) @ v_axis)
        raw_positions.append((position, float(line["weight"]), float(line["length_px"])))
    raw_positions.sort(key=lambda item: item[0])

    merge_threshold = 18.0
    if expected_short_px is not None and np.isfinite(expected_short_px):
        merge_threshold = max(18.0, float(expected_short_px) * 0.06)
    clusters: list[dict[str, Any]] = []
    for position, weight, length in raw_positions:
        if not clusters or abs(position - float(clusters[-1]["position"])) > merge_threshold:
            clusters.append({"position": position, "weight": weight, "count": 1, "max_length_px": length})
            continue
        cluster = clusters[-1]
        total = float(cluster["weight"]) + weight
        cluster["position"] = (float(cluster["position"]) * float(cluster["weight"]) + position * weight) / total
        cluster["weight"] = total
        cluster["count"] = int(cluster["count"]) + 1
        cluster["max_length_px"] = max(float(cluster["max_length_px"]), length)

    if len(clusters) < 2:
        return None

    best: tuple[float, float, dict[str, Any]] | None = None
    best_score = -1.0
    for i, first in enumerate(clusters):
        for second in clusters[i + 1 :]:
            separation = abs(float(second["position"]) - float(first["position"]))
            if expected_short_px is not None and np.isfinite(expected_short_px):
                min_sep = max(35.0, float(expected_short_px) * min_ratio)
                max_sep = max(min_sep, float(expected_short_px) * max_ratio)
                if separation < min_sep or separation > max_sep:
                    continue
                closeness = 1.0 - min(abs(separation - float(expected_short_px)) / float(expected_short_px), 1.0)
            else:
                if separation < 35.0:
                    continue
                closeness = 0.5
            score = (float(first["weight"]) + float(second["weight"])) * (0.65 + 0.35 * closeness)
            if score > best_score:
                center_v = (float(first["position"]) + float(second["position"])) * 0.5
                pair_support = {
                    "cluster_count": len(clusters),
                    "first": {
                        "position": float(first["position"]),
                        "line_count": int(first["count"]),
                        "max_length_px": float(first["max_length_px"]),
                    },
                    "second": {
                        "position": float(second["position"]),
                        "line_count": int(second["count"]),
                        "max_length_px": float(second["max_length_px"]),
                    },
                    "separation_px": float(separation),
                    "score": float(score),
                }
                best = (float(center_v), float(separation), pair_support)
                best_score = score
    return best


def choose_fallback_short_length(points_xy: np.ndarray, v_axis: np.ndarray, expected_short_px: float | None) -> float:
    projection = np.asarray(points_xy, dtype=np.float64) @ v_axis
    q10, q80 = np.quantile(projection, [0.10, 0.80])
    observed = float(max(q80 - q10, 20.0))
    if expected_short_px is None or not np.isfinite(expected_short_px) or expected_short_px <= 0.0:
        return observed
    return float(max(20.0, min(observed, expected_short_px)))


def choose_fallback_short_center(points_xy: np.ndarray, v_axis: np.ndarray, short_len_px: float) -> float:
    projection = np.asarray(points_xy, dtype=np.float64) @ v_axis
    q05, q75 = np.quantile(projection, [0.05, 0.75])
    if q75 - q05 > short_len_px * 1.15:
        return float(q05 + short_len_px * 0.5)
    return float((q05 + q75) * 0.5)


def rectangle_corners(
    center_xy: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    long_len_px: float,
    short_len_px: float,
) -> np.ndarray:
    center = np.asarray(center_xy, dtype=np.float64).reshape(2)
    half_long = float(long_len_px) * 0.5
    half_short = float(short_len_px) * 0.5
    return np.array(
        [
            center - half_long * u_axis - half_short * v_axis,
            center + half_long * u_axis - half_short * v_axis,
            center + half_long * u_axis + half_short * v_axis,
            center - half_long * u_axis + half_short * v_axis,
        ],
        dtype=np.float64,
    )


def score_rectangle_perimeter(edge: np.ndarray, mask: np.ndarray, corners: np.ndarray) -> dict[str, Any]:
    edge_u8 = _require_mask(edge)
    mask_u8 = _require_mask(mask)
    h, w = edge_u8.shape
    edge_binary = np.where(edge_u8 > 0, 255, 0).astype(np.uint8)
    dist = cv2.distanceTransform(255 - edge_binary, cv2.DIST_L2, 3)
    dilated_mask = cv2.dilate(mask_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    samples = sample_polygon_perimeter(np.asarray(corners, dtype=np.float64), samples_per_edge=90)
    cols = np.rint(samples[:, 0]).astype(int)
    rows = np.rint(samples[:, 1]).astype(int)
    inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    if not np.any(inside):
        return {
            "visible_samples": 0,
            "edge_support_fraction": 0.0,
            "mask_support_fraction": 0.0,
            "mean_edge_distance_px": None,
        }

    rows = rows[inside]
    cols = cols[inside]
    distances = dist[rows, cols].astype(np.float64)
    edge_support = distances <= 10.0
    mask_support = dilated_mask[rows, cols] > 0
    return {
        "visible_samples": int(rows.size),
        "edge_support_fraction": float(np.mean(edge_support)),
        "mask_support_fraction": float(np.mean(mask_support)),
        "mean_edge_distance_px": float(np.mean(distances)),
    }


def sample_polygon_perimeter(corners: np.ndarray, *, samples_per_edge: int) -> np.ndarray:
    pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    samples = []
    count = max(int(samples_per_edge), 2)
    for index in range(4):
        start = pts[index]
        end = pts[(index + 1) % 4]
        alpha = np.linspace(0.0, 1.0, count, endpoint=False, dtype=np.float64)
        samples.append(start[None, :] * (1.0 - alpha[:, None]) + end[None, :] * alpha[:, None])
    return np.vstack(samples)


def backproject_center(center_xy: np.ndarray, depth_m: float, intrinsics: CameraIntrinsics) -> np.ndarray:
    center = np.asarray(center_xy, dtype=np.float64).reshape(2)
    z = float(depth_m)
    x = (float(center[0]) - intrinsics.cx) * z / intrinsics.fx
    y = (float(center[1]) - intrinsics.cy) * z / intrinsics.fy
    return np.array([x, y, z], dtype=np.float64)


def known_size_confidence_score(
    reasons: list[str],
    line_support_fraction: float,
    perimeter_support: dict[str, Any],
) -> float:
    edge_support = float(perimeter_support.get("edge_support_fraction") or 0.0)
    mask_support = float(perimeter_support.get("mask_support_fraction") or 0.0)
    raw = 0.25 + 0.35 * min(max(line_support_fraction, 0.0), 1.0)
    raw += 0.25 * min(max(edge_support / 0.25, 0.0), 1.0)
    raw += 0.15 * min(max(mask_support / 0.55, 0.0), 1.0)
    if reasons:
        raw *= 0.45
    return float(round(min(max(raw, 0.0), 1.0), 3))


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
        short_axis_camera=None,
        long_axis_t5=None,
        short_axis_t5=None,
        yaw_mod_180=None,
        yaw_frame=yaw_frame,
        long_length_m=None,
        short_length_m=None,
        confidence=confidence,
        failure_reasons=confidence.reasons,
    )


def _known_size_failure(
    reason: str,
    *,
    box_long_m: float,
    box_short_m: float,
    box_height_m: float,
    projection_plane: str,
    support: dict[str, Any] | None = None,
) -> KnownSizeBoxEstimate:
    confidence = Confidence(False, 0.0, (reason,))
    return KnownSizeBoxEstimate(
        center_image=None,
        center_top_camera_m=None,
        yaw_mod_180=None,
        long_axis_image=None,
        short_axis_image=None,
        model_corners=None,
        model_long_length_px=None,
        model_short_length_px=None,
        depth_reference_m=None,
        box_long_m=float(box_long_m),
        box_short_m=float(box_short_m),
        box_height_m=float(box_height_m),
        projection_plane=projection_plane,
        support={} if support is None else support,
        confidence=confidence,
        failure_reasons=confidence.reasons,
    )
