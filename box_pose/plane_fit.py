"""Rim-plane metric-space fitting for the fixed-size box.

The image-space known-size estimator assumes one pixel scale for the whole
box, which breaks under the tilted D405 mount: the near rim is ~0.33 m away
while the far rim is ~0.5 m, so the fitted rectangle drifts toward the near
edge. This estimator removes the perspective problem instead of compensating
for it:

1. gate mask pixels to the box top/rim plane using depth
2. intersect the camera rays of the gated pixels with the plane, giving
   metric top-down 2D coordinates where the rim outline is a true 505x335
   rectangle regardless of cropping
3. fit the fixed-size rectangle with 3 DOF (center u/v + yaw), trusting only
   box edges that are not clipped by the image border

The rim plane itself is a fixed property of the setup: the table height, the
box height, and the camera mount are all constant, and the robot base moves
on a flat floor, so the rim plane in camera coordinates does not change from
frame to frame. `discover_rim_plane` finds it with RANSAC on frames where the
box is well visible (near frames are dominated by the front/back wall planes
and are unreliable for discovery), and `estimate_plane_box` then reuses the
calibrated plane, only refining it slightly per frame to absorb base tilt.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from .geometry import (
    CameraIntrinsics,
    Confidence,
    KnownSizeBoxEstimate,
    _known_size_failure,
    _require_mask,
    normalize,
    plane_basis,
)


def estimate_plane_box(
    mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics | dict[str, Any],
    *,
    box_long_m: float = 0.505,
    box_short_m: float = 0.335,
    box_height_m: float = 0.195,
    rim_plane: tuple[Any, Any] | None = None,
    refine_plane: bool = True,
    refine_max_angle_deg: float = 3.0,
    refine_max_offset_m: float = 0.010,
    projection_plane: str = "box_rim_plane",
    min_evidence_pixels: int = 500,
    min_rim_points: int = 250,
    ransac_iterations: int = 250,
    ransac_tolerance_m: float = 0.006,
    rim_band_below_m: float = 0.006,
    rim_band_above_m: float = 0.012,
    border_margin_px: int = 8,
    edge_band_m: float = 0.015,
    max_fit_points: int = 40000,
    seed: int = 12345,
) -> KnownSizeBoxEstimate:
    """Fit the fixed-size box footprint on the rim plane in metric space.

    When `rim_plane=(normal, point)` is provided (the calibrated setup
    constant), no per-frame plane search happens; depth only gates and
    verifies. Without it the rim plane is discovered per frame with RANSAC,
    which is only reliable while the box top dominates the mask.

    The asymmetric rim band matters: pixels on the front wall just below the
    rim would ray-project slightly outside the true front edge and bias the
    center toward the camera, so the band keeps less below the plane than
    above it.
    """

    mask_u8 = _require_mask(mask)
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2 or depth.shape != mask_u8.shape:
        raise ValueError(f"depth_m must be 2D and match mask shape, got {depth.shape} vs {mask_u8.shape}")
    intr = intrinsics if isinstance(intrinsics, CameraIntrinsics) else CameraIntrinsics.from_mapping(intrinsics)

    rows, cols = np.nonzero((mask_u8 > 0) & np.isfinite(depth) & (depth > 0.0))
    if rows.size < min_evidence_pixels:
        return _failure(
            "insufficient_masked_depth",
            box_long_m,
            box_short_m,
            box_height_m,
            projection_plane,
            {"masked_depth_pixels": int(rows.size)},
        )

    z_all = depth[rows, cols]
    x_all = (cols.astype(np.float64) - intr.cx) * z_all / intr.fx
    y_all = (rows.astype(np.float64) - intr.cy) * z_all / intr.fy
    points_all = np.column_stack((x_all, y_all, z_all))
    sample = points_all[_even_indices(rows.size, 4000)]

    reasons: list[str] = []
    if rim_plane is not None:
        normal = normalize(np.asarray(rim_plane[0], dtype=np.float64).reshape(3))
        origin = np.asarray(rim_plane[1], dtype=np.float64).reshape(3)
        if float(normal @ origin) > 0.0:
            normal = -normal
        normal, origin, plane_support = _verify_fixed_plane(
            sample,
            normal,
            origin,
            band_m=rim_band_above_m,
            refine=refine_plane,
            max_angle_deg=refine_max_angle_deg,
            max_offset_m=refine_max_offset_m,
        )
        if plane_support["gated_fraction"] < 0.03:
            reasons.append("rim_plane_support_too_low")
    else:
        rng = np.random.default_rng(seed)
        discovered = _discover_plane(
            sample,
            box_short_m=box_short_m,
            iterations=ransac_iterations,
            tolerance_m=ransac_tolerance_m,
            rng=rng,
        )
        if discovered is None:
            return _failure(
                "rim_plane_not_found",
                box_long_m,
                box_short_m,
                box_height_m,
                projection_plane,
                {"masked_depth_pixels": int(rows.size)},
            )
        normal, origin, plane_support = discovered
        if plane_support["rim_dim_score"] < 0.2:
            reasons.append("rim_plane_dim_mismatch")

    signed = (points_all - origin) @ normal
    gated = (signed >= -rim_band_below_m) & (signed <= rim_band_above_m)
    if int(np.count_nonzero(gated)) < min_rim_points:
        return _failure(
            "rim_points_too_sparse",
            box_long_m,
            box_short_m,
            box_height_m,
            projection_plane,
            {"masked_depth_pixels": int(rows.size), "rim_points": int(np.count_nonzero(gated))},
        )

    rim_rows = rows[gated]
    rim_cols = cols[gated]

    on_plane, valid = _intersect_pixels_with_plane(rim_cols, rim_rows, intr, normal, origin)
    rim_rows = rim_rows[valid]
    rim_cols = rim_cols[valid]
    if rim_rows.size < min_rim_points:
        return _failure(
            "rim_projection_failed",
            box_long_m,
            box_short_m,
            box_height_m,
            projection_plane,
            {"rim_points": int(rim_rows.size)},
        )

    e1, e2 = plane_basis(normal)
    coords = np.column_stack(((on_plane - origin) @ e1, (on_plane - origin) @ e2))

    # Skin passes the HSV gate and arms hanging at rim height also pass the
    # depth gate, but they sit well outside the box footprint. Components
    # whose union would exceed the box diagonal cannot all be rim.
    component_keep, component_info = _filter_footprint_components(
        rim_rows,
        rim_cols,
        coords,
        image_shape=mask_u8.shape,
        max_diameter_m=math.hypot(box_long_m, box_short_m) + 0.04,
    )
    rim_rows = rim_rows[component_keep]
    rim_cols = rim_cols[component_keep]
    coords = coords[component_keep]
    if rim_rows.size < min_rim_points:
        return _failure(
            "rim_points_too_sparse",
            box_long_m,
            box_short_m,
            box_height_m,
            projection_plane,
            {"rim_points": int(rim_rows.size), "components": component_info},
        )

    keep = _even_indices(rim_rows.size, max_fit_points)
    rim_rows = rim_rows[keep]
    rim_cols = rim_cols[keep]
    coords = coords[keep]

    yaw_plane, yaw_support = _estimate_plane_yaw(
        coords,
        rim_rows,
        rim_cols,
        image_shape=mask_u8.shape,
        border_margin_px=border_margin_px,
    )
    if not yaw_support["hull_edge_count"]:
        reasons.append("yaw_from_extent_fallback")

    u_dir = np.array([math.cos(yaw_plane), math.sin(yaw_plane)], dtype=np.float64)
    v_dir = np.array([-u_dir[1], u_dir[0]], dtype=np.float64)
    near_border = _near_border(rim_rows, rim_cols, mask_u8.shape, border_margin_px)

    edge_u = _axis_edges(coords @ u_dir, coords @ v_dir, near_border, edge_band_m)
    edge_v = _axis_edges(coords @ v_dir, coords @ u_dir, near_border, edge_band_m)
    long_edges, short_edges, long_dir, short_dir, swap_penalty = _assign_axes(
        edge_u, edge_v, u_dir, v_dir, box_long_m, box_short_m
    )

    center_long, long_info = _axis_center(long_edges, box_long_m, "long_axis", reasons)
    center_short, short_info = _axis_center(short_edges, box_short_m, "short_axis", reasons)
    if swap_penalty > 0.05:
        reasons.append("axis_assignment_ambiguous")

    center_2d = center_long * long_dir + center_short * short_dir
    center_3d = origin + center_2d[0] * e1 + center_2d[1] * e2
    long_3d = normalize(long_dir[0] * e1 + long_dir[1] * e2)
    short_3d = normalize(short_dir[0] * e1 + short_dir[1] * e2)

    corners_3d = [
        center_3d + sx * 0.5 * box_long_m * long_3d + sy * 0.5 * box_short_m * short_3d
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]
    if center_3d[2] <= 0.0 or any(corner[2] <= 0.0 for corner in corners_3d):
        return _failure(
            "center_behind_camera",
            box_long_m,
            box_short_m,
            box_height_m,
            projection_plane,
            {"rim_points": int(rim_rows.size)},
        )

    center_image = _project_point(intr, center_3d)
    corners_image = [_project_point(intr, corner) for corner in corners_3d]
    long_axis_image = _project_axis(intr, center_3d, long_3d)
    short_axis_image = _project_axis(intr, center_3d, short_3d)
    yaw_image = math.degrees(math.atan2(long_axis_image[1], long_axis_image[0])) % 180.0

    image_h, image_w = mask_u8.shape
    if not (0.0 <= center_image[0] < image_w and 0.0 <= center_image[1] < image_h):
        reasons.append("center_outside_image")

    long_edge_px = 0.5 * (
        float(np.linalg.norm(corners_image[1] - corners_image[0]))
        + float(np.linalg.norm(corners_image[2] - corners_image[3]))
    )
    short_edge_px = 0.5 * (
        float(np.linalg.norm(corners_image[2] - corners_image[1]))
        + float(np.linalg.norm(corners_image[3] - corners_image[0]))
    )

    trusted_edges = int(long_info["trusted_edges"]) + int(short_info["trusted_edges"])
    plane_support["normal_camera"] = [float(v) for v in normal]
    plane_support["origin_camera"] = [float(v) for v in origin]
    support: dict[str, Any] = {
        "method": "rim_plane",
        "masked_depth_pixels": int(rows.size),
        "rim_points": int(rim_rows.size),
        "plane": plane_support,
        "yaw": yaw_support,
        "long_axis": long_info,
        "short_axis": short_info,
        "components": component_info,
        "trusted_edges": trusted_edges,
        "long_axis_camera": [float(v) for v in long_3d],
        "short_axis_camera": [float(v) for v in short_3d],
    }

    score = _confidence_score(
        reasons,
        plane_quality=plane_support["quality"],
        trusted_edges=trusted_edges,
        span_errors_m=[info["span_error_m"] for info in (long_info, short_info) if info["span_error_m"] is not None],
    )
    confidence = Confidence(not reasons, score, tuple(reasons))
    return KnownSizeBoxEstimate(
        center_image=(float(center_image[0]), float(center_image[1])),
        center_top_camera_m=tuple(float(v) for v in center_3d),
        yaw_mod_180=float(yaw_image),
        long_axis_image=(float(long_axis_image[0]), float(long_axis_image[1])),
        short_axis_image=(float(short_axis_image[0]), float(short_axis_image[1])),
        model_corners=tuple((float(c[0]), float(c[1])) for c in corners_image),
        model_long_length_px=long_edge_px,
        model_short_length_px=short_edge_px,
        depth_reference_m=float(center_3d[2]),
        box_long_m=float(box_long_m),
        box_short_m=float(box_short_m),
        box_height_m=float(box_height_m),
        projection_plane=projection_plane,
        support=support,
        confidence=confidence,
        failure_reasons=confidence.reasons,
    )


def discover_rim_plane(
    mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics | dict[str, Any],
    *,
    box_short_m: float = 0.335,
    min_evidence_pixels: int = 500,
    ransac_iterations: int = 250,
    ransac_tolerance_m: float = 0.006,
    min_dim_tolerance_m: float = 0.045,
    max_rms_m: float = 0.005,
    seed: int = 12345,
) -> dict[str, Any] | None:
    """Find the rim plane on a frame where the box top is well visible.

    Returns None unless the accepted plane footprint spans the known 0.335 m
    short side, which separates the rim from the front/back wall planes that
    dominate near frames.
    """

    mask_u8 = _require_mask(mask)
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2 or depth.shape != mask_u8.shape:
        raise ValueError(f"depth_m must be 2D and match mask shape, got {depth.shape} vs {mask_u8.shape}")
    intr = intrinsics if isinstance(intrinsics, CameraIntrinsics) else CameraIntrinsics.from_mapping(intrinsics)

    rows, cols = np.nonzero((mask_u8 > 0) & np.isfinite(depth) & (depth > 0.0))
    if rows.size < min_evidence_pixels:
        return None
    z = depth[rows, cols]
    x = (cols.astype(np.float64) - intr.cx) * z / intr.fx
    y = (rows.astype(np.float64) - intr.cy) * z / intr.fy
    sample = np.column_stack((x, y, z))[_even_indices(rows.size, 4000)]

    rng = np.random.default_rng(seed)
    discovered = _discover_plane(
        sample,
        box_short_m=box_short_m,
        iterations=ransac_iterations,
        tolerance_m=ransac_tolerance_m,
        rng=rng,
    )
    if discovered is None:
        return None
    normal, origin, support = discovered
    if abs(support["rim_min_dim_m"] - box_short_m) > min_dim_tolerance_m:
        return None
    if support["inlier_rms_m"] > max_rms_m:
        return None
    return {
        "normal": [float(v) for v in normal],
        "point": [float(v) for v in origin],
        "rim_min_dim_m": support["rim_min_dim_m"],
        "inlier_rms_m": support["inlier_rms_m"],
        "inlier_fraction": support["inlier_fraction"],
    }


def average_rim_planes(planes: list[dict[str, Any]], *, cluster_tolerance_m: float = 0.02) -> dict[str, Any]:
    """Robustly combine per-frame rim plane discoveries into one setup plane.

    Occasional per-frame discoveries latch onto a parallel interior plane
    below the rim, so the offsets are clustered and only the dominant cluster
    (ties broken toward the topmost plane) is averaged.
    """

    if not planes:
        raise ValueError("Need at least one discovered plane.")
    normals = np.asarray([plane["normal"] for plane in planes], dtype=np.float64)
    reference = normals[0]
    aligned = np.where((normals @ reference)[:, None] < 0.0, -normals, normals)
    normal = normalize(np.median(aligned, axis=0))
    offsets = np.asarray(
        [float(np.asarray(plane["point"], dtype=np.float64) @ normal) for plane in planes],
        dtype=np.float64,
    )

    order = np.argsort(offsets)
    clusters: list[list[int]] = []
    for index in order:
        if clusters and offsets[index] - offsets[clusters[-1][0]] <= cluster_tolerance_m:
            clusters[-1].append(int(index))
        else:
            clusters.append([int(index)])
    best_cluster = max(clusters, key=lambda c: (len(c), float(np.max(offsets[c]))))
    cluster_offsets = offsets[best_cluster]
    offset = float(np.median(cluster_offsets))
    cluster_normals = aligned[best_cluster]
    normal = normalize(np.median(cluster_normals, axis=0))
    return {
        "normal": [float(v) for v in normal],
        "point": [float(v) for v in normal * offset],
        "frames_used": len(best_cluster),
        "frames_discovered": len(planes),
        "offset_spread_m": float(np.max(cluster_offsets) - np.min(cluster_offsets)) if len(best_cluster) > 1 else 0.0,
    }


def _failure(
    reason: str,
    box_long_m: float,
    box_short_m: float,
    box_height_m: float,
    projection_plane: str,
    support: dict[str, Any],
) -> KnownSizeBoxEstimate:
    support = dict(support)
    support["method"] = "rim_plane"
    return _known_size_failure(
        reason,
        box_long_m=box_long_m,
        box_short_m=box_short_m,
        box_height_m=box_height_m,
        projection_plane=projection_plane,
        support=support,
    )


def _even_indices(count: int, limit: int) -> np.ndarray:
    if count <= limit:
        return np.arange(count)
    return np.linspace(0, count - 1, limit).astype(np.int64)


def _verify_fixed_plane(
    sample: np.ndarray,
    normal: np.ndarray,
    origin: np.ndarray,
    *,
    band_m: float,
    refine: bool,
    max_angle_deg: float,
    max_offset_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Gate the sample against the calibrated plane and optionally refine it.

    Refinement absorbs small base tilt/vibration but is rejected when it
    drifts too far from the calibrated plane, which would mean it latched
    onto a wall instead.
    """

    signed = (sample - origin) @ normal
    gated = np.abs(signed) <= band_m
    gated_count = int(np.count_nonzero(gated))
    gated_fraction = float(gated_count / max(sample.shape[0], 1))
    rms = float(np.sqrt(np.mean(signed[gated] ** 2))) if gated_count else float("inf")

    refined = False
    if refine and gated_count >= 100:
        pts = sample[gated]
        centroid = pts.mean(axis=0)
        _, _, vh = np.linalg.svd(pts - centroid, full_matrices=False)
        new_normal = normalize(vh[-1])
        if float(new_normal @ normal) < 0.0:
            new_normal = -new_normal
        angle = math.degrees(math.acos(min(max(float(new_normal @ normal), -1.0), 1.0)))
        offset = abs(float((centroid - origin) @ normal))
        if angle <= max_angle_deg and offset <= max_offset_m:
            normal = new_normal
            origin = centroid
            signed = (sample - origin) @ normal
            gated = np.abs(signed) <= band_m
            rms = float(np.sqrt(np.mean(signed[gated] ** 2))) if np.any(gated) else float("inf")
            refined = True

    quality = min(gated_fraction / 0.15, 1.0) * math.exp(-((rms / 0.006) ** 2)) if math.isfinite(rms) else 0.0
    return normal, origin, {
        "mode": "fixed",
        "refined": refined,
        "gated_fraction": gated_fraction,
        "inlier_rms_m": rms if math.isfinite(rms) else None,
        "quality": float(quality),
    }


def _ransac_plane(
    points: np.ndarray,
    *,
    iterations: int,
    tolerance_m: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    count = points.shape[0]
    if count < 3:
        return None
    best_inliers: np.ndarray | None = None
    best_count = -1
    for _ in range(iterations):
        idx = rng.choice(count, size=3, replace=False)
        p1, p2, p3 = points[idx]
        normal = np.cross(p2 - p1, p3 - p1)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            continue
        normal = normal / norm
        distances = np.abs((points - p1) @ normal)
        inliers = distances <= tolerance_m
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count > best_count:
            best_count = inlier_count
            best_inliers = inliers
    if best_inliers is None or best_count < 3:
        return None

    inlier_points = points[best_inliers]
    origin = inlier_points.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_points - origin, full_matrices=False)
    normal = normalize(vh[-1])
    if float(normal @ origin) > 0.0:
        normal = -normal
    distances = np.abs((points - origin) @ normal)
    inliers = distances <= tolerance_m
    return normal, origin, inliers


def _discover_plane(
    sample: np.ndarray,
    *,
    box_short_m: float,
    iterations: int,
    tolerance_m: float,
    rng: np.random.Generator,
    max_candidates: int = 4,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    """RANSAC plane candidates scored by rim-shaped footprint.

    The front wall is also a big planar mask region, but its projected extent
    is only the box height (~0.195 m) across, while the rim ring always spans
    the full 0.335 m short side. That extent separates the two.
    """

    min_points = max(150, sample.shape[0] // 12)
    remaining = np.arange(sample.shape[0])
    best: tuple[np.ndarray, np.ndarray, dict[str, Any]] | None = None
    best_score = -1.0
    candidate_count = 0
    while candidate_count < max_candidates and remaining.size >= max(min_points, 3):
        fit = _ransac_plane(sample[remaining], iterations=iterations, tolerance_m=tolerance_m, rng=rng)
        if fit is None:
            break
        normal, origin, inliers = fit
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count < min_points:
            break
        candidate_count += 1

        signed = (sample - origin) @ normal
        above_fraction = float(np.mean(signed > 0.025))
        if above_fraction > 0.08:
            # The rim is the topmost box surface; a plane with substantial
            # mask mass above it (toward the camera) is an interior surface.
            remaining = remaining[~inliers]
            continue
        gated_pts = sample[np.abs(signed) <= max(tolerance_m, 0.012)]
        e1, e2 = plane_basis(normal)
        coords = np.column_stack(((gated_pts - origin) @ e1, (gated_pts - origin) @ e2)).astype(np.float32)
        if coords.shape[0] >= 5:
            (_, _), (dim_a, dim_b), _ = cv2.minAreaRect(coords)
            min_dim = float(min(dim_a, dim_b))
        else:
            min_dim = 0.0
        dim_score = math.exp(-(((min_dim - box_short_m) / 0.06) ** 2))
        inlier_pts = sample[remaining][inliers]
        rms = float(np.sqrt(np.mean(((inlier_pts - origin) @ normal) ** 2)))
        score = inlier_count * (0.15 + 0.85 * dim_score)
        if score > best_score:
            best_score = score
            best = (
                normal,
                origin,
                {
                    "mode": "ransac",
                    "inlier_fraction": float(inlier_count / max(sample.shape[0], 1)),
                    "inlier_rms_m": rms,
                    "rim_dim_score": float(dim_score),
                    "rim_min_dim_m": min_dim,
                    "above_fraction": above_fraction,
                    "candidate_count": candidate_count,
                    "quality": float(dim_score * math.exp(-((rms / 0.006) ** 2))),
                },
            )
        remaining = remaining[~inliers]
    return best


def _intersect_pixels_with_plane(
    cols: np.ndarray,
    rows: np.ndarray,
    intr: CameraIntrinsics,
    normal: np.ndarray,
    origin: np.ndarray,
    *,
    eps: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    x = (cols.astype(np.float64) - intr.cx) / intr.fx
    y = (rows.astype(np.float64) - intr.cy) / intr.fy
    rays = np.column_stack((x, y, np.ones_like(x)))
    denom = rays @ normal
    numer = float(origin @ normal)
    valid = np.abs(denom) > eps
    t = np.zeros_like(denom)
    t[valid] = numer / denom[valid]
    valid &= t > eps
    return rays[valid] * t[valid, None], valid


def _filter_footprint_components(
    rows: np.ndarray,
    cols: np.ndarray,
    coords: np.ndarray,
    *,
    image_shape: tuple[int, int],
    max_diameter_m: float,
    min_component_points: int = 50,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Greedily keep gated components that fit inside one box footprint.

    Components are merged largest-first while the union's plane-space
    diameter stays within the box diagonal; distant blobs (arms, other
    objects at rim height) are dropped.
    """

    gated_image = np.zeros(image_shape, dtype=np.uint8)
    gated_image[rows, cols] = 255
    dilated = cv2.dilate(gated_image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    _, labels = cv2.connectedComponents((dilated > 0).astype(np.uint8), connectivity=8)
    point_labels = labels[rows, cols]
    sizes = np.bincount(point_labels)

    order = [int(label) for label in np.argsort(sizes)[::-1] if label != 0 and sizes[label] > 0]
    kept_labels: list[int] = []
    hull_pts: np.ndarray | None = None
    for label in order:
        if sizes[label] < min_component_points and kept_labels:
            continue
        comp_coords = coords[point_labels == label].astype(np.float32)
        comp_hull = cv2.convexHull(comp_coords).reshape(-1, 2)
        candidate = comp_hull if hull_pts is None else np.vstack((hull_pts, comp_hull))
        candidate_hull = cv2.convexHull(candidate).reshape(-1, 2)
        deltas = candidate_hull[:, None, :] - candidate_hull[None, :, :]
        diameter = float(np.sqrt((deltas**2).sum(axis=2)).max())
        if hull_pts is None or diameter <= max_diameter_m:
            kept_labels.append(label)
            hull_pts = candidate_hull

    keep = np.isin(point_labels, kept_labels)
    info = {
        "total": len(order),
        "kept": len(kept_labels),
        "dropped_points": int(rows.size - np.count_nonzero(keep)),
    }
    return keep, info


def _near_border(rows: np.ndarray, cols: np.ndarray, shape: tuple[int, int], margin: int) -> np.ndarray:
    height, width = shape
    return (rows < margin) | (rows >= height - margin) | (cols < margin) | (cols >= width - margin)


def _estimate_plane_yaw(
    coords: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    image_shape: tuple[int, int],
    border_margin_px: int,
    min_edge_m: float = 0.04,
) -> tuple[float, dict[str, Any]]:
    """Yaw mod 90 deg from convex-hull edge directions of the rim outline.

    Hull edges created by the image border (both endpoints near the border)
    follow the FOV boundary instead of the box, so they are excluded.
    """

    hull_idx = cv2.convexHull(coords.astype(np.float32), returnPoints=False).reshape(-1)
    hull_pts = coords[hull_idx]
    near_border = _near_border(rows[hull_idx], cols[hull_idx], image_shape, border_margin_px)
    occupancy, grid_min, cell_m = _occupancy_grid(coords)

    angles: list[float] = []
    weights: list[float] = []
    for i in range(hull_pts.shape[0]):
        j = (i + 1) % hull_pts.shape[0]
        if near_border[i] and near_border[j]:
            continue
        delta = hull_pts[j] - hull_pts[i]
        length = float(np.linalg.norm(delta))
        if length < min_edge_m:
            continue
        # Hull edges that jump across empty space (e.g. between the two rim
        # bands when the sides are cropped) are not box edges; real edges run
        # alongside rim points for their whole length.
        support = _edge_point_support(hull_pts[i], hull_pts[j], occupancy, grid_min, cell_m)
        if support < 0.5:
            continue
        angles.append(math.atan2(float(delta[1]), float(delta[0])))
        weights.append(length * length * support)

    if angles:
        quad = np.array(angles) * 4.0
        w = np.array(weights)
        yaw0 = math.atan2(float(np.average(np.sin(quad), weights=w)), float(np.average(np.cos(quad), weights=w))) / 4.0
    else:
        rect = cv2.minAreaRect(coords.astype(np.float32))
        yaw0 = math.radians(float(rect[2]))

    yaw = _refine_yaw(coords, yaw0)
    return yaw % (math.pi / 2.0), {
        "hull_edge_count": len(angles),
        "initial_yaw_deg": math.degrees(yaw0) % 90.0,
        "refined_yaw_deg": math.degrees(yaw) % 90.0,
    }


def _occupancy_grid(coords: np.ndarray, *, cell_m: float = 0.008) -> tuple[np.ndarray, np.ndarray, float]:
    grid_min = coords.min(axis=0)
    cells = np.floor((coords - grid_min) / cell_m).astype(np.int64)
    shape = cells.max(axis=0) + 1
    occupancy = np.zeros((int(shape[0]) + 2, int(shape[1]) + 2), dtype=bool)
    occupancy[cells[:, 0] + 1, cells[:, 1] + 1] = True
    return occupancy, grid_min, cell_m


def _edge_point_support(
    start: np.ndarray,
    end: np.ndarray,
    occupancy: np.ndarray,
    grid_min: np.ndarray,
    cell_m: float,
) -> float:
    """Fraction of the segment that runs next to observed rim points."""

    length = float(np.linalg.norm(end - start))
    count = max(int(length / cell_m), 2)
    alpha = np.linspace(0.0, 1.0, count)
    samples = start[None, :] * (1.0 - alpha[:, None]) + end[None, :] * alpha[:, None]
    cells = np.floor((samples - grid_min) / cell_m).astype(np.int64) + 1
    cells[:, 0] = np.clip(cells[:, 0], 1, occupancy.shape[0] - 2)
    cells[:, 1] = np.clip(cells[:, 1], 1, occupancy.shape[1] - 2)
    hit = np.zeros(count, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            hit |= occupancy[cells[:, 0] + di, cells[:, 1] + dj]
    return float(np.mean(hit))


def _refine_yaw(coords: np.ndarray, yaw0: float, *, half_range_deg: float = 4.0, step_deg: float = 0.25) -> float:
    sample = coords[_even_indices(coords.shape[0], 8000)]
    best_yaw = yaw0
    best_score = -1.0
    for offset in np.arange(-half_range_deg, half_range_deg + 1e-9, step_deg):
        yaw = yaw0 + math.radians(float(offset))
        score = _yaw_alignment_score(sample, yaw)
        if score > best_score:
            best_score = score
            best_yaw = yaw
    return best_yaw


def _yaw_alignment_score(coords: np.ndarray, yaw: float, *, band_m: float = 0.010) -> float:
    """Count points concentrated at the four extreme edges for this yaw."""

    c, s = math.cos(yaw), math.sin(yaw)
    score = 0.0
    for axis in (np.array([c, s]), np.array([-s, c])):
        proj = coords @ axis
        lo, hi = np.quantile(proj, [0.005, 0.995])
        score += float(np.count_nonzero(proj >= hi - band_m) + np.count_nonzero(proj <= lo + band_m))
    return score


def _axis_edges(
    proj: np.ndarray,
    perp: np.ndarray,
    near_border: np.ndarray,
    band_m: float,
    *,
    edge_inset_m: float = 0.008,
) -> dict[str, Any]:
    lo, hi = (float(v) for v in np.quantile(proj, [0.003, 0.997]))
    # The observed extremes overshoot the true outer edges: the HSV mask
    # bleeds past the physical edge and front-wall pixels inside the rim band
    # ray-project slightly outside the footprint. The inset compensates for
    # that systematic smear (measured ~15 mm of span excess on the recordings).
    return {
        "lo": lo + edge_inset_m,
        "hi": hi - edge_inset_m,
        "span": (hi - lo) - 2.0 * edge_inset_m,
        "lo_edge": _edge_trust(proj, perp, near_border, lo, lo + band_m),
        "hi_edge": _edge_trust(proj, perp, near_border, hi - band_m, hi),
    }


def _edge_trust(
    proj: np.ndarray,
    perp: np.ndarray,
    near_border: np.ndarray,
    band_lo: float,
    band_hi: float,
    *,
    min_points: int = 40,
    max_border_fraction: float = 0.10,
    min_coverage_m: float = 0.08,
) -> dict[str, Any]:
    in_band = (proj >= band_lo) & (proj <= band_hi)
    count = int(np.count_nonzero(in_band))
    if count == 0:
        return {"trusted": False, "points": 0, "border_fraction": 1.0, "coverage_m": 0.0}
    border_fraction = float(np.count_nonzero(near_border[in_band]) / count)
    perp_band = perp[in_band]
    q05, q95 = np.quantile(perp_band, [0.05, 0.95])
    coverage = float(q95 - q05)
    trusted = count >= min_points and border_fraction <= max_border_fraction and coverage >= min_coverage_m
    return {
        "trusted": bool(trusted),
        "points": count,
        "border_fraction": border_fraction,
        "coverage_m": coverage,
    }


def _assign_axes(
    edge_u: dict[str, Any],
    edge_v: dict[str, Any],
    u_dir: np.ndarray,
    v_dir: np.ndarray,
    box_long_m: float,
    box_short_m: float,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray, float]:
    """Decide which perpendicular direction is the 505 mm long axis.

    Simply taking the larger observed span fails when both short sides are
    cropped and the visible long span shrinks below 335 mm, so both
    assignments are scored against the known dimensions instead.
    """

    def assignment_penalty(long_edge: dict[str, Any], short_edge: dict[str, Any]) -> float:
        penalty = 0.0
        for edge, dim in ((long_edge, box_long_m), (short_edge, box_short_m)):
            span = float(edge["span"])
            if edge["lo_edge"]["trusted"] and edge["hi_edge"]["trusted"]:
                penalty += abs(span - dim)
            elif span > dim + 0.03:
                penalty += span - dim
        return penalty

    penalty_uv = assignment_penalty(edge_u, edge_v)
    penalty_vu = assignment_penalty(edge_v, edge_u)
    if penalty_uv < penalty_vu or (penalty_uv == penalty_vu and edge_u["span"] >= edge_v["span"]):
        return edge_u, edge_v, u_dir, v_dir, penalty_uv
    return edge_v, edge_u, v_dir, u_dir, penalty_vu


def _axis_center(
    edges: dict[str, Any],
    dim_m: float,
    axis_name: str,
    reasons: list[str],
    *,
    max_span_error_m: float = 0.035,
) -> tuple[float, dict[str, Any]]:
    lo = float(edges["lo"])
    hi = float(edges["hi"])
    lo_trusted = bool(edges["lo_edge"]["trusted"])
    hi_trusted = bool(edges["hi_edge"]["trusted"])
    span = hi - lo
    span_error: float | None = None

    if lo_trusted and hi_trusted:
        center = (lo + hi) * 0.5
        span_error = span - dim_m
        if abs(span_error) > max_span_error_m:
            reasons.append(f"{axis_name}_span_mismatch")
        mode = "both_edges"
    elif lo_trusted:
        center = lo + dim_m * 0.5
        mode = "lo_edge_only"
    elif hi_trusted:
        center = hi - dim_m * 0.5
        mode = "hi_edge_only"
    else:
        center = (lo + hi) * 0.5
        reasons.append(f"{axis_name}_center_underconstrained")
        mode = "underconstrained"

    info = {
        "mode": mode,
        "span_m": span,
        "span_error_m": span_error,
        "trusted_edges": int(lo_trusted) + int(hi_trusted),
        "lo_edge": edges["lo_edge"],
        "hi_edge": edges["hi_edge"],
    }
    return float(center), info


def _project_point(intr: CameraIntrinsics, point: np.ndarray) -> np.ndarray:
    return np.array(
        [intr.fx * point[0] / point[2] + intr.cx, intr.fy * point[1] / point[2] + intr.cy],
        dtype=np.float64,
    )


def _project_axis(intr: CameraIntrinsics, center: np.ndarray, direction: np.ndarray, *, step_m: float = 0.05) -> np.ndarray:
    tip = center + direction * step_m
    delta = _project_point(intr, tip) - _project_point(intr, center)
    return normalize(delta)


def _confidence_score(
    reasons: list[str],
    *,
    plane_quality: float,
    trusted_edges: int,
    span_errors_m: list[float],
) -> float:
    raw = 0.25
    raw += 0.20 * min(max(plane_quality, 0.0), 1.0)
    raw += 0.30 * min(max(trusted_edges / 4.0, 0.0), 1.0)
    if span_errors_m:
        worst = max(abs(float(err)) for err in span_errors_m)
        raw += 0.25 * math.exp(-((worst / 0.02) ** 2))
    else:
        raw += 0.10
    if reasons:
        raw *= 0.45
    return float(round(min(max(raw, 0.0), 1.0), 3))
