from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from box_pose import (
    CameraIntrinsics,
    MaskStats,
    evaluate_still_frame_spread,
    estimate_known_size_box,
    estimate_metric_box,
    estimate_pixel_box,
    safe_output_dict,
    segment_yellow_box,
)


class BoxPoseTests(unittest.TestCase):
    def test_hsv_segmentation_keeps_dominant_box_over_noise(self) -> None:
        image = np.zeros((160, 220, 3), dtype=np.uint8)
        cv2.rectangle(image, (45, 45), (175, 115), (0, 190, 255), -1)
        cv2.circle(image, (15, 20), 2, (0, 190, 255), -1)
        cv2.circle(image, (205, 140), 2, (0, 190, 255), -1)

        mask, stats = segment_yellow_box(image)

        self.assertGreater(stats.dominant_area, 8000)
        self.assertGreater(stats.dominant_fraction, 0.95)
        self.assertEqual(stats.significant_components, 1)
        self.assertEqual(int(mask[80, 110]), 255)
        self.assertEqual(int(mask[20, 15]), 0)

    def test_hsv_segmentation_can_keep_all_cleaned_components(self) -> None:
        image = np.zeros((160, 220, 3), dtype=np.uint8)
        cv2.rectangle(image, (30, 45), (80, 115), (0, 190, 255), -1)
        cv2.rectangle(image, (140, 45), (190, 115), (0, 190, 255), -1)

        dominant, dominant_stats = segment_yellow_box(image)
        evidence, evidence_stats = segment_yellow_box(image, keep_largest_component=False)

        self.assertLess(np.count_nonzero(dominant), np.count_nonzero(evidence))
        self.assertEqual(dominant_stats.significant_components, 2)
        self.assertEqual(evidence_stats.significant_components, 2)
        self.assertEqual(int(evidence[80, 55]), 255)
        self.assertEqual(int(evidence[80, 165]), 255)

    def test_synthetic_rotated_rectangles_normalize_yaw_mod_180(self) -> None:
        for angle in (-75, -30, 0, 25, 70):
            mask = rotated_rectangle_mask((160, 160), (80, 40), angle)
            stats = stats_for_mask(mask)
            estimate = estimate_pixel_box(mask, stats)
            self.assertTrue(estimate.confidence.ok, estimate.failure_reasons)
            self.assertLessEqual(angle_error_mod_180(estimate.yaw_mod_180, angle), 2.0)

    def test_yaw_normalization_boundary_cases(self) -> None:
        for angle in (0, 1, 89, 90, 91, 179):
            mask = rotated_rectangle_mask((180, 180), (90, 35), angle)
            estimate = estimate_pixel_box(mask, stats_for_mask(mask))
            self.assertTrue(estimate.confidence.ok, (angle, estimate.failure_reasons))
            self.assertLessEqual(angle_error_mod_180(estimate.yaw_mod_180, angle), 2.0)

    def test_long_short_swap_preserves_long_axis_identity(self) -> None:
        mask = rotated_rectangle_mask((160, 160), (110, 35), 82)
        estimate = estimate_pixel_box(mask, stats_for_mask(mask))

        self.assertTrue(estimate.confidence.ok, estimate.failure_reasons)
        self.assertGreater(estimate.pixel_obb["long_length_px"], estimate.pixel_obb["short_length_px"])
        self.assertGreaterEqual(estimate.pixel_obb["aspect_ratio"], 1.25)
        self.assertAlmostEqual(float(np.dot(estimate.long_axis_image, estimate.short_axis_image)), 0.0, places=6)

    def test_near_square_returns_low_confidence(self) -> None:
        mask = rotated_rectangle_mask((140, 140), (60, 56), 15)
        estimate = estimate_pixel_box(mask, stats_for_mask(mask))

        self.assertFalse(estimate.confidence.ok)
        self.assertIn("aspect_ratio_ambiguous", estimate.failure_reasons)

    def test_tiny_mask_below_area_threshold_returns_low_confidence(self) -> None:
        mask = rotated_rectangle_mask((220, 220), (30, 16), 0)
        estimate = estimate_pixel_box(mask, stats_for_mask(mask))

        self.assertFalse(estimate.confidence.ok)
        self.assertIn("mask_area_too_small", estimate.failure_reasons)

    def test_sparse_partial_contour_fill_returns_low_confidence(self) -> None:
        mask = np.zeros((180, 180), dtype=np.uint8)
        cv2.rectangle(mask, (35, 50), (145, 130), 255, thickness=4)
        estimate = estimate_pixel_box(mask, stats_for_mask(mask))

        self.assertFalse(estimate.confidence.ok)
        self.assertIn("partial_or_sparse_contour", estimate.failure_reasons)

    def test_fragmented_mask_returns_low_confidence(self) -> None:
        mask = rotated_rectangle_mask((200, 200), (80, 35), 0)
        cv2.rectangle(mask, (10, 10), (35, 35), 255, -1)
        cv2.rectangle(mask, (160, 20), (190, 45), 255, -1)
        cv2.rectangle(mask, (20, 160), (48, 190), 255, -1)
        estimate = estimate_pixel_box(mask, stats_for_mask(mask))

        self.assertFalse(estimate.confidence.ok)
        self.assertIn("mask_fragmented", estimate.failure_reasons)

    def test_phase0_perspective_output_is_image_frame_only(self) -> None:
        mask = np.zeros((180, 220), dtype=np.uint8)
        trapezoid = np.array([[50, 40], [170, 60], [150, 140], [70, 125]], dtype=np.int32)
        cv2.fillConvexPoly(mask, trapezoid, 255)
        estimate = estimate_pixel_box(mask, stats_for_mask(mask))
        output = estimate.to_dict()

        self.assertEqual(output["yaw_frame"], "image")
        self.assertNotIn("camera_T_box", output)
        self.assertNotIn("t5_T_box", output)
        self.assertNotIn("center_m", output)
        self.assertNotIn("metric_tolerance_cm", output)

    def test_output_schema_rejects_unsafe_command_fields(self) -> None:
        with self.assertRaises(ValueError):
            safe_output_dict({"target_t5_T_ee": np.eye(4).tolist()})

    def test_output_schema_rejects_nested_unsafe_command_fields(self) -> None:
        with self.assertRaises(ValueError):
            safe_output_dict({"debug": {"command": {"target_t5_T_ee": np.eye(4).tolist()}}})

    def test_offline_pallet_image_segments_and_estimates_pixel_obb(self) -> None:
        image = cv2.imread(str(Path(__file__).resolve().parents[1] / "pallet_box.png"), cv2.IMREAD_COLOR)
        self.assertIsNotNone(image)
        mask, stats = segment_yellow_box(image)
        estimate = estimate_pixel_box(mask, stats)

        self.assertGreater(stats.dominant_area, 0)
        self.assertTrue(np.isfinite(estimate.yaw_mod_180))
        self.assertTrue(np.all(np.isfinite(estimate.long_axis_image)))
        self.assertTrue(np.all(np.isfinite(estimate.short_axis_image)))
        self.assertIn("pixel_obb", estimate.to_dict())

    def test_metric_projection_returns_camera_transform_and_axes(self) -> None:
        mask = rotated_rectangle_mask((160, 160), (80, 40), 0)
        depth = np.where(mask > 0, 1.0, 0.0).astype(np.float64)
        intr = CameraIntrinsics(fx=100.0, fy=100.0, cx=80.0, cy=80.0)

        estimate = estimate_metric_box(mask, depth, intr, min_boundary_depth_points=20)

        self.assertTrue(estimate.confidence.ok, estimate.failure_reasons)
        self.assertIsNotNone(estimate.camera_T_box)
        self.assertEqual(estimate.yaw_frame, "camera_table_plane")
        rot = estimate.camera_T_box[:3, :3]
        np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1e-8)
        np.testing.assert_allclose(estimate.camera_T_box[:3, 3], np.array([0.0, 0.0, 1.0]), atol=0.02)
        self.assertLessEqual(angle_error_mod_180(estimate.yaw_mod_180, 0.0), 2.0)
        self.assertAlmostEqual(estimate.long_length_m, 0.8, delta=0.04)
        self.assertAlmostEqual(estimate.short_length_m, 0.4, delta=0.04)

    def test_metric_projection_uses_boundary_not_corrupt_interior_depth(self) -> None:
        mask = rotated_rectangle_mask((160, 160), (80, 40), 0)
        depth = np.where(mask > 0, 2.5, 0.0).astype(np.float64)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(contours[0])
        boundary = np.zeros_like(mask)
        cv2.drawContours(boundary, [hull], -1, 255, thickness=1)
        depth[boundary > 0] = 1.0
        intr = CameraIntrinsics(fx=100.0, fy=100.0, cx=80.0, cy=80.0)

        estimate = estimate_metric_box(mask, depth, intr, min_boundary_depth_points=20)

        self.assertTrue(estimate.confidence.ok, estimate.failure_reasons)
        self.assertAlmostEqual(float(estimate.camera_T_box[2, 3]), 1.0, places=6)

    def test_metric_projection_rejects_spatially_biased_boundary_depth(self) -> None:
        mask = rotated_rectangle_mask((160, 160), (80, 40), 0)
        depth = np.zeros_like(mask, dtype=np.float64)
        depth[60, 40:121] = 1.0
        intr = CameraIntrinsics(fx=100.0, fy=100.0, cx=80.0, cy=80.0)

        estimate = estimate_metric_box(mask, depth, intr, min_boundary_depth_points=20)

        self.assertFalse(estimate.confidence.ok)
        self.assertIn("boundary_support_biased", estimate.failure_reasons)

    def test_metric_projection_table_plane_fallback(self) -> None:
        mask = rotated_rectangle_mask((160, 160), (80, 40), 0)
        depth = np.zeros_like(mask, dtype=np.float64)
        intr = CameraIntrinsics(fx=100.0, fy=100.0, cx=80.0, cy=80.0)

        estimate = estimate_metric_box(
            mask,
            depth,
            intr,
            table_plane=(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0])),
            min_boundary_depth_points=20,
        )

        self.assertTrue(estimate.confidence.ok, estimate.failure_reasons)
        self.assertAlmostEqual(float(estimate.camera_T_box[2, 3]), 1.0, places=6)

    def test_metric_projection_invalid_table_plane_fallback_fails(self) -> None:
        mask = rotated_rectangle_mask((160, 160), (80, 40), 0)
        depth = np.zeros_like(mask, dtype=np.float64)
        intr = CameraIntrinsics(fx=100.0, fy=100.0, cx=80.0, cy=80.0)

        estimate = estimate_metric_box(
            mask,
            depth,
            intr,
            table_plane=(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])),
            min_boundary_depth_points=20,
        )

        self.assertFalse(estimate.confidence.ok)
        self.assertIn("invalid_table_plane_fallback", estimate.failure_reasons)

    def test_metric_projection_rejects_depth_inconsistent_table_plane_fallback(self) -> None:
        mask = rotated_rectangle_mask((160, 160), (80, 40), 0)
        depth = np.zeros_like(mask, dtype=np.float64)
        for x, y in boundary_sample(mask, count=29):
            depth[y, x] = 2.0
        intr = CameraIntrinsics(fx=100.0, fy=100.0, cx=80.0, cy=80.0)

        estimate = estimate_metric_box(
            mask,
            depth,
            intr,
            table_plane=(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0])),
            min_boundary_depth_points=30,
        )

        self.assertFalse(estimate.confidence.ok)
        self.assertIn("depth_plane_inconsistent", estimate.failure_reasons)

    def test_metric_projection_computes_t5_transform(self) -> None:
        mask = rotated_rectangle_mask((160, 160), (80, 40), 0)
        depth = np.where(mask > 0, 1.0, 0.0).astype(np.float64)
        intr = CameraIntrinsics(fx=100.0, fy=100.0, cx=80.0, cy=80.0)
        t5_T_camera = np.eye(4, dtype=np.float64)
        t5_T_camera[:3, 3] = [0.1, 0.2, 0.3]

        estimate = estimate_metric_box(mask, depth, intr, t5_T_camera=t5_T_camera, min_boundary_depth_points=20)

        self.assertTrue(estimate.confidence.ok, estimate.failure_reasons)
        np.testing.assert_allclose(estimate.t5_T_box, t5_T_camera @ estimate.camera_T_box, atol=1e-8)
        self.assertEqual(estimate.yaw_frame, "t5")

    def test_metric_projection_fails_without_depth_support(self) -> None:
        mask = rotated_rectangle_mask((160, 160), (80, 40), 0)
        depth = np.zeros_like(mask, dtype=np.float64)
        intr = CameraIntrinsics(fx=100.0, fy=100.0, cx=80.0, cy=80.0)

        estimate = estimate_metric_box(mask, depth, intr)

        self.assertFalse(estimate.confidence.ok)
        self.assertIn("insufficient_boundary_depth", estimate.failure_reasons)

    def test_known_size_box_recovers_center_and_yaw_from_cropped_rim(self) -> None:
        mask = cropped_rim_mask(
            shape=(320, 300),
            center=(150.0, 160.0),
            long_len=360.0,
            short_len=115.0,
            yaw_deg=0.0,
        )
        depth = np.where(mask > 0, 1.0, 0.0).astype(np.float64)
        intr = CameraIntrinsics(fx=712.0, fy=712.0, cx=150.0, cy=160.0)

        estimate = estimate_known_size_box(
            mask,
            depth,
            intr,
            box_long_m=360.0 / 712.0,
            box_short_m=115.0 / 712.0,
            min_depth_pixels=50,
            top_plane_short_projection_scale=1.0,
        )

        self.assertTrue(estimate.confidence.ok, estimate.failure_reasons)
        self.assertIsNotNone(estimate.center_image)
        self.assertLess(np.linalg.norm(np.asarray(estimate.center_image) - np.array([150.0, 160.0])), 6.0)
        self.assertLessEqual(angle_error_mod_180(float(estimate.yaw_mod_180), 0.0), 2.0)

    def test_known_size_box_recovers_yaw_from_partial_rotated_rim(self) -> None:
        mask = cropped_rim_mask(
            shape=(340, 320),
            center=(160.0, 172.0),
            long_len=330.0,
            short_len=120.0,
            yaw_deg=22.0,
        )
        depth = np.where(mask > 0, 1.0, 0.0).astype(np.float64)
        intr = CameraIntrinsics(fx=650.0, fy=650.0, cx=160.0, cy=172.0)

        estimate = estimate_known_size_box(
            mask,
            depth,
            intr,
            box_long_m=330.0 / 650.0,
            box_short_m=120.0 / 650.0,
            min_depth_pixels=50,
        )

        self.assertTrue(estimate.confidence.ok, estimate.failure_reasons)
        self.assertLessEqual(angle_error_mod_180(float(estimate.yaw_mod_180), 22.0), 4.0)

    def test_known_size_box_prefers_top_rim_over_lower_wall_line(self) -> None:
        mask = np.zeros((280, 360), dtype=np.uint8)
        cv2.line(mask, (45, 60), (315, 60), 255, 5)
        cv2.line(mask, (45, 160), (315, 160), 255, 5)
        cv2.line(mask, (45, 230), (315, 230), 255, 5)
        depth = np.zeros_like(mask, dtype=np.float64)
        depth[mask > 0] = 0.55
        depth[55:66, 40:321] = np.where(mask[55:66, 40:321] > 0, 0.66, depth[55:66, 40:321])
        depth[155:166, 40:321] = np.where(mask[155:166, 40:321] > 0, 0.42, depth[155:166, 40:321])
        depth[225:236, 40:321] = np.where(mask[225:236, 40:321] > 0, 0.55, depth[225:236, 40:321])
        intr = CameraIntrinsics(fx=650.0, fy=650.0, cx=180.0, cy=140.0)

        estimate = estimate_known_size_box(
            mask,
            depth,
            intr,
            box_long_m=270.0 * 0.55 / 650.0,
            box_short_m=(100.0 / 0.72) * 0.55 / 650.0,
            min_depth_pixels=50,
        )

        self.assertTrue(estimate.confidence.ok, estimate.failure_reasons)
        self.assertIsNotNone(estimate.center_image)
        self.assertLess(abs(float(estimate.center_image[1]) - 110.0), 15.0)
        self.assertLess(abs(float(estimate.model_short_length_px) - 100.0), 20.0)

    def test_known_size_box_clamps_lower_wall_pair_to_top_plane_extent(self) -> None:
        mask = np.zeros((280, 360), dtype=np.uint8)
        cv2.line(mask, (45, 60), (315, 60), 255, 5)
        cv2.line(mask, (45, 230), (315, 230), 255, 5)
        depth = np.zeros_like(mask, dtype=np.float64)
        depth[55:66, 40:321] = np.where(mask[55:66, 40:321] > 0, 0.66, depth[55:66, 40:321])
        depth[225:236, 40:321] = np.where(mask[225:236, 40:321] > 0, 0.42, depth[225:236, 40:321])
        intr = CameraIntrinsics(fx=650.0, fy=650.0, cx=180.0, cy=140.0)

        estimate = estimate_known_size_box(
            mask,
            depth,
            intr,
            box_long_m=270.0 * 0.54 / 650.0,
            box_short_m=(100.0 / 0.72) * 0.54 / 650.0,
            min_depth_pixels=50,
        )

        self.assertTrue(estimate.confidence.ok, estimate.failure_reasons)
        self.assertIsNotNone(estimate.center_image)
        self.assertLess(abs(float(estimate.center_image[1]) - 110.0), 15.0)
        self.assertLess(abs(float(estimate.model_short_length_px) - 100.0), 20.0)
        support = estimate.support.get("short_axis_pair_support", {})
        self.assertTrue(support.get("top_plane_extent_clamped"))
        self.assertGreater(float(support.get("raw_separation_px")), float(support.get("clamped_separation_px")))

    def test_known_size_box_searches_scale_for_better_perimeter_fit(self) -> None:
        mask = cropped_rim_mask(
            shape=(320, 360),
            center=(180.0, 160.0),
            long_len=313.2,
            short_len=116.0,
            yaw_deg=0.0,
        )
        depth = np.where(mask > 0, 1.0, 0.0).astype(np.float64)
        intr = CameraIntrinsics(fx=700.0, fy=700.0, cx=180.0, cy=160.0)

        estimate = estimate_known_size_box(
            mask,
            depth,
            intr,
            box_long_m=270.0 / 700.0,
            box_short_m=100.0 / 700.0,
            min_depth_pixels=50,
            top_plane_short_projection_scale=1.0,
        )

        self.assertTrue(estimate.confidence.ok, estimate.failure_reasons)
        fit_search = estimate.support.get("fit_search", {})
        self.assertGreater(float(fit_search.get("selected_common_scale")), 1.0)
        self.assertGreater(float(estimate.model_long_length_px), 285.0)
        self.assertGreater(float(estimate.model_short_length_px), 110.0)
        perimeter = estimate.support.get("perimeter", {})
        self.assertGreaterEqual(float(perimeter.get("edge_support_fraction")), 0.95)
        self.assertGreaterEqual(float(perimeter.get("mask_support_fraction")), 0.95)

    def test_confidence_threshold_edges(self) -> None:
        base = rotated_rectangle_mask((100, 100), (60, 20), 0)
        base_stats = stats_for_mask(base)
        exact_area_stats = MaskStats(
            int(round(base_stats.dominant_area / 0.03)),
            base_stats.total_mask_area,
            base_stats.dominant_area,
            1.0,
            1,
        )
        self.assertTrue(estimate_pixel_box(base, exact_area_stats).confidence.ok)

        weak_dominant = MaskStats(10000, 1000, 699, 0.699, 1)
        self.assertIn("dominant_component_too_weak", estimate_pixel_box(base, weak_dominant).failure_reasons)

        three_components = MaskStats(10000, 1000, 900, 0.9, 3)
        self.assertTrue(estimate_pixel_box(base, three_components).confidence.ok)
        four_components = MaskStats(10000, 1000, 900, 0.9, 4)
        self.assertIn("mask_fragmented", estimate_pixel_box(base, four_components).failure_reasons)

    def test_still_frame_spread_confidence_gate(self) -> None:
        stable = evaluate_still_frame_spread(
            np.array([[0.0, 0.0, 1.0], [0.004, 0.0, 1.0], [0.002, 0.003, 1.0]]),
            np.array([179.0, 1.0, 0.5]),
        )
        self.assertTrue(stable.ok, stable.reasons)

        unstable = evaluate_still_frame_spread(
            np.array([[0.0, 0.0, 1.0], [0.02, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            np.array([0.0, 12.0, 1.0]),
        )
        self.assertFalse(unstable.ok)
        self.assertIn("center_spread_too_large", unstable.reasons)
        self.assertIn("yaw_spread_too_large", unstable.reasons)


def rotated_rectangle_mask(shape: tuple[int, int], size: tuple[float, float], angle_deg: float) -> np.ndarray:
    h, w = shape
    rect = ((w / 2.0, h / 2.0), (float(size[0]), float(size[1])), float(angle_deg))
    corners = cv2.boxPoints(rect).astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, corners, 255)
    return mask


def cropped_rim_mask(
    *,
    shape: tuple[int, int],
    center: tuple[float, float],
    long_len: float,
    short_len: float,
    yaw_deg: float,
) -> np.ndarray:
    h, w = shape
    angle = math.radians(yaw_deg)
    u = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
    v = np.array([-math.sin(angle), math.cos(angle)], dtype=np.float64)
    c = np.array(center, dtype=np.float64)
    corners = np.array(
        [
            c - long_len * 0.5 * u - short_len * 0.5 * v,
            c + long_len * 0.5 * u - short_len * 0.5 * v,
            c + long_len * 0.5 * u + short_len * 0.5 * v,
            c - long_len * 0.5 * u + short_len * 0.5 * v,
        ],
        dtype=np.int32,
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.polylines(mask, [corners], isClosed=True, color=255, thickness=14)
    return mask


def stats_for_mask(mask: np.ndarray) -> MaskStats:
    mask_bool = mask > 0
    area = int(mask_bool.size)
    count = int(np.count_nonzero(mask_bool))
    if count == 0:
        return MaskStats(area, 0, 0, 0.0, 0)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(mask_bool.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if component_count > 1 else np.array([], dtype=np.int32)
    dominant = int(np.max(areas)) if areas.size else 0
    threshold = max(int(round(dominant * 0.05)), 1)
    significant = int(np.count_nonzero(areas >= threshold))
    return MaskStats(area, count, dominant, float(dominant / count), significant)


def angle_error_mod_180(actual: float, expected: float) -> float:
    delta = abs((actual - expected) % 180.0)
    return min(delta, 180.0 - delta)


def boundary_sample(mask: np.ndarray, *, count: int) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(contours[0])
    boundary = np.zeros_like(mask)
    cv2.drawContours(boundary, [hull], -1, 255, thickness=1)
    rows, cols = np.nonzero(boundary > 0)
    pixels = np.column_stack((cols, rows)).astype(np.int32)
    return pixels[:count]


if __name__ == "__main__":
    unittest.main()
