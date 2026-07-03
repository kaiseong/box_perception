from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from box_pose import (
    CameraIntrinsics,
    average_rim_planes,
    discover_rim_plane,
    estimate_plane_box,
)

BOX_LONG = 0.505
BOX_SHORT = 0.335
BOX_HEIGHT = 0.195
RIM_WIDTH = 0.03


def make_scene(
    *,
    image_width: int = 520,
    image_height: int = 760,
    yaw_plane_deg: float = 12.0,
    center_shift_long_m: float = 0.0,
    with_wall: bool = True,
    depth_noise_m: float = 0.002,
    seed: int = 7,
) -> dict:
    """Render a synthetic rim ring (plus front wall) seen by a tilted camera."""

    rng = np.random.default_rng(seed)
    intr = CameraIntrinsics(fx=340.0, fy=340.0, cx=image_width / 2.0, cy=image_height / 2.0)

    normal_away = np.array([0.0, 0.766, 0.643])
    normal_away /= np.linalg.norm(normal_away)
    plane_point = np.array([0.0, 0.10, 0.46])

    e_a = np.array([1.0, 0.0, 0.0])
    e_a = e_a - (e_a @ normal_away) * normal_away
    e_a /= np.linalg.norm(e_a)
    e_b = np.cross(normal_away, e_a)
    phi = math.radians(yaw_plane_deg)
    long_3d = math.cos(phi) * e_a + math.sin(phi) * e_b
    short_3d = np.cross(normal_away, long_3d)
    center_3d = plane_point + center_shift_long_m * long_3d

    depth = np.zeros((image_height, image_width), dtype=np.float32)
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    us, vs = np.meshgrid(np.arange(image_width, dtype=np.float64), np.arange(image_height, dtype=np.float64))
    ray_x = (us - intr.cx) / intr.fx
    ray_y = (vs - intr.cy) / intr.fy
    denom = ray_x * normal_away[0] + ray_y * normal_away[1] + normal_away[2]
    t = np.where(np.abs(denom) > 1e-9, (plane_point @ normal_away) / denom, np.nan)
    px = ray_x * t
    py = ray_y * t
    pz = t
    rel = np.stack((px - center_3d[0], py - center_3d[1], pz - center_3d[2]), axis=-1)
    a = rel @ long_3d
    b = rel @ short_3d
    outer = (np.abs(a) <= BOX_LONG / 2) & (np.abs(b) <= BOX_SHORT / 2)
    inner = (np.abs(a) <= BOX_LONG / 2 - RIM_WIDTH) & (np.abs(b) <= BOX_SHORT / 2 - RIM_WIDTH)
    ring = outer & ~inner & (t > 0)
    mask[ring] = 255
    depth[ring] = (pz[ring] + rng.normal(0.0, depth_noise_m, int(np.count_nonzero(ring)))).astype(np.float32)

    if with_wall:
        front_sign = 1.0
        if (center_3d + (BOX_SHORT / 2) * short_3d)[2] > (center_3d - (BOX_SHORT / 2) * short_3d)[2]:
            front_sign = -1.0
        aa, hh = np.meshgrid(
            np.linspace(-BOX_LONG / 2, BOX_LONG / 2, 900),
            np.linspace(0.005, BOX_HEIGHT, 320),
        )
        wall = (
            center_3d[None, None, :]
            + aa[..., None] * long_3d[None, None, :]
            + front_sign * (BOX_SHORT / 2) * short_3d[None, None, :]
            + hh[..., None] * normal_away[None, None, :]
        ).reshape(-1, 3)
        wall_u = np.rint(intr.fx * wall[:, 0] / wall[:, 2] + intr.cx).astype(int)
        wall_v = np.rint(intr.fy * wall[:, 1] / wall[:, 2] + intr.cy).astype(int)
        inside = (wall_u >= 0) & (wall_u < image_width) & (wall_v >= 0) & (wall_v < image_height)
        wall_u, wall_v, wall_z = wall_u[inside], wall_v[inside], wall[inside, 2]
        empty = mask[wall_v, wall_u] == 0
        mask[wall_v[empty], wall_u[empty]] = 255
        depth[wall_v[empty], wall_u[empty]] = (
            wall_z[empty] + rng.normal(0.0, depth_noise_m, int(np.count_nonzero(empty)))
        ).astype(np.float32)

    tip = center_3d + 0.05 * long_3d
    tip_px = np.array([intr.fx * tip[0] / tip[2] + intr.cx, intr.fy * tip[1] / tip[2] + intr.cy])
    center_px = np.array(
        [intr.fx * center_3d[0] / center_3d[2] + intr.cx, intr.fy * center_3d[1] / center_3d[2] + intr.cy]
    )
    delta = tip_px - center_px
    expected_yaw = math.degrees(math.atan2(delta[1], delta[0])) % 180.0

    return {
        "mask": mask,
        "depth": depth,
        "intrinsics": intr,
        "center_3d": center_3d,
        "long_3d": long_3d,
        "short_3d": short_3d,
        "expected_yaw_image": expected_yaw,
    }


def yaw_error_mod_180(a: float, b: float) -> float:
    delta = abs((a - b) % 180.0)
    return min(delta, 180.0 - delta)


class PlaneFitTests(unittest.TestCase):
    def test_full_box_recovers_center_and_yaw(self) -> None:
        scene = make_scene()
        estimate = estimate_plane_box(scene["mask"], scene["depth"], scene["intrinsics"])

        self.assertIsNotNone(estimate.center_top_camera_m)
        self.assertTrue(estimate.confidence.ok, estimate.failure_reasons)
        error = np.linalg.norm(np.asarray(estimate.center_top_camera_m) - scene["center_3d"])
        self.assertLess(float(error), 0.010, f"center error {error:.4f} m")
        self.assertLess(yaw_error_mod_180(estimate.yaw_mod_180, scene["expected_yaw_image"]), 1.5)
        self.assertEqual(estimate.support["method"], "rim_plane")
        self.assertEqual(estimate.support["trusted_edges"], 4)

    def test_wall_plane_is_not_selected_over_rim(self) -> None:
        scene = make_scene(with_wall=True)
        estimate = estimate_plane_box(scene["mask"], scene["depth"], scene["intrinsics"])

        self.assertIsNotNone(estimate.center_top_camera_m)
        rim_min_dim = estimate.support["plane"]["rim_min_dim_m"]
        self.assertGreater(rim_min_dim, 0.27, "selected plane footprint should span the 0.335 m short side")

    def test_one_cropped_short_side_still_recovers_center(self) -> None:
        scene = make_scene(center_shift_long_m=0.16)
        estimate = estimate_plane_box(scene["mask"], scene["depth"], scene["intrinsics"])

        self.assertIsNotNone(estimate.center_top_camera_m)
        self.assertTrue(estimate.confidence.ok, estimate.failure_reasons)
        error = np.linalg.norm(np.asarray(estimate.center_top_camera_m) - scene["center_3d"])
        self.assertLess(float(error), 0.015, f"center error {error:.4f} m")
        self.assertIn(estimate.support["long_axis"]["mode"], ("lo_edge_only", "hi_edge_only"))

    def test_both_short_sides_cropped_flags_long_axis(self) -> None:
        scene = make_scene(image_width=240, yaw_plane_deg=2.0)
        estimate = estimate_plane_box(scene["mask"], scene["depth"], scene["intrinsics"])

        self.assertIsNotNone(estimate.center_top_camera_m)
        self.assertIn("long_axis_center_underconstrained", estimate.failure_reasons)
        self.assertFalse(estimate.confidence.ok)
        short_error = abs(
            float((np.asarray(estimate.center_top_camera_m) - scene["center_3d"]) @ scene["short_3d"])
        )
        self.assertLess(short_error, 0.015, f"short-axis center error {short_error:.4f} m")
        self.assertLess(yaw_error_mod_180(estimate.yaw_mod_180, scene["expected_yaw_image"]), 2.0)

    def test_discovery_accepts_full_view_and_matches_true_plane(self) -> None:
        scene = make_scene()
        plane = discover_rim_plane(scene["mask"], scene["depth"], scene["intrinsics"])

        self.assertIsNotNone(plane)
        self.assertLess(abs(plane["rim_min_dim_m"] - BOX_SHORT), 0.045)
        normal = np.asarray(plane["normal"])
        true_normal = np.array([0.0, 0.766, 0.643])
        true_normal /= np.linalg.norm(true_normal)
        alignment = abs(float(normal @ true_normal))
        self.assertGreater(alignment, math.cos(math.radians(3.0)))

    def test_calibrated_plane_bypasses_per_frame_discovery(self) -> None:
        full = make_scene()
        plane = discover_rim_plane(full["mask"], full["depth"], full["intrinsics"])
        self.assertIsNotNone(plane)
        calibration = average_rim_planes([plane])

        cropped = make_scene(image_width=240, yaw_plane_deg=2.0)
        estimate = estimate_plane_box(
            cropped["mask"],
            cropped["depth"],
            cropped["intrinsics"],
            rim_plane=(calibration["normal"], calibration["point"]),
        )

        self.assertIsNotNone(estimate.center_top_camera_m)
        self.assertEqual(estimate.support["plane"]["mode"], "fixed")
        short_error = abs(
            float((np.asarray(estimate.center_top_camera_m) - cropped["center_3d"]) @ cropped["short_3d"])
        )
        self.assertLess(short_error, 0.015, f"short-axis center error {short_error:.4f} m")
        self.assertIn("long_axis_center_underconstrained", estimate.failure_reasons)

    def test_empty_mask_fails_cleanly(self) -> None:
        mask = np.zeros((200, 200), dtype=np.uint8)
        depth = np.zeros((200, 200), dtype=np.float32)
        intr = CameraIntrinsics(fx=300.0, fy=300.0, cx=100.0, cy=100.0)

        estimate = estimate_plane_box(mask, depth, intr)

        self.assertIsNone(estimate.center_top_camera_m)
        self.assertFalse(estimate.confidence.ok)
        self.assertIn("insufficient_masked_depth", estimate.failure_reasons)


if __name__ == "__main__":
    unittest.main()
