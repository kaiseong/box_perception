"""Debug overlay rendering for box estimates."""

from __future__ import annotations

import cv2
import numpy as np

from .geometry import KnownSizeBoxEstimate, PixelBoxEstimate


def draw_pixel_estimate(image_bgr: np.ndarray, estimate: PixelBoxEstimate) -> np.ndarray:
    image = np.asarray(image_bgr).copy()
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image_bgr must have shape HxWx3, got {image.shape}")
    if not estimate.pixel_obb:
        return image

    corners = np.asarray(estimate.pixel_obb["corners"], dtype=np.int32).reshape((-1, 1, 2))
    center = np.asarray(estimate.pixel_obb["center"], dtype=np.float64)
    long_axis = np.asarray(estimate.long_axis_image, dtype=np.float64)
    grasp_axis = np.asarray(estimate.grasp_axis_image, dtype=np.float64)
    axis_length = min(max(float(estimate.pixel_obb["short_length_px"]) * 0.25, 25.0), 120.0)

    cv2.drawContours(image, [corners], -1, (0, 255, 0), 3)
    cv2.circle(image, tuple(np.rint(center).astype(int)), 5, (255, 255, 255), -1)
    _draw_arrow(image, center, center + long_axis * axis_length, (255, 0, 0), "long")
    _draw_arrow(image, center, center + grasp_axis * axis_length, (0, 0, 255), "grasp")

    text = f"yaw={estimate.yaw_mod_180:.1f} conf={estimate.confidence.ok}"
    cv2.putText(image, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(image, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)
    if estimate.failure_reasons:
        reason = ",".join(estimate.failure_reasons[:3])
        cv2.putText(image, reason, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
    return image


def draw_known_size_estimate(image_bgr: np.ndarray, estimate: KnownSizeBoxEstimate) -> np.ndarray:
    image = np.asarray(image_bgr).copy()
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image_bgr must have shape HxWx3, got {image.shape}")
    if estimate.model_corners is None or estimate.center_image is None:
        return image

    corners = np.asarray(estimate.model_corners, dtype=np.int32).reshape((-1, 1, 2))
    center = np.asarray(estimate.center_image, dtype=np.float64)
    cv2.drawContours(image, [corners], -1, (255, 0, 255), 3)
    cv2.circle(image, tuple(np.rint(center).astype(int)), 6, (0, 255, 255), -1)

    if estimate.long_axis_image is not None and estimate.grasp_axis_image is not None:
        long_axis = np.asarray(estimate.long_axis_image, dtype=np.float64)
        grasp_axis = np.asarray(estimate.grasp_axis_image, dtype=np.float64)
        axis_length = min(max(float(estimate.model_short_length_px or 120.0) * 0.25, 25.0), 130.0)
        _draw_arrow(image, center, center + long_axis * axis_length, (255, 255, 0), "known long")
        _draw_arrow(image, center, center + grasp_axis * axis_length, (255, 0, 255), "known grasp")

    yaw = float("nan") if estimate.yaw_mod_180 is None else float(estimate.yaw_mod_180)
    text = f"known yaw={yaw:.1f} conf={estimate.confidence.ok} score={estimate.confidence.score:.2f}"
    cv2.putText(image, text, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(image, text, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 1, cv2.LINE_AA)
    if estimate.failure_reasons:
        reason = ",".join(estimate.failure_reasons[:3])
        cv2.putText(image, reason, (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 0, 255), 2, cv2.LINE_AA)
    return image


def _draw_arrow(image: np.ndarray, start: np.ndarray, end: np.ndarray, color: tuple[int, int, int], label: str) -> None:
    p0 = tuple(np.rint(start).astype(int))
    p1 = tuple(np.rint(end).astype(int))
    cv2.arrowedLine(image, p0, p1, color, 3, tipLength=0.2)
    cv2.putText(image, label, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
