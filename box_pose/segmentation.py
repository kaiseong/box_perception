"""HSV segmentation for the yellow/orange pallet box."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MaskStats:
    image_area: int
    total_mask_area: int
    dominant_area: int
    dominant_fraction: float
    significant_components: int

    def to_dict(self) -> dict[str, int | float]:
        return {
            "image_area": self.image_area,
            "total_mask_area": self.total_mask_area,
            "dominant_area": self.dominant_area,
            "dominant_fraction": self.dominant_fraction,
            "significant_components": self.significant_components,
        }


def segment_yellow_box(
    image_bgr: np.ndarray,
    *,
    lower_hsv: tuple[int, int, int] = (5, 70, 70),
    upper_hsv: tuple[int, int, int] = (45, 255, 255),
    morph_kernel_size: int = 7,
    keep_largest_component: bool = True,
) -> tuple[np.ndarray, MaskStats]:
    """Return a cleaned 0/255 mask for yellow/orange box evidence.

    The default keeps the historic dominant-component behavior used by the
    simple OBB estimator. Partial-crop fitting can request all cleaned yellow
    components because separated rims and walls are still useful evidence.
    """

    image = _require_bgr(image_bgr)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    raw = cv2.inRange(hsv, np.array(lower_hsv, dtype=np.uint8), np.array(upper_hsv, dtype=np.uint8))

    kernel_size = max(int(morph_kernel_size), 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    if keep_largest_component:
        return _largest_component(cleaned)
    return np.where(cleaned > 0, 255, 0).astype(np.uint8), _mask_stats(cleaned)


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, MaskStats]:
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    base_stats = _mask_stats(mask_u8)
    total_area = base_stats.total_mask_area
    if total_area == 0:
        empty = np.zeros_like(mask_u8, dtype=np.uint8)
        return empty, base_stats

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if count <= 1:
        empty = np.zeros_like(mask_u8, dtype=np.uint8)
        return empty, base_stats

    areas = stats[1:, cv2.CC_STAT_AREA]
    dominant_index = int(np.argmax(areas)) + 1
    component = np.where(labels == dominant_index, 255, 0).astype(np.uint8)
    return component, base_stats


def _mask_stats(mask: np.ndarray) -> MaskStats:
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    image_area = int(mask_u8.size)
    total_area = int(np.count_nonzero(mask_u8))
    if total_area == 0:
        return MaskStats(image_area, 0, 0, 0.0, 0)

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if count <= 1:
        return MaskStats(image_area, total_area, 0, 0.0, 0)

    areas = stats[1:, cv2.CC_STAT_AREA]
    dominant_area = int(np.max(areas)) if areas.size else 0
    significant_threshold = max(int(round(dominant_area * 0.05)), 1)
    significant_components = int(np.count_nonzero(areas >= significant_threshold))
    dominant_fraction = float(dominant_area / total_area) if total_area else 0.0
    return MaskStats(image_area, total_area, dominant_area, dominant_fraction, significant_components)


def _require_bgr(image_bgr: np.ndarray) -> np.ndarray:
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image_bgr must have shape HxWx3, got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image
