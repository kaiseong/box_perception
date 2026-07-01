#!/usr/bin/env python3
"""Run the Phase 0 offline pallet-box estimator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from box_pose import estimate_pixel_box, segment_yellow_box
from box_pose.visualization import draw_pixel_estimate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate yellow box pixel yaw from an offline image.")
    parser.add_argument("--image", default="KETI/pallet_box.png", help="Input image path.")
    parser.add_argument("--save-debug", nargs="?", const="KETI/pallet_box_debug.png", help="Save debug overlay image.")
    parser.add_argument("--mask-output", help="Optional path for the binary mask.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Failed to read image: {image_path}")

    mask, stats = segment_yellow_box(image)
    estimate = estimate_pixel_box(mask, stats)

    if args.mask_output:
        cv2.imwrite(str(args.mask_output), mask)
    if args.save_debug:
        overlay = draw_pixel_estimate(image, estimate)
        cv2.imwrite(str(args.save_debug), overlay)

    print(json.dumps(estimate.to_dict(), indent=2, sort_keys=True))
    return 0 if estimate.confidence.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
