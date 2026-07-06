#!/usr/bin/env python3
"""Vision-adjusted pre-pick motion for the D405 pallet-box task.

This variant keeps the recorded START_TO_PICKING hand height and hand
orientation, but shifts only x/y from the latest camera-estimated box center.

Default sequence:
  1. ready                           (joint position control)
  2. live vision at ready pose       (D405 + rim-plane estimator, usable center frames)
  3. vision_pre_push                 (START_TO_PICKING z/rotation, vision x/y)
  4. inward y-axis push              (Cartesian impedance control)
  5. lift                            (dual-target Cartesian control)

Pass --pre-push-only to stop at the pose just before the inward y-axis push.

Vision input, one of:
  - default: LIVE capture from the D405 at the pre-push pose (median over
    LIVE_VISION_FRAMES_NEEDED usable center frames; aborts if not enough).
  - --vision-json <path>: latest record of inference.py --print-json output.
  - --box-center-camera X Y Z: manual center_top_camera_m.
All are in the cw90 analysis camera frame (see --view-rotation). They are
transformed to base for vision_pre_push so the x/y correction is
table-horizontal and the START_TO_PICKING base-frame EEF z/rotation stay fixed.
The detected box yaw is printed for reference but NOT commanded in this
stage-1 script.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import rby1_sdk as rby

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

# Time (seconds) the robot takes to reach each joint pose.
MINIMUM_TIME = 2.0

# Link indices into the dynamics state (order MUST match DYN_LINK_NAMES below).
DYN_LINK_NAMES = ["base", "link_torso_5", "ee_right", "ee_left", "link_head_2"]
BASE_INDEX, TORSO_INDEX, EE_RIGHT_INDEX, EE_LEFT_INDEX, HEAD2_INDEX = 0, 1, 2, 3, 4

# ---- Vision pre-push Cartesian positioning parameters ----
# Use base for pre-push so x/y corrections are table-horizontal even when the
# torso is pitched. START_TO_PICKING supplies the target EEF z and rotation.
VISION_PRE_PUSH_REFERENCE_LINK = "base"
VISION_PRE_PUSH_REFERENCE_INDEX = BASE_INDEX
VISION_PRE_PUSH_HAND_Y_MARGIN_M = 0.05

# ---- Cartesian impedance "push inward" parameters ----
# Use the same torso-tip reference as the baseline picking_box.py push. The
# torso posture is fixed, and this push is only along +/- y, so it preserves the
# intended gripper-closing direction without introducing the base-z coupling that
# x-direction torso-frame moves would have.
IMPEDANCE_REFERENCE_LINK = "link_torso_5"
# Actual inward travel from the widened vision pre-push pose.
PUSH_DISTANCE = 0.10
IMPEDANCE_TRANSLATION_WEIGHT = [500.0, 1000.0, 500.0]
IMPEDANCE_ROTATION_WEIGHT = [50.0, 50.0, 50.0]
PUSH_HOLD_TIME = 3.0

# ---- Cartesian lift parameters ----
IMPEDANCE_LIFT_REFERENCE_LINK = "base"
LIFT_HEIGHT = 0.10
LIFT_MINIMUM_TIME = 5.0
LIFT_LINEAR_VELOCITY_LIMIT = 0.1
LIFT_ANGULAR_VELOCITY_LIMIT = float(np.pi / 4)
LIFT_LINEAR_ACCELERATION_LIMIT = 0.5   # m/s^2 (CartesianImpedance add_target)
LIFT_ANGULAR_ACCELERATION_LIMIT = float(np.pi)  # rad/s^2
LIFT_JOINT_STIFFNESS = [250.0] * 7
LIFT_JOINT_DAMPING_RATIO = 1.0
LIFT_JOINT_TORQUE_LIMIT = [50.0] * 7
LIFT_HOLD_TIME = 100.0
# CartesianCommandBuilder.add_target's last argument is a DIMENSIONLESS scaling
# of the internal Cartesian acceleration limits (0.5 = half), NOT m/s^2.
CARTESIAN_ACCELERATION_LIMIT_SCALING = 0.5

# Command wait watchdog. Expected motion/hold time is stage-specific; these
# values add slack around that expectation and prevent command.get() from
# blocking forever when a target is unreachable.
COMMAND_TIMEOUT_MARGIN_SEC = 5.0
COMMAND_TIMEOUT_MIN_SEC = 8.0
COMMAND_WAIT_POLL_SEC = 0.1
COMMAND_WAIT_LOG_INTERVAL_SEC = 2.0
COMMAND_CANCEL_GRACE_SEC = 2.0

# ---- Vision extrinsic: shared single source of truth (camera_extrinsics.py) ----
from camera_extrinsics import (
    compute_camera_to_t5_for_view_rotation as _compute_camera_to_t5_for_view_rotation,
)


def compute_camera_to_t5_for_view_rotation(
    view_rotation: str,
    dyn_model: Any = None,
    dyn_state: Any = None,
    q: Any = None,
) -> np.ndarray:
    """camera(view)->T5, bound to this script's dynamics-state link indices."""
    return _compute_camera_to_t5_for_view_rotation(
        view_rotation, dyn_model, dyn_state, q, torso_index=TORSO_INDEX, head2_index=HEAD2_INDEX
    )


def compute_t5_to_base_transform(
    dyn_model: Any,
    dyn_state: Any,
    q: Any,
) -> np.ndarray:
    """Return the current T5->base transform."""
    dyn_state.set_q(q)
    dyn_model.compute_forward_kinematics(dyn_state)
    return np.asarray(
        dyn_model.compute_transformation(dyn_state, BASE_INDEX, TORSO_INDEX),
        dtype=np.float64,
    )


def compute_camera_to_base_for_view_rotation(
    view_rotation: str,
    dyn_model: Any,
    dyn_state: Any,
    q: Any,
) -> np.ndarray:
    """Return camera(view)->base for the current robot state."""
    camera_to_t5 = compute_camera_to_t5_for_view_rotation(
        view_rotation,
        dyn_model,
        dyn_state,
        q,
    )
    return compute_t5_to_base_transform(dyn_model, dyn_state, q) @ camera_to_t5


def transform_camera_point_to_base(
    point_camera_m: Any,
    camera_to_base: Any,
) -> np.ndarray:
    """Transform one 3D camera-frame point into the base/table-horizontal frame."""
    T = np.asarray(camera_to_base, dtype=np.float64)
    p_camera = np.ones(4, dtype=np.float64)
    p_camera[:3] = np.asarray(point_camera_m, dtype=np.float64)
    return (T @ p_camera)[:3]


def build_dual_target_cartesian_command(
    *,
    reference_link: str,
    right_target: np.ndarray,
    left_target: np.ndarray,
    minimum_time: float,
    hold_time: float,
) -> Any:
    """Build one synchronized arm-only command with independent EEF targets.

    Rotation, crop offset, or asymmetric hand placement should be handled by
    computing different `right_target` / `left_target` transforms. They are sent
    together in one RobotCommand, but as right/left arm-component commands so
    torso and head are not used to satisfy the Cartesian targets.
    """
    def arm_cartesian_command(link_name: str, target: np.ndarray) -> Any:
        command = (
            rby.CartesianCommandBuilder()
            .add_target(
                reference_link,
                link_name,
                target,
                LIFT_LINEAR_VELOCITY_LIMIT,
                LIFT_ANGULAR_VELOCITY_LIMIT,
                CARTESIAN_ACCELERATION_LIMIT_SCALING,
            )
            .set_minimum_time(minimum_time)
        )
        if hold_time > 0.0:
            command = command.set_command_header(
                rby.CommandHeaderBuilder().set_control_hold_time(hold_time)
            )
        return command

    return rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_body_command(
            rby.BodyComponentBasedCommandBuilder()
            .set_right_arm_command(arm_cartesian_command("ee_right", right_target))
            .set_left_arm_command(arm_cartesian_command("ee_left", left_target))
        )
    )


VISION_APPROACH_MINIMUM_TIME = 1.0
VISION_APPROACH_HOLD_TIME = 1.0
VISION_APPROACH_MAX_REFERENCE_XY_SHIFT_M = 0.25

# ---- Live vision (default input source) ----
# Rim-plane estimator stack (box_pose/, replay_recording.py) and the stored
# plane calibration live in THIS repository, so box_codex runs standalone.
BOX_PERCEPTION_ROOT = Path(__file__).resolve().parent
RIM_PLANE_CONFIG = BOX_PERCEPTION_ROOT / "config_2" / "rim_plane.json"
CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS = 1280, 720, 30
# The box is static, so a few confident frames (median) reject one-off noise.
LIVE_VISION_FRAMES_NEEDED = 5
LIVE_VISION_TIMEOUT_SEC = 10.0
LIVE_VISION_MAX_CENTER_SPREAD_M = 0.025
LIVE_VISION_CENTER_ONLY_ALLOWED_REASONS = {
    "long_axis_center_underconstrained",
    "yaw_from_extent_fallback",
}

# ========================================================================================
# Local recorded joint sets [rad] for picking_box_2.
# Keep these independent from picking_box.py so vision-adjusted picking can evolve without
# silently inheriting posture edits from the baseline script.
# ========================================================================================

READY = {
    "torso": [0.000, 0.000, 0.000, 0.349, 0.000, 0.000],
    "right_arm": [0.0, -0.96, -0.524, -1.221, -2.443, 1.134, 0.0],
    "left_arm": [-0.0, 0.96, 0.524, -1.221, 2.443, 1.134, 0.0],
    "head": [0.000, 0.436],
}

START_TO_PICKING = {
    "torso": [0.000, 0.000, 0.000, 0.349, 0.000, 0.000],
    "right_arm": [-0.171, -0.841, -0.153, -1.511, -2.403, 1.743, 0.3],
    "left_arm": [-0.171, 0.841, 0.153, -1.511, 2.403, 1.743, -0.3],
    "head": [0.000, 0.436],
}

# Joint-space stages executed before live vision and Cartesian motion.
# START_TO_PICKING is not executed here; it only supplies the reference hand
# z/rotation for the later vision_pre_push target.
JOINT_SEQUENCE = [
    ("ready", READY),
]


def latest_json_record_from_path(path: str | Path) -> dict[str, Any]:
    """Load the last JSON object from a JSON or JSONL inference output file."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"vision JSON file is empty: {path}")

    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        records: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                records.append(item)
        if not records:
            raise ValueError(f"no JSON object found in vision output: {path}")
        return records[-1]

    if isinstance(loaded, list):
        for item in reversed(loaded):
            if isinstance(item, dict):
                return item
        raise ValueError(f"JSON list has no object records: {path}")
    if not isinstance(loaded, dict):
        raise ValueError(f"expected JSON object/list in vision output: {path}")
    return loaded


def box_center_camera_from_record(record: dict[str, Any], *, allow_unreliable: bool) -> np.ndarray:
    """Extract center_top_camera_m from inference.py output."""
    # inference_2.py --print-json emits {"reliable": ..., "reasons": [...]};
    # replay frames.jsonl nests it as known_size.confidence.ok. Support both.
    reliable = record.get("reliable", record.get("ok"))
    if reliable is None and isinstance(record.get("known_size"), dict):
        reliable = record["known_size"].get("confidence", {}).get("ok")
    if not allow_unreliable and reliable is not True:
        reasons = (
            record.get("reasons")
            or record.get("failure_reasons")
            or record.get("plane_failure_reasons")
            or []
        )
        raise ValueError(f"vision record is not reliable; refusing to move. reasons={reasons}")

    center = record.get("center_top_camera_m")
    if center is None and isinstance(record.get("known_size"), dict):
        center = record["known_size"].get("center_top_camera_m")
    if center is None:
        raise ValueError("vision record does not contain center_top_camera_m")

    point = np.asarray(center, dtype=np.float64).reshape(-1)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError(f"invalid center_top_camera_m: {center!r}")
    return point


def live_estimate_center_mode(estimate: Any) -> tuple[bool, str]:
    """Return whether an estimate is usable for center-only pre-push motion."""
    if estimate.center_top_camera_m is None:
        return False, "no_center"
    if estimate.confidence.ok:
        return True, "confidence_ok"
    reasons = set(estimate.failure_reasons)
    if reasons and reasons.issubset(LIVE_VISION_CENTER_ONLY_ALLOWED_REASONS):
        return True, "center_only:" + ",".join(sorted(reasons))
    return False, ",".join(estimate.failure_reasons[:2]) or "low_confidence"


def capture_box_live(
    view_rotation: str,
    *,
    frames_needed: int = LIVE_VISION_FRAMES_NEEDED,
    timeout_sec: float = LIVE_VISION_TIMEOUT_SEC,
    max_center_spread_m: float = LIVE_VISION_MAX_CENTER_SPREAD_M,
    visualize: bool = False,
    hold_after_capture: bool = False,
    camera_to_base: Any = None,
    run_forever: bool = False,
) -> dict[str, Any] | None:
    """Detect the box from the live D405 at the current posture.

    Runs the rim-plane estimator with the stored plane calibration and keeps
    confidence-ok frames are preferred. For this stage-1 motion, yaw is not
    commanded, so frames whose only issue is an underconstrained long-axis
    center are also accepted as center-only candidates and checked by a
    multi-frame center-spread gate. Returns the per-axis median center
    (analysis camera frame), optional long-axis direction, and frame count, or
    None when not enough usable frames arrive within the timeout.

    With visualize=True an OpenCV window shows each frame with the fitted box
    and, when camera_to_base is given, the base-frame x/y/yaw the robot would
    act on -- exactly what the grasp-reach experiments need. If
    hold_after_capture=True, the live window stays up after enough usable frames
    are captured until c/space/enter continues or q/ESC aborts. run_forever=True
    streams until q/ESC without collecting a result (observation mode).
    """
    import time

    import cv2

    cv2.setNumThreads(1)

    if str(BOX_PERCEPTION_ROOT) not in sys.path:
        sys.path.insert(0, str(BOX_PERCEPTION_ROOT))
    from box_pose import estimate_plane_box, segment_yellow_box
    from box_pose.visualization import draw_known_size_estimate
    from inference_2 import iter_live_frames

    rim_cfg = json.loads(RIM_PLANE_CONFIG.read_text(encoding="utf-8"))
    rim_plane = (rim_cfg["normal"], rim_cfg["point"])

    T_base = None if camera_to_base is None else np.asarray(camera_to_base, dtype=np.float64)
    window_name = "picking_box_2 vision (q/ESC to close)"
    centers: list[Any] = []
    long_axes: list[Any] = []
    modes: list[str] = []
    quit_requested = False
    continue_requested = False
    long_axis_unconstrained = False
    deadline = time.monotonic() + (float("inf") if run_forever else float(timeout_sec))
    frames = iter_live_frames(
        width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS, view_rotation=view_rotation
    )
    try:
        for _, image, depth_m, intrinsics in frames:
            captured_enough = not run_forever and len(centers) >= int(frames_needed)
            if captured_enough and not (visualize and hold_after_capture):
                break
            if time.monotonic() >= deadline and not (captured_enough and visualize and hold_after_capture):
                break
            mask, _ = segment_yellow_box(image, keep_largest_component=False)
            estimate = estimate_plane_box(mask, depth_m, intrinsics, rim_plane=rim_plane)
            usable, mode = live_estimate_center_mode(estimate)
            collecting = not run_forever and len(centers) < int(frames_needed)
            if collecting:
                if usable:
                    centers.append(estimate.center_top_camera_m)
                    modes.append(mode)
                    axis = estimate.support.get("long_axis_camera")
                    if axis is not None:
                        long_axes.append(axis)
                    if not estimate.confidence.ok:
                        # The long-axis center of this frame is a visible-span
                        # midpoint, not a measurement; remember to drop that
                        # axis from the commanded correction.
                        if "long_axis_center_underconstrained" in estimate.failure_reasons:
                            long_axis_unconstrained = True
                        if len(centers) == 1 or len(centers) % 3 == 0:
                            print(f"[vision] accepted center-only frame: {mode}")
                else:
                    print(f"[vision] frame rejected: {mode}")

            if visualize:
                vis = draw_known_size_estimate(image.copy(), estimate)
                status_color = (30, 160, 30) if usable else (40, 40, 220)
                lines = [(f"{'USABLE' if usable else 'REJECTED'}: {mode}", status_color)]
                if not run_forever and len(centers) >= int(frames_needed):
                    lines.append((
                        "CAPTURED: press c/space/enter to continue, q/ESC to abort",
                        (0, 220, 255),
                    ))
                if T_base is not None and estimate.center_top_camera_m is not None:
                    center_base = (T_base @ np.append(estimate.center_top_camera_m, 1.0))[:3]
                    text = f"base x={center_base[0] * 100:+.1f}cm  y={center_base[1] * 100:+.1f}cm"
                    long_axis = estimate.support.get("long_axis_camera")
                    if long_axis is not None:
                        axis_base = T_base[:3, :3] @ np.asarray(long_axis, dtype=np.float64)
                        yaw_base = float(np.degrees(np.arctan2(axis_base[1], axis_base[0])) % 180.0)
                        text += f"  yaw={yaw_base:.1f}deg"
                    lines.append((text, (255, 255, 255)))
                for i, (text, color) in enumerate(lines):
                    y_pos = 34 + 32 * i
                    cv2.putText(vis, text, (12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(vis, text, (12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
                cv2.imshow(window_name, vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    quit_requested = True
                    break
                if len(centers) >= int(frames_needed) and key in (ord("c"), ord(" "), 10, 13):
                    continue_requested = True
                    break
    finally:
        frames.close()  # stops the RealSense pipeline (generator finally)
        if visualize:
            cv2.destroyAllWindows()

    if run_forever or quit_requested:
        return None
    if visualize and hold_after_capture and not continue_requested and len(centers) >= int(frames_needed):
        return None

    if len(centers) < int(frames_needed):
        return None
    center_array = np.asarray(centers, dtype=np.float64)
    center_median = np.median(center_array, axis=0)
    center_spread_m = float(np.max(np.linalg.norm(center_array - center_median, axis=1)))
    if center_spread_m > float(max_center_spread_m):
        print(
            "[vision] FAILED: live center spread too large "
            f"({center_spread_m:.3f} m > {max_center_spread_m:.3f} m)."
        )
        return None

    return {
        "center_camera_m": center_median,
        "center_spread_m": center_spread_m,
        "long_axis_camera": None if not long_axes else np.median(np.asarray(long_axes, dtype=np.float64), axis=0),
        "long_axis_unconstrained": long_axis_unconstrained,
        "frames_used": len(centers),
        "modes": modes,
    }


def resolve_box_center_camera(args: argparse.Namespace) -> np.ndarray | None:
    """Return the box center from CLI inputs, or None to capture live later."""
    sources = [args.box_center_camera is not None, args.vision_json is not None]
    if sum(sources) == 0:
        return None  # live capture at the pre-push pose
    if sum(sources) != 1:
        raise ValueError("provide at most one of --box-center-camera or --vision-json")
    if args.box_center_camera is not None:
        point = np.asarray(args.box_center_camera, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(point)):
            raise ValueError("--box-center-camera must be finite x y z values")
        return point

    record = latest_json_record_from_path(args.vision_json)
    return box_center_camera_from_record(record, allow_unreliable=args.allow_unreliable_vision)


def q_for_recorded_pose(robot_model: Any, q_template: np.ndarray, pose: dict[str, list[float]]) -> np.ndarray:
    """Return a full robot q vector with the recorded pose inserted."""
    q = np.asarray(q_template, dtype=np.float64).copy()
    q[robot_model.torso_idx] = np.asarray(pose["torso"], dtype=np.float64)
    q[robot_model.right_arm_idx] = np.asarray(pose["right_arm"], dtype=np.float64)
    q[robot_model.left_arm_idx] = np.asarray(pose["left_arm"], dtype=np.float64)
    q[robot_model.head_idx] = np.asarray(pose["head"], dtype=np.float64)
    return q


def start_to_picking_reference_targets(
    dyn_model: Any,
    dyn_state: Any,
    robot_model: Any,
    q_template: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """FK of recorded START_TO_PICKING EEF poses in the pre-push reference frame."""
    q_start = q_for_recorded_pose(robot_model, q_template, START_TO_PICKING)
    dyn_state.set_q(q_start)
    dyn_model.compute_forward_kinematics(dyn_state)
    T_ref_right = np.asarray(
        dyn_model.compute_transformation(
            dyn_state,
            VISION_PRE_PUSH_REFERENCE_INDEX,
            EE_RIGHT_INDEX,
        ),
        dtype=np.float64,
    ).copy()
    T_ref_left = np.asarray(
        dyn_model.compute_transformation(
            dyn_state,
            VISION_PRE_PUSH_REFERENCE_INDEX,
            EE_LEFT_INDEX,
        ),
        dtype=np.float64,
    ).copy()
    return T_ref_right, T_ref_left


def vision_pre_push_targets(
    dyn_model: Any,
    dyn_state: Any,
    robot_model: Any,
    q_template: np.ndarray,
    box_center_base_m: np.ndarray,
    *,
    midpoint_offset_xy_m: tuple[float, float] = (0.0, 0.0),
    drop_delta_axis_base_xy: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Shift START_TO_PICKING hand targets so their midpoint matches the box center.

    z and rotation are copied from the recorded START_TO_PICKING FK in base.
    The midpoint receives the vision x/y delta, then each hand is opened farther
    along base y to give the box extra pre-grasp clearance.

    When the live capture flags the box long axis as underconstrained (both
    short sides cropped), the detected center along that axis is a visible-span
    midpoint, not a measurement; `drop_delta_axis_base_xy` names that direction
    and its component is removed from the correction instead of being trusted.
    """
    T_right_ref, T_left_ref = start_to_picking_reference_targets(
        dyn_model,
        dyn_state,
        robot_model,
        q_template,
    )
    reference_midpoint_xy = 0.5 * (T_right_ref[:2, 3] + T_left_ref[:2, 3])
    target_midpoint_xy = np.asarray(box_center_base_m[:2], dtype=np.float64) + np.asarray(
        midpoint_offset_xy_m,
        dtype=np.float64,
    )
    xy_delta = target_midpoint_xy - reference_midpoint_xy
    dropped_delta_m = 0.0
    if drop_delta_axis_base_xy is not None:
        axis = np.asarray(drop_delta_axis_base_xy, dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if norm > 1e-9:
            axis = axis / norm
            dropped_delta_m = float(xy_delta @ axis)
            xy_delta = xy_delta - dropped_delta_m * axis
            target_midpoint_xy = reference_midpoint_xy + xy_delta

    T_right_target = T_right_ref.copy()
    T_left_target = T_left_ref.copy()
    T_right_target[:2, 3] += xy_delta
    T_left_target[:2, 3] += xy_delta
    right_y_direction = np.sign(T_right_ref[1, 3] - reference_midpoint_xy[1])
    left_y_direction = np.sign(T_left_ref[1, 3] - reference_midpoint_xy[1])
    if right_y_direction == 0.0:
        right_y_direction = -1.0
    if left_y_direction == 0.0:
        left_y_direction = 1.0
    T_right_target[1, 3] += right_y_direction * VISION_PRE_PUSH_HAND_Y_MARGIN_M
    T_left_target[1, 3] += left_y_direction * VISION_PRE_PUSH_HAND_Y_MARGIN_M

    return {
        "right_target": T_right_target,
        "left_target": T_left_target,
        "reference_midpoint_xy": reference_midpoint_xy,
        "target_midpoint_xy": target_midpoint_xy,
        "xy_delta": xy_delta,
        "dropped_delta_m": dropped_delta_m,
        "hand_y_margin_m": VISION_PRE_PUSH_HAND_Y_MARGIN_M,
    }


def rotation_error_deg(current: np.ndarray, target: np.ndarray) -> float:
    """Return angular difference from current orientation to target orientation."""
    R_delta = np.asarray(target[:3, :3], dtype=np.float64) @ np.asarray(
        current[:3, :3], dtype=np.float64
    ).T
    cos_angle = float(np.clip((np.trace(R_delta) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def print_current_to_target_delta(
    *,
    label: str,
    dyn_model: Any,
    dyn_state: Any,
    q: Any,
    ref_index: int,
    right_target: np.ndarray,
    left_target: np.ndarray,
) -> None:
    """Print current EEF pose, target pose, and delta before sending a command."""
    dyn_state.set_q(q)
    dyn_model.compute_forward_kinematics(dyn_state)
    current_right = np.asarray(
        dyn_model.compute_transformation(dyn_state, ref_index, EE_RIGHT_INDEX),
        dtype=np.float64,
    )
    current_left = np.asarray(
        dyn_model.compute_transformation(dyn_state, ref_index, EE_LEFT_INDEX),
        dtype=np.float64,
    )

    for arm_name, current, target in (
        ("right", current_right, right_target),
        ("left", current_left, left_target),
    ):
        delta_xyz = np.asarray(target[:3, 3] - current[:3, 3], dtype=np.float64)
        print(
            f"[{label}] {arm_name} current xyz: {current[:3, 3]} -> "
            f"target xyz: {target[:3, 3]} | delta={delta_xyz} "
            f"| |delta|={np.linalg.norm(delta_xyz):.3f} m | "
            f"rotation_delta={rotation_error_deg(current, target):.1f} deg",
            flush=True,
        )


def build_vision_pre_push_command(
    dyn_model: Any,
    dyn_state: Any,
    robot_model: Any,
    q_template: np.ndarray,
    box_center_base_m: np.ndarray,
    *,
    approach_time: float,
    hold_time: float,
    max_reference_xy_shift_m: float | None,
    midpoint_offset_xy_m: tuple[float, float],
    drop_delta_axis_base_xy: np.ndarray | None = None,
) -> Any:
    targets = vision_pre_push_targets(
        dyn_model,
        dyn_state,
        robot_model,
        q_template,
        box_center_base_m,
        midpoint_offset_xy_m=midpoint_offset_xy_m,
        drop_delta_axis_base_xy=drop_delta_axis_base_xy,
    )
    if targets["dropped_delta_m"]:
        print(
            "[vision_pre_push] long axis underconstrained: dropped "
            f"{targets['dropped_delta_m']:+.3f} m of correction along it "
            "(that direction keeps the recorded pose)."
        )

    xy_delta = targets["xy_delta"]
    xy_shift = float(np.linalg.norm(xy_delta))
    if max_reference_xy_shift_m is not None and xy_shift > max_reference_xy_shift_m:
        raise ValueError(
            "vision x/y shift is too large: "
            f"{xy_shift:.3f} m > {max_reference_xy_shift_m:.3f} m. "
            "Re-run with --allow-large-vision-shift or a larger --max-reference-xy-shift-m "
            "only after checking the vision estimate."
        )

    print("[vision_pre_push] reference frame:", VISION_PRE_PUSH_REFERENCE_LINK)
    print("[vision_pre_push] box center in base:", np.asarray(box_center_base_m, dtype=np.float64))
    print(
        "[vision_pre_push] hand midpoint xy "
        f"{targets['reference_midpoint_xy']} -> {targets['target_midpoint_xy']} "
        f"(delta={xy_delta}, |delta|={xy_shift:.3f} m)"
    )
    print(
        "[vision_pre_push] right target xyz:",
        targets["right_target"][:3, 3],
        "| left target xyz:",
        targets["left_target"][:3, 3],
    )
    print(
        "[vision_pre_push] extra y clearance:",
        f"{targets['hand_y_margin_m']:.3f} m per hand",
    )
    print_current_to_target_delta(
        label="vision_pre_push",
        dyn_model=dyn_model,
        dyn_state=dyn_state,
        q=q_template,
        ref_index=VISION_PRE_PUSH_REFERENCE_INDEX,
        right_target=targets["right_target"],
        left_target=targets["left_target"],
    )
    print(
        "[vision_pre_push] base-frame z and rotation are inherited from "
        "recorded START_TO_PICKING; base x/y midpoint is shifted and y spacing is widened."
    )

    return build_dual_target_cartesian_command(
        reference_link=VISION_PRE_PUSH_REFERENCE_LINK,
        right_target=targets["right_target"],
        left_target=targets["left_target"],
        minimum_time=approach_time,
        hold_time=hold_time,
    )


def build_pose_command(pose: dict[str, list[float]], minimum_time: float) -> Any:
    """Build a one-shot joint position command for torso, both arms, and head."""
    return rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder()
        .set_body_command(
            rby.BodyComponentBasedCommandBuilder()
            .set_torso_command(
                rby.JointPositionCommandBuilder()
                .set_minimum_time(minimum_time)
                .set_position(pose["torso"])
            )
            .set_right_arm_command(
                rby.JointPositionCommandBuilder()
                .set_minimum_time(minimum_time)
                .set_position(pose["right_arm"])
            )
            .set_left_arm_command(
                rby.JointPositionCommandBuilder()
                .set_minimum_time(minimum_time)
                .set_position(pose["left_arm"])
            )
        )
        .set_head_command(
            rby.HeadCommandBuilder(
                rby.JointPositionCommandBuilder()
                .set_minimum_time(minimum_time)
                .set_position(pose["head"])
            )
        )
    )


def print_stage(stage: str, message: str) -> None:
    """Print the currently active picking stage immediately."""
    print(f"[stage] {stage} | {message}", flush=True)


def stage_timeout_sec(
    *expected_durations_sec: float,
    min_timeout_sec: float = COMMAND_TIMEOUT_MIN_SEC,
    margin_sec: float = COMMAND_TIMEOUT_MARGIN_SEC,
) -> float:
    """Return watchdog timeout for a stage with known minimum/hold durations."""
    expected = sum(max(0.0, float(value)) for value in expected_durations_sec)
    return max(float(min_timeout_sec), expected + float(margin_sec))


def wait_for_command_feedback(
    command: Any,
    stage: str,
    *,
    timeout_sec: float,
    poll_sec: float = COMMAND_WAIT_POLL_SEC,
    log_interval_sec: float = COMMAND_WAIT_LOG_INTERVAL_SEC,
) -> Any | None:
    """Poll an RBY1 command handler until feedback is ready or timeout expires."""
    deadline = time.monotonic() + float(timeout_sec)
    next_log = time.monotonic() + float(log_interval_sec)
    while True:
        now = time.monotonic()
        remaining = deadline - now
        if remaining <= 0.0:
            return None

        wait_ms = max(1, int(min(float(poll_sec), remaining) * 1000.0))
        if command.wait_for(wait_ms):
            return command.get()

        now = time.monotonic()
        if now >= next_log:
            elapsed = max(0.0, float(timeout_sec) - max(0.0, deadline - now))
            print_stage(stage, f"still waiting ({elapsed:.1f}/{float(timeout_sec):.1f}s)")
            next_log = now + float(log_interval_sec)


def cancel_timed_out_command(robot: Any, command: Any, stage: str) -> None:
    """Best-effort cancellation after a stage watchdog timeout."""
    print_stage(stage, "TIMEOUT; canceling command")
    try:
        command.cancel()
        print_stage(stage, "command.cancel() requested")
    except Exception as exc:  # pragma: no cover - SDK/hardware dependent
        print_stage(stage, f"command.cancel() failed: {exc}")

    try:
        robot.cancel_control()
        print_stage(stage, "robot.cancel_control() requested")
    except Exception as exc:  # pragma: no cover - SDK/hardware dependent
        print_stage(stage, f"robot.cancel_control() failed: {exc}")

    try:
        if command.wait_for(int(COMMAND_CANCEL_GRACE_SEC * 1000.0)):
            feedback = command.get()
            print_stage(stage, f"finish_code_after_cancel={feedback.finish_code}")
        else:
            print_stage(stage, "cancel feedback not received within grace window")
    except Exception as exc:  # pragma: no cover - SDK/hardware dependent
        print_stage(stage, f"failed while waiting for cancel feedback: {exc}")


def send_stage(robot: Any, builder: Any, stage: str, *, timeout_sec: float) -> bool:
    """Send a command and leave a clear log while waiting for completion."""
    print_stage(stage, "sending command")
    command = robot.send_command(builder)
    print_stage(stage, f"waiting for finish_code (timeout={float(timeout_sec):.1f}s)")
    feedback = wait_for_command_feedback(command, stage, timeout_sec=float(timeout_sec))
    if feedback is None:
        cancel_timed_out_command(robot, command, stage)
        return False
    print_stage(stage, f"finish_code={feedback.finish_code}")
    return feedback.finish_code == rby.RobotCommandFeedback.FinishCode.Ok


def offset_translation(T: Any, dy: float = 0.0, dz: float = 0.0) -> np.ndarray:
    """Return a copy of T translated in the reference frame's y/z axes."""
    T_offset = np.asarray(T, dtype=np.float64).copy()
    T_offset[1, 3] += dy
    T_offset[2, 3] += dz
    return T_offset


def build_dual_arm_impedance_command(
    dyn_model: Any,
    dyn_state: Any,
    q: Any,
    reference_link: str,
    ref_index: int,
    inward: float,
    lift: float,
    translation_weight: list[float],
    hold_time: float,
    label: str,
) -> Any:
    """Build a dual-arm Cartesian impedance command from current EEF poses."""
    dyn_state.set_q(q)
    dyn_model.compute_forward_kinematics(dyn_state)
    T_ref2right = dyn_model.compute_transformation(dyn_state, ref_index, EE_RIGHT_INDEX)
    T_ref2left = dyn_model.compute_transformation(dyn_state, ref_index, EE_LEFT_INDEX)

    T_right_target = offset_translation(T_ref2right, +inward, lift)
    T_left_target = offset_translation(T_ref2left, -inward, lift)

    print(
        f"[{label}] ref='{reference_link}' right EEF (y,z): "
        f"({T_ref2right[1, 3]:+.3f},{T_ref2right[2, 3]:+.3f})"
        f" -> ({T_right_target[1, 3]:+.3f},{T_right_target[2, 3]:+.3f}) m"
    )
    print(
        f"[{label}] ref='{reference_link}' left  EEF (y,z): "
        f"({T_ref2left[1, 3]:+.3f},{T_ref2left[2, 3]:+.3f})"
        f" -> ({T_left_target[1, 3]:+.3f},{T_left_target[2, 3]:+.3f}) m"
    )

    def arm_impedance(link_name: str, T_target: np.ndarray) -> Any:
        return (
            rby.ImpedanceControlCommandBuilder()
            .set_command_header(
                rby.CommandHeaderBuilder().set_control_hold_time(hold_time)
            )
            .set_reference_link_name(reference_link)
            .set_link_name(link_name)
            .set_translation_weight(translation_weight)
            .set_rotation_weight(IMPEDANCE_ROTATION_WEIGHT)
            .set_transformation(T_target)
        )

    return rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_body_command(
            rby.BodyComponentBasedCommandBuilder()
            .set_right_arm_command(arm_impedance("ee_right", T_right_target))
            .set_left_arm_command(arm_impedance("ee_left", T_left_target))
        )
    )


def build_impedance_push_command(dyn_model: Any, dyn_state: Any, q: Any) -> Any:
    """Push both hands inward to grip the box, expressed in link_torso_5.

    The pre-push pose already opened each hand wider for approach clearance.
    PUSH_DISTANCE is the total inward travel from that widened pose.
    """
    return build_dual_arm_impedance_command(
        dyn_model,
        dyn_state,
        q,
        reference_link=IMPEDANCE_REFERENCE_LINK,
        ref_index=TORSO_INDEX,
        inward=PUSH_DISTANCE,
        lift=0.0,
        translation_weight=IMPEDANCE_TRANSLATION_WEIGHT,
        hold_time=PUSH_HOLD_TIME,
        label="push",
    )


def build_impedance_lift_command(dyn_model: Any, dyn_state: Any, q: Any) -> Any:
    """Raise both hands straight up by LIFT_HEIGHT along base +z, compliantly.

    Uses per-arm CartesianImpedanceControlCommand (timed trajectory + joint
    stiffness/damping/torque limit) so the box keeps being HELD while it rises:
    a pure position-control lift maintains no inward preload once the push
    command's hold expires, and the box can slip.
    """
    dyn_state.set_q(q)
    dyn_model.compute_forward_kinematics(dyn_state)
    T_base2right = dyn_model.compute_transformation(dyn_state, BASE_INDEX, EE_RIGHT_INDEX)
    T_base2left = dyn_model.compute_transformation(dyn_state, BASE_INDEX, EE_LEFT_INDEX)

    T_right_target = offset_translation(T_base2right, 0.0, +LIFT_HEIGHT)
    T_left_target = offset_translation(T_base2left, 0.0, +LIFT_HEIGHT)

    print(
        f"[lift] right EEF z: {T_base2right[2, 3]:+.3f} -> {T_right_target[2, 3]:+.3f} m"
        f"  |  left EEF z: {T_base2left[2, 3]:+.3f} -> {T_left_target[2, 3]:+.3f} m"
        f"  (over {LIFT_MINIMUM_TIME:.0f}s, Cartesian impedance)"
    )

    def arm_cartesian_impedance(link_name: str, T_target: np.ndarray) -> Any:
        return (
            rby.CartesianImpedanceControlCommandBuilder()
            .set_command_header(
                rby.CommandHeaderBuilder().set_control_hold_time(LIFT_HOLD_TIME)
            )
            .add_target(
                IMPEDANCE_LIFT_REFERENCE_LINK, link_name, T_target,
                LIFT_LINEAR_VELOCITY_LIMIT, LIFT_ANGULAR_VELOCITY_LIMIT,
                LIFT_LINEAR_ACCELERATION_LIMIT, LIFT_ANGULAR_ACCELERATION_LIMIT,
            )
            .set_joint_stiffness(LIFT_JOINT_STIFFNESS)
            .set_joint_damping_ratio(LIFT_JOINT_DAMPING_RATIO)
            .set_joint_torque_limit(LIFT_JOINT_TORQUE_LIMIT)
            .set_minimum_time(LIFT_MINIMUM_TIME)
        )

    return rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_body_command(
            rby.BodyComponentBasedCommandBuilder()
            .set_right_arm_command(arm_cartesian_impedance("ee_right", T_right_target))
            .set_left_arm_command(arm_cartesian_impedance("ee_left", T_left_target))
        )
    )


def main(
    *,
    address: str,
    model: str,
    power: str,
    box_center_camera_m: np.ndarray | None,
    view_rotation: str,
    approach_time: float,
    hold_time: float,
    max_reference_xy_shift_m: float | None,
    midpoint_offset_xy_m: tuple[float, float],
    live_vision_frames_needed: int,
    live_vision_timeout_sec: float,
    live_center_spread_m: float,
    continue_pick: bool,
    visualize: bool,
    visualize_only: bool,
    command_timeout_margin_sec: float,
    min_command_timeout_sec: float,
) -> bool:
    robot = rby.create_robot(address, model)
    robot.connect()
    if not robot.is_connected():
        print("Robot is not connected")
        return False

    if not robot.is_power_on(power):
        if not robot.power_on(power):
            print("Failed to power on")
            return False
    if not robot.is_servo_on(".*"):
        if not robot.servo_on(".*"):
            print("Failed to servo on")
            return False
    robot.reset_fault_control_manager()
    if not robot.enable_control_manager():
        print("Failed to enable control manager")
        return False

    robot_model = robot.model()
    dyn_model = robot.get_dynamics()
    dyn_state = dyn_model.make_state(DYN_LINK_NAMES, robot_model.robot_joint_names)

    done = False
    try:
        if visualize_only:
            # Observation mode: no motion commands. Stream the live estimate
            # with base-frame x/y/yaw at the CURRENT posture until q/ESC.
            q = robot.get_state().position
            camera_to_base = compute_camera_to_base_for_view_rotation(
                view_rotation, dyn_model, dyn_state, q
            )
            print_stage("visualize_only", "streaming; robot will NOT move (quit with q/ESC)")
            capture_box_live(
                view_rotation,
                visualize=True,
                camera_to_base=camera_to_base,
                run_forever=True,
            )
            return True

        for name, pose in JOINT_SEQUENCE:
            stage = f"1/5 {name}"
            print_stage(stage, "joint position move")
            if not send_stage(
                robot,
                build_pose_command(pose, MINIMUM_TIME),
                stage,
                timeout_sec=stage_timeout_sec(
                    MINIMUM_TIME,
                    min_timeout_sec=min_command_timeout_sec,
                    margin_sec=command_timeout_margin_sec,
                ),
            ):
                print_stage(stage, "FAILED; aborting")
                return done
            print_stage(stage, "reached")

        q = robot.get_state().position
        camera_to_base = compute_camera_to_base_for_view_rotation(
            view_rotation,
            dyn_model,
            dyn_state,
            q,
        )

        long_axis_camera = None
        drop_delta_axis_base_xy = None
        if box_center_camera_m is None:
            # Live capture AT this pose, so the FK-based camera->base above and
            # the measurement share the exact same posture.
            print_stage("2/5 live_vision", "capturing D405 frames")
            live = capture_box_live(
                view_rotation,
                frames_needed=live_vision_frames_needed,
                timeout_sec=live_vision_timeout_sec,
                max_center_spread_m=live_center_spread_m,
                visualize=visualize,
                hold_after_capture=visualize,
                camera_to_base=camera_to_base,
            )
            if live is None:
                print_stage(
                    "2/5 live_vision",
                    "FAILED: not enough usable center frames within "
                    f"{live_vision_timeout_sec:.0f}s; aborting before contact",
                )
                return done
            box_center_camera_m = live["center_camera_m"]
            long_axis_camera = live["long_axis_camera"]
            print_stage(
                "2/5 live_vision",
                f"done; usable frames={live['frames_used']} "
                f"| center spread={live['center_spread_m'] * 1000.0:.1f} mm "
                f"| modes={sorted(set(live['modes']))}",
            )
            if live["long_axis_unconstrained"]:
                if long_axis_camera is None:
                    print_stage(
                        "2/5 live_vision",
                        "FAILED: long axis underconstrained and no axis direction; aborting",
                    )
                    return done
                axis_base = np.asarray(camera_to_base, dtype=np.float64)[:3, :3] @ np.asarray(
                    long_axis_camera, dtype=np.float64
                )
                drop_delta_axis_base_xy = axis_base[:2]
                print_stage(
                    "2/5 live_vision",
                    "long axis underconstrained (both short sides cropped); its "
                    "correction component will be dropped",
                )
        else:
            print_stage("2/5 live_vision", "skipped; using supplied center_top_camera_m")

        box_center_base_m = transform_camera_point_to_base(
            box_center_camera_m,
            camera_to_base,
        )
        print(f"[vision] view_rotation={view_rotation}")
        print("[vision] center_top_camera_m:", np.asarray(box_center_camera_m, dtype=np.float64))
        print("[vision] camera(view)->base transform:")
        print(camera_to_base)
        print("[vision] center_top_base_m:", box_center_base_m)
        if long_axis_camera is not None:
            axis_base = np.asarray(camera_to_base, dtype=np.float64)[:3, :3] @ np.asarray(
                long_axis_camera, dtype=np.float64
            )
            yaw_base_deg = float(np.degrees(np.arctan2(axis_base[1], axis_base[0])) % 180.0)
            print(
                f"[vision] box yaw in base (long axis, mod 180): {yaw_base_deg:.1f} deg "
                "-- stage 1: printed only, NOT commanded."
            )

        print_stage("3/5 vision_pre_push", "building target")
        command = build_vision_pre_push_command(
            dyn_model,
            dyn_state,
            robot_model,
            q,
            box_center_base_m,
            approach_time=approach_time,
            hold_time=hold_time,
            max_reference_xy_shift_m=max_reference_xy_shift_m,
            midpoint_offset_xy_m=midpoint_offset_xy_m,
            drop_delta_axis_base_xy=drop_delta_axis_base_xy,
        )
        if not send_stage(
            robot,
            command,
            "3/5 vision_pre_push",
            timeout_sec=stage_timeout_sec(
                approach_time,
                hold_time,
                min_timeout_sec=min_command_timeout_sec,
                margin_sec=command_timeout_margin_sec,
            ),
        ):
            print_stage("3/5 vision_pre_push", "FAILED; aborting")
            return done
        print_stage("3/5 vision_pre_push", "reached")

        if not continue_pick:
            done = True
            print_stage("3/5 vision_pre_push", "pre-push-only stop; done=True")
            return done

        print_stage("4/5 inward_push", "building target")
        q = robot.get_state().position
        if not send_stage(
            robot,
            build_impedance_push_command(dyn_model, dyn_state, q),
            "4/5 inward_push",
            timeout_sec=stage_timeout_sec(
                PUSH_HOLD_TIME,
                min_timeout_sec=min_command_timeout_sec,
                margin_sec=command_timeout_margin_sec,
            ),
        ):
            print_stage("4/5 inward_push", "FAILED; aborting")
            return done
        print_stage("4/5 inward_push", "done")

        print_stage("5/5 lift", "building target")
        q = robot.get_state().position
        if not send_stage(
            robot,
            build_impedance_lift_command(dyn_model, dyn_state, q),
            "5/5 lift",
            timeout_sec=stage_timeout_sec(
                LIFT_MINIMUM_TIME,
                LIFT_HOLD_TIME,
                min_timeout_sec=min_command_timeout_sec,
                margin_sec=command_timeout_margin_sec,
            ),
        ):
            print_stage("5/5 lift", "FAILED; aborting")
            return done
        print_stage("5/5 lift", "done")

        done = True
        print("=" * 60)
        print(f"[picking] picking motion COMPLETED. done = {done}")
        print("=" * 60)
        return done
    except KeyboardInterrupt:
        print_stage("interrupted", "KeyboardInterrupt while waiting in the last printed stage")
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vision-adjusted picking_box_2")
    parser.add_argument("--address", type=str, required=True, help="Robot address")
    parser.add_argument("--model", type=str, default="a", help="Robot Model Name")
    parser.add_argument("--power", type=str, default=".*", help="Power device name regex")
    parser.add_argument(
        "--box-center-camera",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="center_top_camera_m from inference.py, in meters.",
    )
    parser.add_argument(
        "--vision-json",
        type=str,
        help="Path to inference.py JSON/JSONL output; the latest JSON record is used.",
    )
    parser.add_argument(
        "--allow-unreliable-vision",
        action="store_true",
        help="Accept a vision JSON record even when ok=false. Use only for debugging.",
    )
    parser.add_argument(
        "--view-rotation",
        choices=("none", "cw90", "ccw90", "180"),
        default="cw90",
        help="View rotation used by inference.py. Current D405 mount normally uses cw90.",
    )
    parser.add_argument(
        "--approach-time",
        type=float,
        default=VISION_APPROACH_MINIMUM_TIME,
        help="Minimum time for the vision-adjusted pre-push Cartesian motion.",
    )
    parser.add_argument(
        "--hold-time",
        type=float,
        default=VISION_APPROACH_HOLD_TIME,
        help="Control hold time for the vision-adjusted pre-push command.",
    )
    parser.add_argument(
        "--max-reference-xy-shift-m",
        type=float,
        default=VISION_APPROACH_MAX_REFERENCE_XY_SHIFT_M,
        help="Abort if vision target shifts the recorded hand midpoint farther than this.",
    )
    parser.add_argument(
        "--allow-large-vision-shift",
        action="store_true",
        help="Disable the max x/y shift safety gate.",
    )
    parser.add_argument(
        "--midpoint-offset-xy-m",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("DX", "DY"),
        help="Optional base-frame x/y offset added to the transformed box center.",
    )
    parser.add_argument(
        "--box-x-offset",
        "--box-center-offset-x-m",
        dest="box_x_offset",
        type=float,
        default=0.0,
        help=(
            "Additional base-frame x offset for the grasp midpoint. "
            "This is added on top of --midpoint-offset-xy-m DX."
        ),
    )
    parser.add_argument(
        "--live-vision-frames",
        type=int,
        default=LIVE_VISION_FRAMES_NEEDED,
        help="Number of usable live center frames to median before moving.",
    )
    parser.add_argument(
        "--live-vision-timeout-sec",
        type=float,
        default=LIVE_VISION_TIMEOUT_SEC,
        help="Maximum live D405 capture time before aborting.",
    )
    parser.add_argument(
        "--live-center-spread-m",
        type=float,
        default=LIVE_VISION_MAX_CENTER_SPREAD_M,
        help="Abort if accepted live center candidates spread more than this.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help=(
            "Show a live window during vision capture with the fitted box and base-frame x/y/yaw. "
            "After capture, press c/space/enter to continue or q/ESC to abort."
        ),
    )
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Observation mode: stream the visualization at the current posture, send NO motion commands.",
    )
    parser.add_argument(
        "--command-timeout-margin-sec",
        type=float,
        default=COMMAND_TIMEOUT_MARGIN_SEC,
        help=(
            "Extra watchdog slack added to each command's expected motion/hold time. "
            "A timed-out stage is canceled instead of blocking forever."
        ),
    )
    parser.add_argument(
        "--min-command-timeout-sec",
        type=float,
        default=COMMAND_TIMEOUT_MIN_SEC,
        help="Minimum watchdog timeout for any robot command stage.",
    )
    parser.set_defaults(continue_pick=True)
    parser.add_argument(
        "--continue-pick",
        dest="continue_pick",
        action="store_true",
        help="After reaching pre-push, continue with inward y push and lift. This is now the default.",
    )
    parser.add_argument(
        "--pre-push-only",
        dest="continue_pick",
        action="store_false",
        help="Stop after the vision-adjusted pre-push pose, before y-axis inward push.",
    )
    return parser.parse_args(argv)


def combined_midpoint_offset_xy(args: argparse.Namespace) -> tuple[float, float]:
    """Return the final base-frame grasp midpoint offset.

    --box-x-offset and --midpoint-offset-xy-m DX set the same quantity, so
    giving both is refused instead of silently summing them.
    """
    midpoint_dx, midpoint_dy = (float(v) for v in args.midpoint_offset_xy_m)
    box_dx = float(args.box_x_offset)
    if abs(midpoint_dx) > 1e-12 and abs(box_dx) > 1e-12:
        raise SystemExit(
            "Use either --box-x-offset or --midpoint-offset-xy-m DX, not both "
            f"(got {box_dx:+.3f} and {midpoint_dx:+.3f})."
        )
    return midpoint_dx + box_dx, midpoint_dy


if __name__ == "__main__":
    args = parse_args()
    if args.command_timeout_margin_sec < 0.0:
        raise SystemExit("--command-timeout-margin-sec must be non-negative")
    if args.min_command_timeout_sec <= 0.0:
        raise SystemExit("--min-command-timeout-sec must be positive")
    center_camera = resolve_box_center_camera(args)
    max_shift = None if args.allow_large_vision_shift else float(args.max_reference_xy_shift_m)
    midpoint_offset_xy = combined_midpoint_offset_xy(args)
    if any(abs(v) > 1e-12 for v in midpoint_offset_xy):
        print(f"[vision_pre_push] configured base-frame grasp offset xy: {midpoint_offset_xy}")
    raise SystemExit(
        0
        if main(
            address=args.address,
            model=args.model,
            power=args.power,
            box_center_camera_m=center_camera,
            view_rotation=args.view_rotation,
            approach_time=float(args.approach_time),
            hold_time=float(args.hold_time),
            max_reference_xy_shift_m=max_shift,
            midpoint_offset_xy_m=midpoint_offset_xy,
            live_vision_frames_needed=int(args.live_vision_frames),
            live_vision_timeout_sec=float(args.live_vision_timeout_sec),
            live_center_spread_m=float(args.live_center_spread_m),
            continue_pick=bool(args.continue_pick),
            visualize=bool(args.visualize or args.visualize_only),
            visualize_only=bool(args.visualize_only),
            command_timeout_margin_sec=float(args.command_timeout_margin_sec),
            min_command_timeout_sec=float(args.min_command_timeout_sec),
        )
        else 1
    )
