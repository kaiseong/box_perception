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
from pathlib import Path
from typing import Any

import numpy as np
import rby1_sdk as rby

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

# Time (seconds) the robot takes to reach each joint pose.
MINIMUM_TIME = 2.0

# Rate (Hz) at which the FT-sensor monitoring callback is invoked.
FT_MONITOR_RATE = 10.0

# Link indices into the dynamics state (order MUST match DYN_LINK_NAMES below).
DYN_LINK_NAMES = ["base", "link_torso_5", "ee_right", "ee_left", "link_head_2"]
BASE_INDEX, TORSO_INDEX, EE_RIGHT_INDEX, EE_LEFT_INDEX, HEAD2_INDEX = 0, 1, 2, 3, 4

# ---- Vision pre-push Cartesian positioning parameters ----
# Use base for pre-push so x/y corrections are table-horizontal even when the
# torso is pitched. START_TO_PICKING supplies the target EEF z and rotation.
VISION_PRE_PUSH_REFERENCE_LINK = "base"
VISION_PRE_PUSH_REFERENCE_INDEX = BASE_INDEX

# ---- Cartesian impedance "push inward" parameters ----
# Keep the push in base as well. With the current fixed torso posture, base +/-
# y is the intended inward grip direction, and base z stays invariant when the
# pre-push x target changes.
IMPEDANCE_REFERENCE_LINK = "base"
PUSH_DISTANCE = 0.1
IMPEDANCE_TRANSLATION_WEIGHT = [500.0, 1000.0, 500.0]
IMPEDANCE_ROTATION_WEIGHT = [50.0, 50.0, 50.0]
PUSH_HOLD_TIME = 3.0

# ---- Cartesian lift parameters ----
IMPEDANCE_LIFT_REFERENCE_LINK = "base"
LIFT_HEIGHT = 0.15
LIFT_MINIMUM_TIME = 5.0
LIFT_LINEAR_VELOCITY_LIMIT = 0.1
LIFT_ANGULAR_VELOCITY_LIMIT = float(np.pi / 4)
LIFT_LINEAR_ACCELERATION_LIMIT = 0.5
LIFT_ANGULAR_ACCELERATION_LIMIT = float(np.pi)
LIFT_JOINT_STIFFNESS = [100.0] * 7
LIFT_JOINT_DAMPING_RATIO = 1.0
LIFT_JOINT_TORQUE_LIMIT = [50.0] * 7
LIFT_HOLD_TIME = 100.0

# ---- Vision extrinsic: D405 camera frame into T5/link_torso_5 frame ----
HEAD2_TO_CAMERA_XYZ_RPY_ZYX_DEG = np.array(
    [0.023, 0.0, 0.066, 0.0, 90.0, 0.0],
    dtype=np.float64,
)
T5_TO_HEAD1_ZERO_HEAD0_XYZ_M = np.array([0.022, 0.0, 0.200073451525], dtype=np.float64)
HEAD_1_PITCH_RAD_STATIC = 0.436


def _rot_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def rotation_from_euler_zyx_deg(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Return R = Rz(yaw) @ Ry(pitch) @ Rx(roll), angles in degrees."""
    roll, pitch, yaw = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    return _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)


def make_transform(translation: Any, rotation: Any = None) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    if rotation is not None:
        T[:3, :3] = np.asarray(rotation, dtype=np.float64)
    T[:3, 3] = np.asarray(translation, dtype=np.float64)
    return T


def transform_from_xyz_rpy_zyx_deg(xyz_rpy_deg: Any) -> np.ndarray:
    x, y, z, roll_deg, pitch_deg, yaw_deg = np.asarray(xyz_rpy_deg, dtype=np.float64)
    return make_transform(
        [x, y, z],
        rotation_from_euler_zyx_deg(roll_deg, pitch_deg, yaw_deg),
    )


def invert_transform(T: Any) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    T_inv = np.eye(4, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


T_HEAD2_FROM_CAMERA = transform_from_xyz_rpy_zyx_deg(HEAD2_TO_CAMERA_XYZ_RPY_ZYX_DEG)
T_T5_FROM_HEAD2_STATIC = make_transform(T5_TO_HEAD1_ZERO_HEAD0_XYZ_M) @ make_transform(
    [0.0, 0.0, 0.0],
    _rot_y(HEAD_1_PITCH_RAD_STATIC),
)
CAMERA_TO_T5_STATIC = T_T5_FROM_HEAD2_STATIC @ T_HEAD2_FROM_CAMERA


def raw_camera_from_view_rotation_transform(view_rotation: str) -> np.ndarray:
    """Return transform from rotated analysis camera frame to raw camera frame."""
    normalized = str(view_rotation).lower()
    T = np.eye(4, dtype=np.float64)
    if normalized in ("none", "raw"):
        return T
    if normalized == "cw90":
        T[:3, :3] = _rot_z(-np.pi / 2.0)
        return T
    if normalized == "ccw90":
        T[:3, :3] = _rot_z(np.pi / 2.0)
        return T
    if normalized == "180":
        T[:3, :3] = _rot_z(np.pi)
        return T
    raise ValueError(f"unsupported view_rotation: {view_rotation!r}")


def camera_to_t5_for_view_rotation(camera_to_t5: Any, view_rotation: str) -> np.ndarray:
    """Adapt a raw camera->T5 transform for inference output in a rotated view."""
    return np.asarray(camera_to_t5, dtype=np.float64) @ raw_camera_from_view_rotation_transform(
        view_rotation
    )


def compute_camera_to_t5_transform(
    dyn_model: Any = None,
    dyn_state: Any = None,
    q: Any = None,
) -> np.ndarray:
    """Return camera->T5 transform for the current robot state."""
    if dyn_model is None and dyn_state is None and q is None:
        return CAMERA_TO_T5_STATIC.copy()
    if dyn_model is None or dyn_state is None or q is None:
        raise ValueError("dyn_model, dyn_state, and q must be provided together.")

    dyn_state.set_q(q)
    dyn_model.compute_forward_kinematics(dyn_state)
    T_t5_from_head2 = dyn_model.compute_transformation(dyn_state, TORSO_INDEX, HEAD2_INDEX)
    return np.asarray(T_t5_from_head2, dtype=np.float64) @ T_HEAD2_FROM_CAMERA


def compute_camera_to_t5_for_view_rotation(
    view_rotation: str,
    dyn_model: Any = None,
    dyn_state: Any = None,
    q: Any = None,
) -> np.ndarray:
    return camera_to_t5_for_view_rotation(
        compute_camera_to_t5_transform(dyn_model, dyn_state, q),
        view_rotation,
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
                LIFT_LINEAR_ACCELERATION_LIMIT,
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
    "right_arm": [-0.175, -1.309, -0.262, -1.571, -2.618, 0.000, -0.175],
    "left_arm": [-0.175, 1.309, 0.262, -1.571, 2.618, 0.000, 0.175],
    "head": [0.000, 0.436],
}

READY_TO_PICKING = {
    "torso": [0.000, 0.000, 0.000, 0.349, 0.000, 0.000],
    "right_arm": [-0.111, -0.987, -0.205, -1.463, -2.454, 1.744, 0.50],
    "left_arm": [-0.111, 0.987, 0.205, -1.463, 2.454, 1.744, -0.50],
    "head": [0.000, 0.436],
}

START_TO_PICKING = {
    "torso": [0.000, 0.000, 0.000, 0.349, 0.000, 0.000],
    "right_arm": [-0.171, -0.841, -0.153, -1.511, -2.403, 1.743, 0.5],
    "left_arm": [-0.171, 0.841, 0.153, -1.511, 2.403, 1.743, -0.5],
    "head": [0.000, 0.436],
}

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
    if not allow_unreliable and record.get("ok") is False:
        reasons = record.get("failure_reasons") or record.get("plane_failure_reasons") or []
        raise ValueError(f"vision record is not ok; refusing to move. reasons={reasons}")

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
) -> dict[str, Any] | None:
    """Detect the box from the live D405 at the current posture.

    Runs the rim-plane estimator with the stored plane calibration and keeps
    confidence-ok frames are preferred. For this stage-1 motion, yaw is not
    commanded, so frames whose only issue is an underconstrained long-axis
    center are also accepted as center-only candidates and checked by a
    multi-frame center-spread gate. Returns the per-axis median center
    (analysis camera frame), optional long-axis direction, and frame count, or
    None when not enough usable frames arrive within the timeout.
    """
    import time

    import cv2

    cv2.setNumThreads(1)
    import pyrealsense2 as rs

    if str(BOX_PERCEPTION_ROOT) not in sys.path:
        sys.path.insert(0, str(BOX_PERCEPTION_ROOT))
    from box_pose import CameraIntrinsics, estimate_plane_box, segment_yellow_box
    from replay_recording import rotate_array_for_view, rotate_intrinsics_for_view

    rim_cfg = json.loads(RIM_PLANE_CONFIG.read_text(encoding="utf-8"))
    rim_plane = (rim_cfg["normal"], rim_cfg["point"])

    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.bgr8, CAMERA_FPS)
    rs_config.enable_stream(rs.stream.depth, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.z16, CAMERA_FPS)
    profile = pipeline.start(rs_config)
    align = rs.align(rs.stream.color)
    depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
    raw = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    intrinsics = rotate_intrinsics_for_view(
        CameraIntrinsics(fx=float(raw.fx), fy=float(raw.fy), cx=float(raw.ppx), cy=float(raw.ppy)),
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        rotation=view_rotation,
        intrinsics_cls=CameraIntrinsics,
    )

    centers: list[Any] = []
    long_axes: list[Any] = []
    modes: list[str] = []
    deadline = time.monotonic() + float(timeout_sec)
    try:
        while len(centers) < int(frames_needed) and time.monotonic() < deadline:
            frames = align.process(pipeline.wait_for_frames())
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                continue
            image = rotate_array_for_view(np.asanyarray(color.get_data()), view_rotation)
            depth_m = rotate_array_for_view(
                np.asanyarray(depth.get_data()).astype(np.float32) * depth_scale, view_rotation
            )
            mask, _ = segment_yellow_box(image, keep_largest_component=False)
            estimate = estimate_plane_box(mask, depth_m, intrinsics, rim_plane=rim_plane)
            usable, mode = live_estimate_center_mode(estimate)
            if usable:
                centers.append(estimate.center_top_camera_m)
                modes.append(mode)
                if estimate.confidence.ok:
                    long_axes.append(estimate.support["long_axis_camera"])
                elif len(centers) == 1 or len(centers) % 3 == 0:
                    print(f"[vision] accepted center-only frame: {mode}")
            else:
                print(f"[vision] frame rejected: {mode}")
    finally:
        pipeline.stop()

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
) -> dict[str, np.ndarray]:
    """Shift START_TO_PICKING hand targets so their midpoint matches the box center.

    z and rotation are copied from the recorded START_TO_PICKING FK in base.
    Only base x/y receive the same delta on both hands, preserving recorded
    hand spacing while avoiding T5-pitch-induced vertical drift.
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

    T_right_target = T_right_ref.copy()
    T_left_target = T_left_ref.copy()
    T_right_target[:2, 3] += xy_delta
    T_left_target[:2, 3] += xy_delta

    return {
        "right_target": T_right_target,
        "left_target": T_left_target,
        "reference_midpoint_xy": reference_midpoint_xy,
        "target_midpoint_xy": target_midpoint_xy,
        "xy_delta": xy_delta,
    }


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
) -> Any:
    targets = vision_pre_push_targets(
        dyn_model,
        dyn_state,
        robot_model,
        q_template,
        box_center_base_m,
        midpoint_offset_xy_m=midpoint_offset_xy_m,
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
        "[vision_pre_push] base-frame z and rotation are inherited from "
        "recorded START_TO_PICKING; only base x/y are shifted."
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


def send_once(robot: Any, builder: Any) -> bool:
    """Send a single command and return whether it finished successfully."""
    feedback = robot.send_command(builder).get()
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
    """Push both hands inward to grip the box, expressed in base."""
    return build_dual_arm_impedance_command(
        dyn_model,
        dyn_state,
        q,
        reference_link=IMPEDANCE_REFERENCE_LINK,
        ref_index=BASE_INDEX,
        inward=PUSH_DISTANCE,
        lift=0.0,
        translation_weight=IMPEDANCE_TRANSLATION_WEIGHT,
        hold_time=PUSH_HOLD_TIME,
        label="push",
    )


def build_impedance_lift_command(dyn_model: Any, dyn_state: Any, q: Any) -> Any:
    """Raise both hands straight up by LIFT_HEIGHT along base +z.

    The public name is kept for existing call sites. The implementation sends
    independent right/left Cartesian arm-component commands in one RobotCommand
    so the torso stays fixed while both hands receive synchronized targets.
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
        f"  (over {LIFT_MINIMUM_TIME:.0f}s)"
    )

    return build_dual_target_cartesian_command(
        reference_link=IMPEDANCE_LIFT_REFERENCE_LINK,
        right_target=T_right_target,
        left_target=T_left_target,
        minimum_time=LIFT_MINIMUM_TIME,
        hold_time=LIFT_HOLD_TIME,
    )


class FTMonitor:
    """Monitor both arm force magnitudes through robot state callbacks."""

    def __init__(self) -> None:
        self.samples = 0
        self.peak_force_right = 0.0
        self.peak_force_left = 0.0

    def callback(self, robot_state: Any) -> None:
        ft_right = robot_state.ft_sensor_right
        ft_left = robot_state.ft_sensor_left

        force_right = float(np.linalg.norm(ft_right.force))
        force_left = float(np.linalg.norm(ft_left.force))

        self.samples += 1
        self.peak_force_right = max(self.peak_force_right, force_right)
        self.peak_force_left = max(self.peak_force_left, force_left)

        print(
            f"[FT] right | force {ft_right.force} |F|={force_right:6.2f}N "
            f"  ||  left | force {ft_left.force} |F|={force_left:6.2f}N "
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

    ft_monitor = FTMonitor()
    robot.start_state_update(ft_monitor.callback, FT_MONITOR_RATE)

    done = False
    try:
        for name, pose in JOINT_SEQUENCE:
            print(f"[picking] moving to '{name}' (joint position) ...")
            if not send_once(robot, build_pose_command(pose, MINIMUM_TIME)):
                print(f"[picking] FAILED while moving to '{name}'. Aborting.")
                return done
            print(f"[picking] reached '{name}'.")

        q = robot.get_state().position
        camera_to_base = compute_camera_to_base_for_view_rotation(
            view_rotation,
            dyn_model,
            dyn_state,
            q,
        )

        long_axis_camera = None
        if box_center_camera_m is None:
            # Live capture AT this pose, so the FK-based camera->base above and
            # the measurement share the exact same posture.
            print("[vision] live capture from D405 ...")
            live = capture_box_live(
                view_rotation,
                frames_needed=live_vision_frames_needed,
                timeout_sec=live_vision_timeout_sec,
                max_center_spread_m=live_center_spread_m,
            )
            if live is None:
                print(
                    "[vision] FAILED: not enough usable center frames within "
                    f"{live_vision_timeout_sec:.0f}s. Aborting before contact."
                )
                return done
            box_center_camera_m = live["center_camera_m"]
            long_axis_camera = live["long_axis_camera"]
            print(
                f"[vision] usable frames used: {live['frames_used']} "
                f"| center spread={live['center_spread_m'] * 1000.0:.1f} mm "
                f"| modes={sorted(set(live['modes']))}"
            )

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

        print("[picking] moving to vision-adjusted pre-push pose ...")
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
        )
        if not send_once(robot, command):
            print("[picking] FAILED during vision-adjusted pre-push motion. Aborting.")
            return done
        print("[picking] reached vision-adjusted pre-push pose.")

        if not continue_pick:
            done = True
            print("[picking] stopped before inward y-axis push. done = True")
            return done

        print("[picking] pushing EEF inward with Cartesian impedance control ...")
        q = robot.get_state().position
        if not send_once(robot, build_impedance_push_command(dyn_model, dyn_state, q)):
            print("[picking] FAILED during Cartesian impedance push. Aborting.")
            return done
        print("[picking] Cartesian impedance push done.")

        print("[picking] lifting the box ...")
        q = robot.get_state().position
        if not send_once(robot, build_impedance_lift_command(dyn_model, dyn_state, q)):
            print("[picking] FAILED during lift. Aborting.")
            return done
        print("[picking] box lifted.")

        done = True
        print("=" * 60)
        print(f"[picking] picking motion COMPLETED. done = {done}")
        print("=" * 60)
        return done
    finally:
        robot.stop_state_update()
        print(
            f"[FT] monitoring stopped. samples={ft_monitor.samples}, "
            f"peak |F| right={ft_monitor.peak_force_right:.2f}N, "
            f"left={ft_monitor.peak_force_left:.2f}N"
        )


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


if __name__ == "__main__":
    args = parse_args()
    center_camera = resolve_box_center_camera(args)
    max_shift = None if args.allow_large_vision_shift else float(args.max_reference_xy_shift_m)
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
            midpoint_offset_xy_m=tuple(float(v) for v in args.midpoint_offset_xy_m),
            live_vision_frames_needed=int(args.live_vision_frames),
            live_vision_timeout_sec=float(args.live_vision_timeout_sec),
            live_center_spread_m=float(args.live_center_spread_m),
            continue_pick=bool(args.continue_pick),
        )
        else 1
    )
