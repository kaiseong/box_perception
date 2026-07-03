#!/usr/bin/env python3
"""Vision-adjusted pre-pick motion for the D405 pallet-box task.

This variant keeps the recorded START_TO_PICKING hand height and hand
orientation, but shifts only x/y from the latest camera-estimated box center.

Default sequence:
  1. ready -> ready_to_picking       (joint position control)
  2. live vision at that pose        (D405 + rim-plane estimator, confident frames only)
  3. vision_pre_push                 (Cartesian impedance control)

The script stops at the pose just before the inward y-axis push. Pass
--continue-pick to run the existing inward push and lift after that.

Vision input, one of:
  - default: LIVE capture from the D405 at the pre-push pose (median over
    LIVE_VISION_FRAMES_NEEDED confident frames; aborts if not enough).
  - --vision-json <path>: latest record of inference.py --print-json output.
  - --box-center-camera X Y Z: manual center_top_camera_m.
All are in the cw90 analysis camera frame (see --view-rotation) and are
transformed to link_torso_5/T5 before building the arm targets. The detected
box yaw is printed for reference but NOT commanded in this stage-1 script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from picking_box import (
    DYN_LINK_NAMES,
    EE_LEFT_INDEX,
    EE_RIGHT_INDEX,
    FTMonitor,
    FT_MONITOR_RATE,
    IMPEDANCE_REFERENCE_LINK,
    JOINT_SEQUENCE,
    LIFT_ANGULAR_ACCELERATION_LIMIT,
    LIFT_ANGULAR_VELOCITY_LIMIT,
    LIFT_JOINT_DAMPING_RATIO,
    LIFT_JOINT_STIFFNESS,
    LIFT_JOINT_TORQUE_LIMIT,
    LIFT_LINEAR_ACCELERATION_LIMIT,
    LIFT_LINEAR_VELOCITY_LIMIT,
    MINIMUM_TIME,
    START_TO_PICKING,
    TORSO_INDEX,
    build_impedance_lift_command,
    build_impedance_push_command,
    build_pose_command,
    compute_camera_to_t5_for_view_rotation,
    rby,
    send_once,
    transform_camera_point_to_t5,
)


VISION_APPROACH_MINIMUM_TIME = 3.0
VISION_APPROACH_HOLD_TIME = 3.0
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


def capture_box_live(view_rotation: str) -> dict[str, Any] | None:
    """Detect the box from the live D405 at the current posture.

    Runs the rim-plane estimator with the stored plane calibration and keeps
    only confidence-ok frames; returns the per-axis median center (analysis
    camera frame), the long-axis direction, and the frame count, or None when
    not enough confident frames arrive within the timeout.
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
    deadline = time.monotonic() + LIVE_VISION_TIMEOUT_SEC
    try:
        while len(centers) < LIVE_VISION_FRAMES_NEEDED and time.monotonic() < deadline:
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
            if estimate.center_top_camera_m is not None and estimate.confidence.ok:
                centers.append(estimate.center_top_camera_m)
                long_axes.append(estimate.support["long_axis_camera"])
            elif estimate.failure_reasons:
                print(f"[vision] frame rejected: {','.join(estimate.failure_reasons[:2])}")
    finally:
        pipeline.stop()

    if len(centers) < LIVE_VISION_FRAMES_NEEDED:
        return None
    return {
        "center_camera_m": np.median(np.asarray(centers, dtype=np.float64), axis=0),
        "long_axis_camera": np.median(np.asarray(long_axes, dtype=np.float64), axis=0),
        "frames_used": len(centers),
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
    """FK of recorded START_TO_PICKING EEF poses in the T5/link_torso_5 frame."""
    q_start = q_for_recorded_pose(robot_model, q_template, START_TO_PICKING)
    dyn_state.set_q(q_start)
    dyn_model.compute_forward_kinematics(dyn_state)
    T_t5_right = np.asarray(
        dyn_model.compute_transformation(dyn_state, TORSO_INDEX, EE_RIGHT_INDEX),
        dtype=np.float64,
    ).copy()
    T_t5_left = np.asarray(
        dyn_model.compute_transformation(dyn_state, TORSO_INDEX, EE_LEFT_INDEX),
        dtype=np.float64,
    ).copy()
    return T_t5_right, T_t5_left


def vision_pre_push_targets(
    dyn_model: Any,
    dyn_state: Any,
    robot_model: Any,
    q_template: np.ndarray,
    box_center_t5_m: np.ndarray,
    *,
    midpoint_offset_xy_m: tuple[float, float] = (0.0, 0.0),
) -> dict[str, np.ndarray]:
    """Shift START_TO_PICKING hand targets so their midpoint matches the box center.

    z and rotation are copied from the recorded START_TO_PICKING FK. Only x/y
    receive the same delta on both hands, preserving the recorded hand spacing.
    """
    T_right_ref, T_left_ref = start_to_picking_reference_targets(
        dyn_model,
        dyn_state,
        robot_model,
        q_template,
    )
    reference_midpoint_xy = 0.5 * (T_right_ref[:2, 3] + T_left_ref[:2, 3])
    target_midpoint_xy = np.asarray(box_center_t5_m[:2], dtype=np.float64) + np.asarray(
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
    box_center_t5_m: np.ndarray,
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
        box_center_t5_m,
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

    print("[vision_pre_push] box center in T5:", np.asarray(box_center_t5_m, dtype=np.float64))
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
    print("[vision_pre_push] z and rotation are inherited from recorded START_TO_PICKING.")

    def arm_cartesian_impedance(link_name: str, T_target: np.ndarray) -> Any:
        return (
            rby.CartesianImpedanceControlCommandBuilder()
            .set_command_header(
                rby.CommandHeaderBuilder().set_control_hold_time(hold_time)
            )
            .add_target(
                IMPEDANCE_REFERENCE_LINK,
                link_name,
                T_target,
                LIFT_LINEAR_VELOCITY_LIMIT,
                LIFT_ANGULAR_VELOCITY_LIMIT,
                LIFT_LINEAR_ACCELERATION_LIMIT,
                LIFT_ANGULAR_ACCELERATION_LIMIT,
            )
            .set_joint_stiffness(LIFT_JOINT_STIFFNESS)
            .set_joint_damping_ratio(LIFT_JOINT_DAMPING_RATIO)
            .set_joint_torque_limit(LIFT_JOINT_TORQUE_LIMIT)
            .set_minimum_time(approach_time)
        )

    return rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_body_command(
            rby.BodyComponentBasedCommandBuilder()
            .set_right_arm_command(
                arm_cartesian_impedance("ee_right", targets["right_target"])
            )
            .set_left_arm_command(
                arm_cartesian_impedance("ee_left", targets["left_target"])
            )
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
        camera_to_t5 = compute_camera_to_t5_for_view_rotation(
            view_rotation,
            dyn_model,
            dyn_state,
            q,
        )

        long_axis_camera = None
        if box_center_camera_m is None:
            # Live capture AT this pose, so the FK-based camera->T5 above and
            # the measurement share the exact same posture.
            print("[vision] live capture from D405 (confident frames only) ...")
            live = capture_box_live(view_rotation)
            if live is None:
                print(
                    "[vision] FAILED: not enough confident frames within "
                    f"{LIVE_VISION_TIMEOUT_SEC:.0f}s. Aborting before contact."
                )
                return done
            box_center_camera_m = live["center_camera_m"]
            long_axis_camera = live["long_axis_camera"]
            print(f"[vision] confident frames used: {live['frames_used']}")

        box_center_t5_m = transform_camera_point_to_t5(box_center_camera_m, camera_to_t5)
        print(f"[vision] view_rotation={view_rotation}")
        print("[vision] center_top_camera_m:", np.asarray(box_center_camera_m, dtype=np.float64))
        print("[vision] camera(view)->T5 transform:")
        print(camera_to_t5)
        print("[vision] center_top_t5_m:", box_center_t5_m)
        if long_axis_camera is not None:
            axis_t5 = np.asarray(camera_to_t5, dtype=np.float64)[:3, :3] @ np.asarray(
                long_axis_camera, dtype=np.float64
            )
            yaw_t5_deg = float(np.degrees(np.arctan2(axis_t5[1], axis_t5[0])) % 180.0)
            print(
                f"[vision] box yaw in T5 (long axis, mod 180): {yaw_t5_deg:.1f} deg "
                "-- stage 1: printed only, NOT commanded."
            )

        print("[picking] moving to vision-adjusted pre-push pose ...")
        command = build_vision_pre_push_command(
            dyn_model,
            dyn_state,
            robot_model,
            q,
            box_center_t5_m,
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
        help="Optional T5-frame x/y offset added to the transformed box center.",
    )
    parser.add_argument(
        "--continue-pick",
        action="store_true",
        help="After reaching pre-push, continue with inward y push and lift.",
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
            continue_pick=bool(args.continue_pick),
        )
        else 1
    )
