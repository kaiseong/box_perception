#!/usr/bin/env python3
"""Vision-adjusted pre-pick motion with combined mobile-base SE(2) alignment.

This variant keeps the recorded START_TO_PICKING hand height and hand
orientation, but shifts only x/y from the latest camera-estimated box center.

Default sequence:
  0. gripper_home_open               (Dynamixel homing + continuous max-open hold)
  1. ready                           (joint position control)
  2. live vision at ready pose       (D405 + rim-plane estimator, usable center frames)
  3-4. mobile_base_se2_align         (M-model SE(2), closed loop, yaw + x/y)
  5. vision_pre_push                 (START_TO_PICKING z/rotation, vision x/y)
  6. inward y-axis push              (Cartesian impedance control)
  7. lift                            (dual-target Cartesian control)

Pass --pre-push-only to stop at the pose just before the inward y-axis push.

Vision input, one of:
  - default: LIVE capture from the D405 at the pre-push pose (median over
    LIVE_VISION_FRAMES_NEEDED usable center frames; aborts if not enough).
  - --vision-json <path>: latest record of inference.py --print-json output.
  - --box-center-camera X Y Z: manual center_top_camera_m.
All are in the cw90 analysis camera frame (see --view-rotation). They are
transformed to base for vision_pre_push so the x/y correction is
table-horizontal and the START_TO_PICKING base-frame EEF z/rotation stay fixed.
For the current side grasp, the box long axis must be parallel to base y
(90/270 degrees in base xy, equivalently zero error modulo 180) before
contact. This script uses the mobile base theta axis for that yaw alignment
and does NOT rotate the wrists in the first pass. In picking_box_5, yaw and x/y
are handled in one closed-loop stage. By default x/y/theta are corrected
together even when yaw error is still large, so the base does not rotate in
place into an off-center box.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
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
# ImpedanceControlCommandBuilder has no trajectory minimum_time in the SDK
# examples. Ramp the target itself over this duration to avoid a spring target
# jump that slams into the box.
PUSH_RAMP_TIME = 1.0
PUSH_STREAM_PERIOD_SEC = 1.0 / 30.0
PUSH_STREAM_COMMAND_HOLD_TIME = 1.0
PUSH_STREAM_PRIORITY = 1
PUSH_HOLD_TIME = 0.5

# ---- Cartesian lift parameters ----
IMPEDANCE_LIFT_REFERENCE_LINK = "base"
LIFT_HEIGHT = 0.12
LIFT_MINIMUM_TIME = 1.0
LIFT_LINEAR_VELOCITY_LIMIT = 0.1
LIFT_ANGULAR_VELOCITY_LIMIT = float(np.pi / 4)
LIFT_LINEAR_ACCELERATION_LIMIT = 0.5   # m/s^2 (CartesianImpedance add_target)
LIFT_ANGULAR_ACCELERATION_LIMIT = float(np.pi)  # rad/s^2
LIFT_JOINT_STIFFNESS = [300.0] * 7
LIFT_JOINT_DAMPING_RATIO = 1.0
# RBY1 M arm effort limits from rby1-sdk/models/rby1m/urdf/model.urdf.
LIFT_JOINT_TORQUE_LIMIT = [70.0, 70.0, 70.0, 40.0, 10.0, 10.0, 8.0]
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

# ---- Gripper max-open hold ----
# The RBY1 gripper is driven through the UPC Dynamixel bus, separate from the
# arm Cartesian commands. Keep it max-open while the arms move inward around the
# box; gripping force comes from the arms, not from closing gripper fingers.
GRIPPER_OPEN_DEFAULT = True
GRIPPER_DEVICE_IDS = (0, 1)
GRIPPER_DIRECTION = False
GRIPPER_OPEN_NORMALIZED_Q = np.array([1.0, 1.0], dtype=np.float64)
GRIPPER_HOMING_TORQUE_NM = 0.5
GRIPPER_HOMING_STALL_COUNT = 15
GRIPPER_HOMING_SLEEP_SEC = 0.05
GRIPPER_POSITION_TORQUE_NM = 5.0
GRIPPER_COMMAND_PERIOD_SEC = 0.1
GRIPPER_SETUP_SETTLE_SEC = 0.5
GRIPPER_STOP_JOIN_TIMEOUT_SEC = 1.0

# ---- Mobile base closed-loop alignment (picking_box_5) ----
# M model supports SE(2) x/y/theta velocity commands. The control law is
# deliberately conservative: use vision, move/turn a small amount, then look
# again. In the combined stage, yaw and x/y share one SE(2) command. The default
# threshold covers the full modulo-180 yaw-error range, so x/y remains enabled
# during yaw alignment and avoids rotating in place into an off-center box.
MOBILE_BASE_ALIGN_DEFAULT = True
MOBILE_BASE_TARGET_X_M = 0.45
MOBILE_BASE_X_TOLERANCE_M = 0.01
MOBILE_BASE_Y_TOLERANCE_M = 0.01
MOBILE_BASE_MAX_SPEED_MPS = 0.04
MOBILE_BASE_MAX_STEP_M = 0.06
MOBILE_BASE_MAX_ITERATIONS = 8
MOBILE_BASE_YAW_ALIGN_DEFAULT = True
MOBILE_BASE_YAW_TARGET_DEG = 90.0
MOBILE_BASE_YAW_TOLERANCE_DEG = 4.0
MOBILE_BASE_YAW_MAX_SPEED_RADPS = 0.10
MOBILE_BASE_YAW_MAX_STEP_DEG = 20.0
MOBILE_BASE_YAW_MAX_ITERATIONS = 8
MOBILE_BASE_YAW_TOTAL_TIMEOUT_SEC = 30.0
MOBILE_BASE_YAW_MOVE_DURATION_SEC = 2.0
MOBILE_BASE_YAW_VISION_FRAMES_NEEDED = 3
MOBILE_BASE_YAW_VISION_TIMEOUT_SEC = 2.0
MOBILE_BASE_COMBINED_COARSE_YAW_THRESHOLD_DEG = 90.0
# Budget must cover the worst realistic initial error at the capped base speed plus a
# vision recapture per step; 8 s only allowed ~8 cm of total travel.
MOBILE_BASE_TOTAL_TIMEOUT_SEC = 30.0
MOBILE_BASE_MOVE_DURATION_SEC = 2.0
MOBILE_BASE_VISION_FRAMES_NEEDED = 3
MOBILE_BASE_VISION_TIMEOUT_SEC = 2.0
MOBILE_BASE_STREAM_PRIORITY = 1
MOBILE_BASE_STREAM_PERIOD_SEC = 1.0 / 30.0
# Match rby1-lerobot's command type (priority stream + SE2 velocity command), not
# its exact loop rate. Keep a bounded hold time so brief Python/D405 stalls do
# not expire the stream, while a process crash cannot keep the base moving long.
MOBILE_BASE_COMMAND_HOLD_TIME_SEC = 2.0
MOBILE_BASE_STOP_REPEATS = 3
# The arms only get to move if alignment ended near the target; a residual
# error beyond this band reproduces the far-reach IK failure this script
# exists to avoid, so the pick aborts instead of continuing.
MOBILE_BASE_MAX_RESIDUAL_XY_M = 0.05

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


def gripper_normalized_to_encoder_target(
    normalized_q: Any,
    min_q: Any,
    max_q: Any,
    *,
    direction: bool = GRIPPER_DIRECTION,
) -> np.ndarray:
    """Map normalized gripper command into homed Dynamixel encoder targets."""
    clipped = np.clip(np.asarray(normalized_q, dtype=np.float64), 0.0, 1.0)
    min_q = np.asarray(min_q, dtype=np.float64)
    max_q = np.asarray(max_q, dtype=np.float64)
    if direction:
        return clipped * (max_q - min_q) + min_q
    return (1.0 - clipped) * (max_q - min_q) + min_q


class MaxOpenGripper:
    """Home the two-finger gripper and keep writing a max-open position."""

    def __init__(self) -> None:
        self.bus = rby.DynamixelBus(rby.upc.GripperDeviceName)
        self.bus.open_port()
        self.bus.set_baud_rate(2_000_000)
        self.bus.set_torque_constant([1, 1])
        self.min_q = np.array([np.inf, np.inf], dtype=np.float64)
        self.max_q = np.array([-np.inf, -np.inf], dtype=np.float64)
        self.target_q: np.ndarray | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    def initialize(self, *, verbose: bool = True) -> bool:
        active_ids: list[int] = []
        for dev_id in GRIPPER_DEVICE_IDS:
            active = bool(self.bus.ping(dev_id))
            if verbose:
                state = "active" if active else "not active"
                print(f"[gripper] Dynamixel ID {dev_id} is {state}")
            if active:
                active_ids.append(dev_id)
        if len(active_ids) != len(GRIPPER_DEVICE_IDS):
            print(f"[gripper] expected active IDs {GRIPPER_DEVICE_IDS}, got {tuple(active_ids)}")
            return False
        self.bus.group_sync_write_torque_enable([(dev_id, 1) for dev_id in GRIPPER_DEVICE_IDS])
        return True

    def set_operating_mode(self, mode: Any) -> None:
        self.bus.group_sync_write_torque_enable([(dev_id, 0) for dev_id in GRIPPER_DEVICE_IDS])
        self.bus.group_sync_write_operating_mode([(dev_id, mode) for dev_id in GRIPPER_DEVICE_IDS])
        self.bus.group_sync_write_torque_enable([(dev_id, 1) for dev_id in GRIPPER_DEVICE_IDS])

    def homing(self) -> bool:
        self.set_operating_mode(rby.DynamixelBus.CurrentControlMode)
        q = np.zeros(len(GRIPPER_DEVICE_IDS), dtype=np.float64)
        prev_q = np.zeros(len(GRIPPER_DEVICE_IDS), dtype=np.float64)
        direction = 0
        stall_counter = 0
        try:
            while direction < 2:
                torque_sign = 1 if direction == 0 else -1
                self.bus.group_sync_write_send_torque(
                    [
                        (dev_id, GRIPPER_HOMING_TORQUE_NM * torque_sign)
                        for dev_id in GRIPPER_DEVICE_IDS
                    ]
                )
                encoders = self.bus.group_fast_sync_read_encoder(list(GRIPPER_DEVICE_IDS))
                if encoders is not None:
                    for dev_id, enc in encoders:
                        if 0 <= int(dev_id) < len(q):
                            q[int(dev_id)] = float(enc)
                self.min_q = np.minimum(self.min_q, q)
                self.max_q = np.maximum(self.max_q, q)
                if np.array_equal(prev_q, q):
                    stall_counter += 1
                prev_q = q.copy()
                if stall_counter >= GRIPPER_HOMING_STALL_COUNT:
                    print(
                        "[gripper] homing direction "
                        f"{direction} limit detected; q={q}, min={self.min_q}, max={self.max_q}"
                    )
                    direction += 1
                    stall_counter = 0
                time.sleep(GRIPPER_HOMING_SLEEP_SEC)
        finally:
            self.bus.group_sync_write_send_torque([(dev_id, 0.0) for dev_id in GRIPPER_DEVICE_IDS])
        if not np.isfinite(self.min_q).all() or not np.isfinite(self.max_q).all():
            print("[gripper] homing failed: encoder limits are not finite")
            return False
        if not np.all(self.max_q > self.min_q):
            print(f"[gripper] homing failed: invalid limits min={self.min_q}, max={self.max_q}")
            return False
        return True

    def set_target(self, normalized_q: Any) -> bool:
        if not np.isfinite(self.min_q).all() or not np.isfinite(self.max_q).all():
            print("[gripper] cannot set target before valid homing limits")
            return False
        self.target_q = gripper_normalized_to_encoder_target(
            normalized_q,
            self.min_q,
            self.max_q,
        )
        return True

    def set_open_target(self) -> bool:
        return self.set_target(GRIPPER_OPEN_NORMALIZED_Q)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=GRIPPER_STOP_JOIN_TIMEOUT_SEC)
            if self._thread.is_alive():
                print(
                    "[gripper] warning: max-open background loop did not stop within "
                    f"{GRIPPER_STOP_JOIN_TIMEOUT_SEC:.1f}s"
                )
            else:
                self._thread = None

    def _loop(self) -> None:
        self.set_operating_mode(rby.DynamixelBus.CurrentBasedPositionControlMode)
        self.bus.group_sync_write_send_torque(
            [(dev_id, GRIPPER_POSITION_TORQUE_NM) for dev_id in GRIPPER_DEVICE_IDS]
        )
        while self._running:
            if self.target_q is not None:
                self.bus.group_sync_write_send_position(
                    [(dev_id, q) for dev_id, q in enumerate(self.target_q.tolist())]
                )
            time.sleep(GRIPPER_COMMAND_PERIOD_SEC)


def enable_gripper_tool_voltage(robot: Any) -> bool:
    """Enable 12 V tool output for both arms when the SDK exposes that API."""
    setter = getattr(robot, "set_tool_flange_output_voltage", None)
    if setter is None:
        print("[gripper] set_tool_flange_output_voltage unavailable; skipping tool voltage setup")
        return True
    for arm in ("right", "left"):
        if not setter(arm, 12):
            print(f"[gripper] failed to set {arm} tool flange output voltage to 12 V")
            return False
    return True


def setup_max_open_gripper(robot: Any) -> MaxOpenGripper | None:
    """Home the gripper and start the continuous max-open command loop."""
    print_stage("0/7 gripper_home_open", "setting tool voltage")
    if not enable_gripper_tool_voltage(robot):
        return None
    gripper = MaxOpenGripper()
    print_stage("0/7 gripper_home_open", "priming Dynamixel bus after tool voltage")
    gripper.initialize(verbose=True)
    time.sleep(GRIPPER_SETUP_SETTLE_SEC)
    print_stage("0/7 gripper_home_open", "verifying Dynamixel bus")
    if not gripper.initialize(verbose=True):
        return None
    print_stage("0/7 gripper_home_open", "homing both gripper fingers")
    if not gripper.homing():
        return None
    if not gripper.set_open_target():
        return None
    gripper.start()
    print_stage("0/7 gripper_home_open", f"max-open hold started; min={gripper.min_q}, max={gripper.max_q}")
    return gripper


def mobile_base_alignment_error_xy(
    center_base_m: Any,
    *,
    target_x_m: float = MOBILE_BASE_TARGET_X_M,
) -> np.ndarray:
    """Return base-frame x/y error that the mobile base should move by.

    A positive x error means the box is too far in front, so the base should
    move forward (+x) to reduce the next measured box x. A positive y error
    means the box appears at +base-y, so the M-base should move +y to center it.
    """
    center = np.asarray(center_base_m, dtype=np.float64)
    return np.array([center[0] - float(target_x_m), center[1]], dtype=np.float64)


def is_mobile_base_aligned(
    center_base_m: Any,
    *,
    target_x_m: float = MOBILE_BASE_TARGET_X_M,
    x_tolerance_m: float = MOBILE_BASE_X_TOLERANCE_M,
    y_tolerance_m: float = MOBILE_BASE_Y_TOLERANCE_M,
) -> bool:
    """Return True when the box center is inside the mobile-base target band."""
    error_xy = mobile_base_alignment_error_xy(center_base_m, target_x_m=target_x_m)
    eps = 1e-9
    return (
        abs(float(error_xy[0])) <= float(x_tolerance_m) + eps
        and abs(float(error_xy[1])) <= float(y_tolerance_m) + eps
    )


def mobile_base_xy_residual_status(
    center_base_m: Any,
    *,
    target_x_m: float = MOBILE_BASE_TARGET_X_M,
    x_tolerance_m: float = MOBILE_BASE_X_TOLERANCE_M,
    y_tolerance_m: float = MOBILE_BASE_Y_TOLERANCE_M,
    max_residual_xy_m: float = MOBILE_BASE_MAX_RESIDUAL_XY_M,
) -> dict[str, Any]:
    """Return final xy residual status for the mobile-base contact gate."""
    residual_xy = mobile_base_alignment_error_xy(center_base_m, target_x_m=target_x_m)
    return {
        "residual_xy_m": residual_xy,
        "within_target_band": is_mobile_base_aligned(
            center_base_m,
            target_x_m=target_x_m,
            x_tolerance_m=x_tolerance_m,
            y_tolerance_m=y_tolerance_m,
        ),
        "within_safety_band": float(np.max(np.abs(residual_xy))) <= float(max_residual_xy_m) + 1e-9,
    }


def mobile_base_alignment_step_xy(
    center_base_m: Any,
    *,
    target_x_m: float = MOBILE_BASE_TARGET_X_M,
    x_tolerance_m: float = MOBILE_BASE_X_TOLERANCE_M,
    y_tolerance_m: float = MOBILE_BASE_Y_TOLERANCE_M,
    max_step_m: float = MOBILE_BASE_MAX_STEP_M,
) -> np.ndarray:
    """Return one conservative x/y correction step for the M mobile base."""
    error_xy = mobile_base_alignment_error_xy(center_base_m, target_x_m=target_x_m)
    step_xy = np.array(
        [
            0.0 if abs(float(error_xy[0])) <= float(x_tolerance_m) + 1e-9 else float(error_xy[0]),
            0.0 if abs(float(error_xy[1])) <= float(y_tolerance_m) + 1e-9 else float(error_xy[1]),
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(step_xy))
    if norm > float(max_step_m) > 0.0:
        step_xy *= float(max_step_m) / norm
    return step_xy


def mobile_base_velocity_for_step_xy(
    step_xy_m: Any,
    *,
    duration_sec: float = MOBILE_BASE_MOVE_DURATION_SEC,
    max_speed_mps: float = MOBILE_BASE_MAX_SPEED_MPS,
) -> np.ndarray:
    """Convert a desired small x/y step into a bounded SE(2) velocity."""
    duration = max(1e-6, float(duration_sec))
    velocity = np.asarray(step_xy_m, dtype=np.float64) / duration
    speed = float(np.linalg.norm(velocity))
    if speed > float(max_speed_mps) > 0.0:
        velocity *= float(max_speed_mps) / speed
    return velocity


def mobile_base_move_plan(
    step_xy_m: Any,
    *,
    base_duration_sec: float = MOBILE_BASE_MOVE_DURATION_SEC,
    max_speed_mps: float = MOBILE_BASE_MAX_SPEED_MPS,
) -> tuple[np.ndarray, float]:
    """Return (velocity_xy, duration) whose displacement equals the planned step.

    With a fixed duration the speed cap silently truncates larger steps
    (v = step/t gets clipped, so a 3 cm step at 0.02 m/s moved only 2 cm);
    extending the duration instead keeps velocity*duration == step.
    """
    step = np.asarray(step_xy_m, dtype=np.float64)
    distance = float(np.linalg.norm(step))
    duration = max(float(base_duration_sec), distance / max(1e-6, float(max_speed_mps)))
    velocity = step / max(1e-6, duration)
    return velocity, duration


def signed_angle_error_mod_180_deg(current_deg: float, target_deg: float) -> float:
    """Return signed current-target angle error in [-90, 90] modulo 180."""
    error = (float(current_deg) - float(target_deg) + 90.0) % 180.0 - 90.0
    if error <= -90.0:
        error += 180.0
    return float(error)


def yaw_deg_mod_180_from_axis_xy(axis_xy: Any) -> float | None:
    """Return axis yaw in degrees modulo 180, or None when the axis is invalid."""
    axis = np.asarray(axis_xy, dtype=np.float64).reshape(-1)
    if axis.shape[0] < 2:
        return None
    xy = axis[:2]
    norm = float(np.linalg.norm(xy))
    if norm <= 1e-9 or not np.all(np.isfinite(xy)):
        return None
    yaw = float(np.degrees(np.arctan2(float(xy[1]), float(xy[0]))) % 180.0)
    return yaw


def box_long_axis_base_yaw_error_deg(
    long_axis_base: Any,
    *,
    target_deg: float = MOBILE_BASE_YAW_TARGET_DEG,
) -> float | None:
    """Return signed error from box long axis to base y, modulo 180.

    Positive error means the box long axis appears rotated CCW past base y in
    the current base frame. For a fixed object, commanding positive mobile-base
    yaw reduces that measured error.
    """
    yaw = yaw_deg_mod_180_from_axis_xy(long_axis_base)
    if yaw is None:
        return None
    return signed_angle_error_mod_180_deg(yaw, target_deg)


def measurement_long_axis_base(measurement: dict[str, Any]) -> np.ndarray | None:
    """Return the measured box long axis in base frame, if available."""
    long_axis_camera = measurement.get("long_axis_camera")
    camera_to_base = measurement.get("camera_to_base")
    if long_axis_camera is None or camera_to_base is None:
        return None
    axis = np.asarray(camera_to_base, dtype=np.float64)[:3, :3] @ np.asarray(
        long_axis_camera,
        dtype=np.float64,
    )
    if not np.all(np.isfinite(axis)):
        return None
    return axis


def mobile_base_yaw_alignment_error_deg(measurement: dict[str, Any]) -> float | None:
    """Return signed yaw error to align box long axis with base y."""
    axis_base = measurement_long_axis_base(measurement)
    if axis_base is None:
        return None
    return box_long_axis_base_yaw_error_deg(axis_base)


def is_mobile_base_yaw_aligned(
    measurement: dict[str, Any],
    *,
    tolerance_deg: float = MOBILE_BASE_YAW_TOLERANCE_DEG,
) -> bool:
    """Return True when the box long axis is within the yaw target band."""
    error = mobile_base_yaw_alignment_error_deg(measurement)
    return error is not None and abs(float(error)) <= float(tolerance_deg) + 1e-9


def verify_yaw_safe_before_contact(
    measurement: dict[str, Any],
    *,
    tolerance_deg: float = MOBILE_BASE_YAW_TOLERANCE_DEG,
    stage: str = "pre_contact_yaw_gate",
) -> bool:
    """Return True only when measured box yaw is safe for the side grasp."""
    yaw_error = mobile_base_yaw_alignment_error_deg(measurement)
    if yaw_error is None:
        print_stage(
            stage,
            "FAILED: no measured long-axis yaw; refusing contact motion",
        )
        return False
    if abs(float(yaw_error)) > float(tolerance_deg) + 1e-9:
        print_stage(
            stage,
            "FAILED: measured yaw error "
            f"{yaw_error:+.2f}deg exceeds {float(tolerance_deg):.1f}deg; "
            "refusing contact motion",
        )
        return False
    print_stage(
        stage,
        f"ok: measured yaw error {yaw_error:+.2f}deg <= {float(tolerance_deg):.1f}deg",
    )
    return True


def mobile_base_yaw_alignment_step_deg(
    yaw_error_deg: float,
    *,
    tolerance_deg: float = MOBILE_BASE_YAW_TOLERANCE_DEG,
    max_step_deg: float = MOBILE_BASE_YAW_MAX_STEP_DEG,
) -> float:
    """Return one conservative signed yaw correction step in degrees."""
    error = float(yaw_error_deg)
    if abs(error) <= float(tolerance_deg) + 1e-9:
        return 0.0
    return float(np.clip(error, -float(max_step_deg), float(max_step_deg)))


def mobile_base_yaw_move_plan(
    step_deg: float,
    *,
    base_duration_sec: float = MOBILE_BASE_YAW_MOVE_DURATION_SEC,
    max_angular_speed_radps: float = MOBILE_BASE_YAW_MAX_SPEED_RADPS,
) -> tuple[float, float]:
    """Return (angular_velocity_radps, duration) for a signed yaw step."""
    step_rad = float(np.deg2rad(float(step_deg)))
    duration = max(
        float(base_duration_sec),
        abs(step_rad) / max(1e-6, float(max_angular_speed_radps)),
    )
    return step_rad / max(1e-6, duration), duration


def is_mobile_base_se2_aligned(
    measurement: dict[str, Any],
    *,
    target_x_m: float = MOBILE_BASE_TARGET_X_M,
    x_tolerance_m: float = MOBILE_BASE_X_TOLERANCE_M,
    y_tolerance_m: float = MOBILE_BASE_Y_TOLERANCE_M,
    yaw_tolerance_deg: float = MOBILE_BASE_YAW_TOLERANCE_DEG,
) -> bool:
    """Return True when both box center and long-axis yaw are in the target band."""
    return is_mobile_base_aligned(
        measurement["center_base_m"],
        target_x_m=target_x_m,
        x_tolerance_m=x_tolerance_m,
        y_tolerance_m=y_tolerance_m,
    ) and is_mobile_base_yaw_aligned(
        measurement,
        tolerance_deg=yaw_tolerance_deg,
    )


def mobile_base_combined_move_plan(
    step_xy_m: Any,
    step_yaw_deg: float,
    *,
    xy_base_duration_sec: float = MOBILE_BASE_MOVE_DURATION_SEC,
    yaw_base_duration_sec: float = MOBILE_BASE_YAW_MOVE_DURATION_SEC,
    max_speed_mps: float = MOBILE_BASE_MAX_SPEED_MPS,
    max_angular_speed_radps: float = MOBILE_BASE_YAW_MAX_SPEED_RADPS,
) -> tuple[np.ndarray, float, float]:
    """Return a bounded SE(2) velocity and duration for one combined step."""
    step_xy = np.asarray(step_xy_m, dtype=np.float64)
    step_yaw_rad = float(np.deg2rad(float(step_yaw_deg)))
    xy_duration = float(np.linalg.norm(step_xy)) / max(1e-6, float(max_speed_mps))
    yaw_duration = abs(step_yaw_rad) / max(1e-6, float(max_angular_speed_radps))
    duration = max(
        float(xy_base_duration_sec),
        float(yaw_base_duration_sec),
        xy_duration,
        yaw_duration,
    )
    duration = max(1e-6, duration)
    return step_xy / duration, step_yaw_rad / duration, duration


def mobile_base_combined_alignment_plan(
    measurement: dict[str, Any],
    *,
    target_x_m: float = MOBILE_BASE_TARGET_X_M,
    x_tolerance_m: float = MOBILE_BASE_X_TOLERANCE_M,
    y_tolerance_m: float = MOBILE_BASE_Y_TOLERANCE_M,
    yaw_tolerance_deg: float = MOBILE_BASE_YAW_TOLERANCE_DEG,
    max_step_m: float = MOBILE_BASE_MAX_STEP_M,
    max_step_deg: float = MOBILE_BASE_YAW_MAX_STEP_DEG,
    max_speed_mps: float = MOBILE_BASE_MAX_SPEED_MPS,
    max_angular_speed_radps: float = MOBILE_BASE_YAW_MAX_SPEED_RADPS,
    xy_move_duration_sec: float = MOBILE_BASE_MOVE_DURATION_SEC,
    yaw_move_duration_sec: float = MOBILE_BASE_YAW_MOVE_DURATION_SEC,
    coarse_yaw_threshold_deg: float = MOBILE_BASE_COMBINED_COARSE_YAW_THRESHOLD_DEG,
) -> dict[str, Any] | None:
    """Plan one conservative combined yaw/x/y correction from a fresh measurement.

    By default x/y remains enabled across the full modulo-180 yaw-error range.
    Lower coarse_yaw_threshold_deg only when you deliberately want a yaw-first
    debug mode.
    """
    yaw_error = mobile_base_yaw_alignment_error_deg(measurement)
    if yaw_error is None:
        return None

    center_base = np.asarray(measurement["center_base_m"], dtype=np.float64)
    error_xy = mobile_base_alignment_error_xy(center_base, target_x_m=target_x_m)
    step_xy = mobile_base_alignment_step_xy(
        center_base,
        target_x_m=target_x_m,
        x_tolerance_m=x_tolerance_m,
        y_tolerance_m=y_tolerance_m,
        max_step_m=max_step_m,
    )
    translation_enabled = abs(float(yaw_error)) <= float(coarse_yaw_threshold_deg) + 1e-9
    if not translation_enabled:
        step_xy = np.zeros(2, dtype=np.float64)

    step_yaw_deg = mobile_base_yaw_alignment_step_deg(
        yaw_error,
        tolerance_deg=yaw_tolerance_deg,
        max_step_deg=max_step_deg,
    )
    velocity_xy, angular_velocity, duration = mobile_base_combined_move_plan(
        step_xy,
        step_yaw_deg,
        xy_base_duration_sec=xy_move_duration_sec,
        yaw_base_duration_sec=yaw_move_duration_sec,
        max_speed_mps=max_speed_mps,
        max_angular_speed_radps=max_angular_speed_radps,
    )
    return {
        "center_base_m": center_base,
        "error_xy_m": error_xy,
        "yaw_error_deg": float(yaw_error),
        "translation_enabled": bool(translation_enabled),
        "step_xy_m": step_xy,
        "step_yaw_deg": float(step_yaw_deg),
        "velocity_xy_mps": velocity_xy,
        "angular_velocity_radps": float(angular_velocity),
        "duration_sec": float(duration),
        "aligned": is_mobile_base_se2_aligned(
            measurement,
            target_x_m=target_x_m,
            x_tolerance_m=x_tolerance_m,
            y_tolerance_m=y_tolerance_m,
            yaw_tolerance_deg=yaw_tolerance_deg,
        ),
    }


class UserAbortRequested(Exception):
    """Operator requested abort from the live vision window."""


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


def build_mobile_base_velocity_command(
    linear_velocity_xy_mps: Any,
    *,
    angular_velocity_radps: float = 0.0,
    minimum_time: float,
    control_hold_time: float = MOBILE_BASE_COMMAND_HOLD_TIME_SEC,
) -> Any:
    """Build an M-model body-frame SE(2) velocity command."""
    velocity = np.asarray(linear_velocity_xy_mps, dtype=np.float64)
    return rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_mobility_command(
            rby.SE2VelocityCommandBuilder()
            .set_command_header(
                rby.CommandHeaderBuilder().set_control_hold_time(float(control_hold_time))
            )
            .set_minimum_time(float(minimum_time))
            .set_velocity(velocity, float(angular_velocity_radps))
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
# Local recorded joint sets [rad] for picking_box_3.
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


def select_stable_live_center_result(
    centers: list[Any],
    long_axes: list[Any | None],
    modes: list[str],
    long_axis_unconstrained_flags: list[bool],
    *,
    frames_needed: int,
    max_center_spread_m: float,
) -> tuple[dict[str, Any] | None, float | None]:
    """Return the tightest stable center cluster from live candidates.

    D405 depth and partial-crop fits occasionally produce isolated center
    outliers. Contact planning should reject genuinely unstable vision, but it
    should not fail just because one candidate jumped while enough nearby
    candidates exist in the same timeout window.
    """
    needed = int(frames_needed)
    if len(centers) < needed:
        return None, None

    center_array = np.asarray(centers, dtype=np.float64)
    best_indices: np.ndarray | None = None
    best_spread = float("inf")
    for anchor in center_array:
        distances = np.linalg.norm(center_array - anchor, axis=1)
        indices = np.argsort(distances)[:needed]
        cluster = center_array[indices]
        cluster_median = np.median(cluster, axis=0)
        spread = float(np.max(np.linalg.norm(cluster - cluster_median, axis=1)))
        if spread < best_spread:
            best_spread = spread
            best_indices = indices

    if best_indices is None:
        return None, None
    if best_spread > float(max_center_spread_m):
        return None, best_spread

    selected_centers = center_array[best_indices]
    selected_axes = [
        np.asarray(long_axes[int(i)], dtype=np.float64)
        for i in best_indices
        if long_axes[int(i)] is not None
    ]
    return {
        "center_camera_m": np.median(selected_centers, axis=0),
        "center_spread_m": best_spread,
        "long_axis_camera": None if not selected_axes else np.median(np.asarray(selected_axes), axis=0),
        "long_axis_unconstrained": any(long_axis_unconstrained_flags[int(i)] for i in best_indices),
        "frames_used": needed,
        "candidate_frames": len(centers),
        "modes": [modes[int(i)] for i in best_indices],
    }, best_spread


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

    Runs the rim-plane estimator with the stored plane calibration and prefers
    confidence-ok frames. Frames whose only issue is an underconstrained
    long-axis center are also accepted as center-only candidates and checked by
    a multi-frame center-spread gate. The optional long-axis direction is still
    retained for the mobile-base yaw safety gate when available. Returns the
    per-axis median center (analysis camera frame), optional long-axis
    direction, and frame count, or None when not enough usable frames arrive
    within the timeout.

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
    window_name = "picking_box_3 vision (q/ESC to close)"
    centers: list[Any] = []
    long_axes: list[Any | None] = []
    modes: list[str] = []
    long_axis_unconstrained_flags: list[bool] = []
    quit_requested = False
    continue_requested = False
    stable_result: dict[str, Any] | None = None
    best_unstable_spread_m: float | None = None
    deadline = time.monotonic() + (float("inf") if run_forever else float(timeout_sec))
    frames = iter_live_frames(
        width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS, view_rotation=view_rotation
    )
    try:
        for _, image, depth_m, intrinsics in frames:
            captured_enough = stable_result is not None
            if captured_enough and not (visualize and hold_after_capture):
                break
            if time.monotonic() >= deadline and not (captured_enough and visualize and hold_after_capture):
                break
            mask, _ = segment_yellow_box(image, keep_largest_component=False)
            estimate = estimate_plane_box(mask, depth_m, intrinsics, rim_plane=rim_plane)
            usable, mode = live_estimate_center_mode(estimate)
            collecting = not run_forever and stable_result is None
            if collecting:
                if usable:
                    centers.append(estimate.center_top_camera_m)
                    modes.append(mode)
                    axis = estimate.support.get("long_axis_camera")
                    long_axes.append(axis)
                    axis_unconstrained = False
                    if not estimate.confidence.ok:
                        # The long-axis center of this frame is a visible-span
                        # midpoint, not a measurement; remember to drop that
                        # axis from the commanded correction.
                        if "long_axis_center_underconstrained" in estimate.failure_reasons:
                            axis_unconstrained = True
                        if len(centers) == 1 or len(centers) % 3 == 0:
                            print(f"[vision] accepted center-only frame: {mode}")
                    long_axis_unconstrained_flags.append(axis_unconstrained)
                    stable_result, best_unstable_spread_m = select_stable_live_center_result(
                        centers,
                        long_axes,
                        modes,
                        long_axis_unconstrained_flags,
                        frames_needed=frames_needed,
                        max_center_spread_m=max_center_spread_m,
                    )
                else:
                    print(f"[vision] frame rejected: {mode}")

            if visualize:
                vis = draw_known_size_estimate(image.copy(), estimate)
                status_color = (30, 160, 30) if usable else (40, 40, 220)
                lines = [(f"{'USABLE' if usable else 'REJECTED'}: {mode}", status_color)]
                if not run_forever and stable_result is not None:
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
                if stable_result is not None and key in (ord("c"), ord(" "), 10, 13):
                    continue_requested = True
                    break
            elif stable_result is not None:
                break
    finally:
        frames.close()  # stops the RealSense pipeline (generator finally)
        if visualize:
            cv2.destroyAllWindows()

    if run_forever or quit_requested:
        return None
    if visualize and hold_after_capture and not continue_requested and stable_result is not None:
        return None

    if stable_result is None:
        if len(centers) >= int(frames_needed) and best_unstable_spread_m is not None:
            print(
                "[vision] FAILED: live center spread too large "
                f"({best_unstable_spread_m:.3f} m > {max_center_spread_m:.3f} m) "
                f"after {len(centers)} usable candidates."
            )
        return None

    return stable_result


def capture_live_box_measurement(
    robot: Any,
    dyn_model: Any,
    dyn_state: Any,
    view_rotation: str,
    *,
    frames_needed: int,
    timeout_sec: float,
    max_center_spread_m: float,
    visualize: bool,
    hold_after_capture: bool,
) -> dict[str, Any] | None:
    """Capture one live box measurement and express its center in base."""
    q = robot.get_state().position
    camera_to_base = compute_camera_to_base_for_view_rotation(
        view_rotation,
        dyn_model,
        dyn_state,
        q,
    )
    live = capture_box_live(
        view_rotation,
        frames_needed=frames_needed,
        timeout_sec=timeout_sec,
        max_center_spread_m=max_center_spread_m,
        visualize=visualize,
        hold_after_capture=hold_after_capture,
        camera_to_base=camera_to_base,
    )
    if live is None:
        return None

    return live_result_to_measurement(robot, dyn_model, dyn_state, view_rotation, live)


def live_result_to_measurement(
    robot: Any,
    dyn_model: Any,
    dyn_state: Any,
    view_rotation: str,
    live: dict[str, Any],
) -> dict[str, Any]:
    """Attach current FK-derived camera->base data to a live vision result."""
    q = robot.get_state().position
    camera_to_base = compute_camera_to_base_for_view_rotation(
        view_rotation,
        dyn_model,
        dyn_state,
        q,
    )
    center_camera = np.asarray(live["center_camera_m"], dtype=np.float64)
    center_base = transform_camera_point_to_base(center_camera, camera_to_base)
    long_axis_camera = live["long_axis_camera"]
    drop_delta_axis_base_xy = None
    if live["long_axis_unconstrained"] and long_axis_camera is not None:
        axis_base = np.asarray(camera_to_base, dtype=np.float64)[:3, :3] @ np.asarray(
            long_axis_camera, dtype=np.float64
        )
        drop_delta_axis_base_xy = axis_base[:2]

    return {
        **live,
        "q": q,
        "camera_to_base": camera_to_base,
        "center_camera_m": center_camera,
        "center_base_m": center_base,
        "drop_delta_axis_base_xy": drop_delta_axis_base_xy,
    }


def live_estimate_to_measurement(
    robot: Any,
    dyn_model: Any,
    dyn_state: Any,
    view_rotation: str,
    estimate: Any,
    mode: str,
) -> dict[str, Any]:
    """Convert one usable live estimate into a base-frame measurement."""
    axis = estimate.support.get("long_axis_camera")
    axis_unconstrained = (
        not estimate.confidence.ok
        and "long_axis_center_underconstrained" in estimate.failure_reasons
    )
    live = {
        "center_camera_m": estimate.center_top_camera_m,
        "center_spread_m": 0.0,
        "long_axis_camera": axis,
        "long_axis_unconstrained": axis_unconstrained,
        "frames_used": 1,
        "candidate_frames": 1,
        "modes": [mode],
    }
    return live_result_to_measurement(robot, dyn_model, dyn_state, view_rotation, live)


class ContinuousLiveBoxView:
    """Keep the D405 vision window alive during closed-loop base alignment."""

    def __init__(self, view_rotation: str) -> None:
        import cv2

        if str(BOX_PERCEPTION_ROOT) not in sys.path:
            sys.path.insert(0, str(BOX_PERCEPTION_ROOT))
        from box_pose import estimate_plane_box, segment_yellow_box
        from box_pose.visualization import draw_known_size_estimate
        from inference_2 import iter_live_frames

        self.cv2 = cv2
        self.cv2.setNumThreads(1)
        self.estimate_plane_box = estimate_plane_box
        self.segment_yellow_box = segment_yellow_box
        self.draw_known_size_estimate = draw_known_size_estimate
        rim_cfg = json.loads(RIM_PLANE_CONFIG.read_text(encoding="utf-8"))
        self.rim_plane = (rim_cfg["normal"], rim_cfg["point"])
        self.frames = iter_live_frames(
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=CAMERA_FPS,
            view_rotation=view_rotation,
        )
        self.window_name = "picking_box_5 mobile align vision (q/ESC abort)"
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.frames.close()
        finally:
            try:
                self.cv2.destroyWindow(self.window_name)
            except Exception:
                pass

    def process_next_frame(
        self,
        *,
        camera_to_base: Any,
        status_lines: list[tuple[str, tuple[int, int, int]]] | None = None,
    ) -> tuple[bool, str, Any, np.ndarray | None, bool]:
        """Process one live frame, update the window, and return estimate info."""
        _, image, depth_m, intrinsics = next(self.frames)
        mask, _ = self.segment_yellow_box(image, keep_largest_component=False)
        estimate = self.estimate_plane_box(mask, depth_m, intrinsics, rim_plane=self.rim_plane)
        usable, mode = live_estimate_center_mode(estimate)

        T_base = np.asarray(camera_to_base, dtype=np.float64)
        vis = self.draw_known_size_estimate(image.copy(), estimate)
        status_color = (30, 160, 30) if usable else (40, 40, 220)
        lines: list[tuple[str, tuple[int, int, int]]] = [
            (f"{'USABLE' if usable else 'REJECTED'}: {mode}", status_color)
        ]
        if estimate.center_top_camera_m is not None:
            center_base = (T_base @ np.append(estimate.center_top_camera_m, 1.0))[:3]
            text = f"base x={center_base[0] * 100:+.1f}cm  y={center_base[1] * 100:+.1f}cm"
            long_axis = estimate.support.get("long_axis_camera")
            if long_axis is not None:
                axis_base = T_base[:3, :3] @ np.asarray(long_axis, dtype=np.float64)
                yaw_base = float(np.degrees(np.arctan2(axis_base[1], axis_base[0])) % 180.0)
                text += f"  yaw={yaw_base:.1f}deg"
            lines.append((text, (255, 255, 255)))
        if status_lines:
            lines.extend(status_lines)
        lines.append(("mobile align live view: q/ESC abort", (0, 220, 255)))

        for i, (text, color) in enumerate(lines):
            y_pos = 34 + 32 * i
            self.cv2.putText(
                vis,
                text,
                (12, y_pos),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 0),
                4,
                self.cv2.LINE_AA,
            )
            self.cv2.putText(
                vis,
                text,
                (12, y_pos),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2,
                self.cv2.LINE_AA,
            )
        self.cv2.imshow(self.window_name, vis)
        key = self.cv2.waitKey(1) & 0xFF
        abort_requested = key in (27, ord("q"))
        return usable, mode, estimate, T_base, abort_requested

    def collect_measurement(
        self,
        robot: Any,
        dyn_model: Any,
        dyn_state: Any,
        view_rotation: str,
        *,
        frames_needed: int,
        timeout_sec: float,
        max_center_spread_m: float,
        stage: str = "mobile_base_align",
        status_lines: list[tuple[str, tuple[int, int, int]]] | None = None,
    ) -> dict[str, Any] | None:
        centers: list[Any] = []
        long_axes: list[Any | None] = []
        modes: list[str] = []
        long_axis_unconstrained_flags: list[bool] = []
        deadline = time.monotonic() + float(timeout_sec)
        best_unstable_spread_m: float | None = None

        while time.monotonic() < deadline:
            if time.monotonic() >= deadline:
                break
            q = robot.get_state().position
            camera_to_base = compute_camera_to_base_for_view_rotation(
                view_rotation,
                dyn_model,
                dyn_state,
                q,
            )
            try:
                usable, mode, estimate, _, abort_requested = self.process_next_frame(
                    camera_to_base=camera_to_base,
                    status_lines=[
                        (
                            f"collecting {len(centers)}/{int(frames_needed)} usable frames",
                            (255, 255, 255),
                        ),
                        *(status_lines or []),
                    ],
                )
            except StopIteration:
                return None
            if abort_requested:
                print_stage(stage, "visualization abort requested")
                raise UserAbortRequested(f"{stage} recapture aborted from the live view")
            if usable:
                centers.append(estimate.center_top_camera_m)
                modes.append(mode)
                axis = estimate.support.get("long_axis_camera")
                long_axes.append(axis)
                axis_unconstrained = False
                if not estimate.confidence.ok and "long_axis_center_underconstrained" in estimate.failure_reasons:
                    axis_unconstrained = True
                long_axis_unconstrained_flags.append(axis_unconstrained)
                live, best_unstable_spread_m = select_stable_live_center_result(
                    centers,
                    long_axes,
                    modes,
                    long_axis_unconstrained_flags,
                    frames_needed=frames_needed,
                    max_center_spread_m=max_center_spread_m,
                )
                if live is not None:
                    return live_result_to_measurement(robot, dyn_model, dyn_state, view_rotation, live)
            else:
                print(f"[vision] frame rejected: {mode}")

        if len(centers) >= int(frames_needed) and best_unstable_spread_m is not None:
            print(
                "[vision] FAILED: live center spread too large "
                f"({best_unstable_spread_m:.3f} m > {max_center_spread_m:.3f} m) "
                f"after {len(centers)} usable candidates."
            )
        return None


def wait_for_mobile_command_with_live_view(
    robot: Any,
    command: Any,
    stage: str,
    *,
    timeout_sec: float,
    live_view: ContinuousLiveBoxView,
    dyn_model: Any,
    dyn_state: Any,
    view_rotation: str,
    status_lines: list[tuple[str, tuple[int, int, int]]] | None = None,
) -> tuple[Any | None, str]:
    """Wait for a mobility command while continuously refreshing vision."""
    deadline = time.monotonic() + float(timeout_sec)
    next_log = time.monotonic() + COMMAND_WAIT_LOG_INTERVAL_SEC
    while True:
        if command.wait_for(1):
            return command.get(), "done"
        now = time.monotonic()
        if now >= deadline:
            return None, "timeout"
        q = robot.get_state().position
        camera_to_base = compute_camera_to_base_for_view_rotation(
            view_rotation,
            dyn_model,
            dyn_state,
            q,
        )
        try:
            _, _, _, _, abort_requested = live_view.process_next_frame(
                camera_to_base=camera_to_base,
                status_lines=status_lines,
            )
        except StopIteration:
            print_stage(stage, "live vision stream ended while waiting for mobility command")
            return None, "stream_ended"
        if abort_requested:
            print_stage(stage, "visualization abort requested")
            return None, "visual_abort"
        now = time.monotonic()
        if now >= next_log:
            elapsed = max(0.0, float(timeout_sec) - max(0.0, deadline - now))
            print_stage(stage, f"still waiting ({elapsed:.1f}/{float(timeout_sec):.1f}s)")
            next_log = now + COMMAND_WAIT_LOG_INTERVAL_SEC


def send_mobile_stage_with_live_view(
    robot: Any,
    builder: Any,
    stage: str,
    *,
    timeout_sec: float,
    live_view: ContinuousLiveBoxView,
    dyn_model: Any,
    dyn_state: Any,
    view_rotation: str,
    status_lines: list[tuple[str, tuple[int, int, int]]] | None = None,
) -> tuple[bool, str]:
    """Send a mobile-base command and keep the vision overlay alive.

    Returns (ok, status) so the caller can tell an operator abort
    ("visual_abort") apart from a timeout or a not-ok finish code.
    """
    print_stage(stage, "sending command")
    command = robot.send_command(builder)
    print_stage(stage, f"waiting for finish_code with live vision (timeout={float(timeout_sec):.1f}s)")
    feedback, wait_status = wait_for_mobile_command_with_live_view(
        robot,
        command,
        stage,
        timeout_sec=float(timeout_sec),
        live_view=live_view,
        dyn_model=dyn_model,
        dyn_state=dyn_state,
        view_rotation=view_rotation,
        status_lines=status_lines,
    )
    if feedback is None:
        if wait_status == "timeout":
            cancel_timed_out_command(robot, command, stage)
        elif wait_status == "visual_abort":
            cancel_active_command(robot, command, stage, reason="visualization abort requested")
        else:
            cancel_active_command(robot, command, stage, reason=f"live vision wait ended: {wait_status}")
        return False, wait_status
    print_stage(stage, f"finish_code={feedback.finish_code}")
    ok = feedback.finish_code == rby.RobotCommandFeedback.FinishCode.Ok
    return ok, "done" if ok else "finish_not_ok"


def stream_mobile_base_velocity_stage(
    robot: Any,
    linear_velocity_xy_mps: Any,
    *,
    angular_velocity_radps: float = 0.0,
    duration_sec: float,
    stage: str,
    live_view: ContinuousLiveBoxView | None,
    dyn_model: Any,
    dyn_state: Any,
    view_rotation: str,
    status_lines: list[tuple[str, tuple[int, int, int]]] | None = None,
    stream_period_sec: float = MOBILE_BASE_STREAM_PERIOD_SEC,
    control_hold_time_sec: float = MOBILE_BASE_COMMAND_HOLD_TIME_SEC,
) -> tuple[bool, str]:
    """Stream a body-frame SE2 velocity command like rby1-lerobot send_action().

    rby1-lerobot does not run the wheel joints directly. It repeatedly sends
    x.vel/y.vel/theta.vel through a priority-1 command stream. Use the same
    pattern for short mobile-base alignment moves, then explicitly stream zero
    velocity so the base stops before the next vision recapture.
    """
    velocity_xy = np.asarray(linear_velocity_xy_mps, dtype=np.float64)
    angular_velocity = float(angular_velocity_radps)
    period = max(0.01, float(stream_period_sec))
    duration = max(0.0, float(duration_sec))
    minimum_time = period * 1.01

    print_stage(
        stage,
        "streaming body-frame SE2 velocity "
        f"vel=[{velocity_xy[0]:+.3f}, {velocity_xy[1]:+.3f}]m/s "
        f"omega={angular_velocity:+.3f}rad/s "
        f"duration={duration:.2f}s period={period:.3f}s",
    )
    try:
        stream = robot.create_command_stream(priority=MOBILE_BASE_STREAM_PRIORITY)
    except Exception as exc:  # pragma: no cover - SDK/hardware dependent
        print_stage(stage, f"failed to create mobile command stream: {exc}")
        return False, "stream_create_failed"

    def send_velocity(v_xy: Any, omega: float) -> None:
        stream.send_command(
            build_mobile_base_velocity_command(
                v_xy,
                angular_velocity_radps=float(omega),
                minimum_time=minimum_time,
                control_hold_time=control_hold_time_sec,
            )
        )

    def refresh_live_view() -> str | None:
        if live_view is None:
            return None
        q = robot.get_state().position
        camera_to_base = compute_camera_to_base_for_view_rotation(
            view_rotation,
            dyn_model,
            dyn_state,
            q,
        )
        try:
            _, _, _, _, abort_requested = live_view.process_next_frame(
                camera_to_base=camera_to_base,
                status_lines=status_lines,
            )
        except StopIteration:
            print_stage(stage, "live vision stream ended while streaming mobility command")
            return "stream_ended"
        if abort_requested:
            print_stage(stage, "visualization abort requested")
            return "visual_abort"
        return None

    status = "done"
    ok = True
    sends = 0
    try:
        start = time.monotonic()
        deadline = start + duration
        next_send = start
        next_log = start + COMMAND_WAIT_LOG_INTERVAL_SEC
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_send:
                send_velocity(velocity_xy, angular_velocity)
                sends += 1
                next_send = now + period

            live_status = refresh_live_view()
            if live_status is not None:
                ok = False
                status = live_status
                break

            now = time.monotonic()
            if now >= next_log:
                elapsed = min(duration, max(0.0, now - start))
                print_stage(stage, f"streaming ({elapsed:.1f}/{duration:.1f}s, sends={sends})")
                next_log = now + COMMAND_WAIT_LOG_INTERVAL_SEC
            sleep_sec = min(0.005, max(0.0, min(next_send, deadline) - time.monotonic()))
            if sleep_sec > 0.0:
                time.sleep(sleep_sec)
    finally:
        zero = np.zeros(2, dtype=np.float64)
        for _ in range(max(1, int(MOBILE_BASE_STOP_REPEATS))):
            send_velocity(zero, 0.0)
            time.sleep(period)
        print_stage(stage, f"stream stop sent; motion sends={sends}")

    return ok, status


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


def cancel_active_command(robot: Any, command: Any, stage: str, *, reason: str) -> None:
    """Best-effort cancellation for an active command."""
    print_stage(stage, f"{reason}; canceling command")
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


def cancel_timed_out_command(robot: Any, command: Any, stage: str) -> None:
    """Best-effort cancellation after a stage watchdog timeout."""
    cancel_active_command(robot, command, stage, reason="TIMEOUT")


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


def run_mobile_base_yaw_alignment(
    robot: Any,
    dyn_model: Any,
    dyn_state: Any,
    view_rotation: str,
    measurement: dict[str, Any],
    *,
    visualize: bool,
    tolerance_deg: float,
    max_angular_speed_radps: float,
    max_step_deg: float,
    max_iterations: int,
    total_timeout_sec: float,
    move_duration_sec: float,
    vision_frames_needed: int,
    vision_timeout_sec: float,
    max_center_spread_m: float,
    stage: str = "3/7 mobile_base_yaw_align",
) -> dict[str, Any] | None:
    """Closed-loop mobile-base theta alignment before x/y and arm contact."""
    deadline = time.monotonic() + float(total_timeout_sec)
    latest = measurement
    live_view: ContinuousLiveBoxView | None = ContinuousLiveBoxView(view_rotation) if visualize else None

    try:
        for iteration in range(1, int(max_iterations) + 1):
            yaw_error = mobile_base_yaw_alignment_error_deg(latest)
            if yaw_error is None:
                print_stage(stage, "FAILED: no usable long-axis yaw in latest vision measurement")
                return None

            if abs(float(yaw_error)) <= float(tolerance_deg) + 1e-9:
                print_stage(
                    stage,
                    f"aligned at iter={iteration - 1}; yaw_error={yaw_error:+.2f}deg "
                    f"(target <= {float(tolerance_deg):.1f}deg)",
                )
                return latest

            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                print_stage(stage, "timeout; returning latest vision measurement for residual gate")
                return latest

            step_deg = mobile_base_yaw_alignment_step_deg(
                yaw_error,
                tolerance_deg=tolerance_deg,
                max_step_deg=max_step_deg,
            )
            angular_velocity, planned_duration = mobile_base_yaw_move_plan(
                step_deg,
                base_duration_sec=move_duration_sec,
                max_angular_speed_radps=max_angular_speed_radps,
            )
            if abs(float(angular_velocity)) <= 1e-6:
                print_stage(stage, "zero yaw correction; returning latest vision measurement")
                return latest

            move_time = min(float(planned_duration), max(0.1, remaining))
            status_lines = [
                (
                    f"yaw iter {iteration}/{int(max_iterations)} target long-axis -> base y",
                    (255, 255, 255),
                ),
                (
                    f"yaw error={yaw_error:+.1f}deg step={step_deg:+.1f}deg "
                    f"omega={angular_velocity:+.3f}rad/s",
                    (0, 220, 255),
                ),
            ]
            print_stage(
                stage,
                f"iter={iteration}/{int(max_iterations)} "
                f"yaw_error={yaw_error:+.2f}deg step={step_deg:+.2f}deg "
                f"omega={angular_velocity:+.3f}rad/s duration={move_time:.2f}s",
            )
            ok, move_status = stream_mobile_base_velocity_stage(
                robot,
                np.zeros(2, dtype=np.float64),
                angular_velocity_radps=angular_velocity,
                duration_sec=move_time,
                stage=stage,
                live_view=live_view,
                dyn_model=dyn_model,
                dyn_state=dyn_state,
                view_rotation=view_rotation,
                status_lines=status_lines,
            )
            if not ok:
                if move_status == "visual_abort":
                    raise UserAbortRequested("mobile yaw alignment aborted from the live view")
                print_stage(stage, f"yaw command not ok ({move_status}); re-measuring before deciding")

            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                print_stage(stage, "timeout after yaw move; recapture required")
                return None
            measure_timeout = min(float(vision_timeout_sec), max(0.1, remaining))
            if live_view is not None:
                refreshed = live_view.collect_measurement(
                    robot,
                    dyn_model,
                    dyn_state,
                    view_rotation,
                    frames_needed=vision_frames_needed,
                    timeout_sec=measure_timeout,
                    max_center_spread_m=max_center_spread_m,
                    stage=stage,
                    status_lines=status_lines,
                )
            else:
                refreshed = capture_live_box_measurement(
                    robot,
                    dyn_model,
                    dyn_state,
                    view_rotation,
                    frames_needed=vision_frames_needed,
                    timeout_sec=measure_timeout,
                    max_center_spread_m=max_center_spread_m,
                    visualize=False,
                    hold_after_capture=False,
                )
            if refreshed is None:
                print_stage(stage, "post-yaw vision failed; aborting before contact")
                return None
            latest = refreshed

        yaw_error = mobile_base_yaw_alignment_error_deg(latest)
        if yaw_error is None:
            print_stage(stage, "max iterations reached with no usable yaw")
        else:
            print_stage(stage, f"max iterations reached; yaw_error={yaw_error:+.2f}deg")
        return latest
    finally:
        if live_view is not None:
            live_view.close()


def run_mobile_base_alignment(
    robot: Any,
    dyn_model: Any,
    dyn_state: Any,
    view_rotation: str,
    measurement: dict[str, Any],
    *,
    visualize: bool,
    target_x_m: float,
    x_tolerance_m: float,
    y_tolerance_m: float,
    max_speed_mps: float,
    max_step_m: float,
    max_iterations: int,
    total_timeout_sec: float,
    move_duration_sec: float,
    vision_frames_needed: int,
    vision_timeout_sec: float,
    max_center_spread_m: float,
    min_command_timeout_sec: float,
    command_timeout_margin_sec: float,
    stage: str = "4/7 mobile_base_xy_align",
) -> dict[str, Any] | None:
    """Closed-loop mobile-base x/y alignment before arm vision_pre_push.

    Returns the latest fresh measurement. If the target band cannot be reached
    but the box is still visible, the caller should continue with vision_pre_push
    using that latest measurement. If a post-move recapture fails, returns None
    so the caller can do one normal live-vision recovery attempt.
    """
    deadline = time.monotonic() + float(total_timeout_sec)
    latest = measurement
    live_view: ContinuousLiveBoxView | None = ContinuousLiveBoxView(view_rotation) if visualize else None

    try:
        for iteration in range(1, int(max_iterations) + 1):
            center_base = np.asarray(latest["center_base_m"], dtype=np.float64)
            error_xy = mobile_base_alignment_error_xy(center_base, target_x_m=target_x_m)
            if is_mobile_base_aligned(
                center_base,
                target_x_m=target_x_m,
                x_tolerance_m=x_tolerance_m,
                y_tolerance_m=y_tolerance_m,
            ):
                print_stage(
                    stage,
                    f"aligned at iter={iteration - 1}; "
                    f"x={center_base[0]:+.3f}m y={center_base[1]:+.3f}m "
                    f"error=[{error_xy[0]:+.3f}, {error_xy[1]:+.3f}]m",
                )
                return latest

            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                print_stage(stage, "timeout; continuing with latest vision measurement")
                return latest

            step_xy = mobile_base_alignment_step_xy(
                center_base,
                target_x_m=target_x_m,
                x_tolerance_m=x_tolerance_m,
                y_tolerance_m=y_tolerance_m,
                max_step_m=max_step_m,
            )
            velocity_xy, planned_duration = mobile_base_move_plan(
                step_xy,
                base_duration_sec=move_duration_sec,
                max_speed_mps=max_speed_mps,
            )
            if float(np.linalg.norm(velocity_xy)) <= 1e-6:
                print_stage(stage, "zero correction; continuing with latest vision measurement")
                return latest

            move_time = min(float(planned_duration), max(0.1, remaining))
            status_lines = [
                (
                    f"align iter {iteration}/{int(max_iterations)}  "
                    f"target x={target_x_m * 100:+.1f}cm y=+0.0cm",
                    (255, 255, 255),
                ),
                (
                    f"error x={error_xy[0] * 100:+.1f}cm y={error_xy[1] * 100:+.1f}cm  "
                    f"vel x={velocity_xy[0] * 100:+.1f}cm/s y={velocity_xy[1] * 100:+.1f}cm/s",
                    (0, 220, 255),
                ),
            ]
            print_stage(
                stage,
                f"iter={iteration}/{int(max_iterations)} "
                f"center=[{center_base[0]:+.3f}, {center_base[1]:+.3f}]m "
                f"target=[{target_x_m:+.3f}, +0.000]m "
                f"step=[{step_xy[0]:+.3f}, {step_xy[1]:+.3f}]m "
                f"vel=[{velocity_xy[0]:+.3f}, {velocity_xy[1]:+.3f}]m/s",
            )
            ok, move_status = stream_mobile_base_velocity_stage(
                robot,
                velocity_xy,
                duration_sec=move_time,
                stage=stage,
                live_view=live_view,
                dyn_model=dyn_model,
                dyn_state=dyn_state,
                view_rotation=view_rotation,
                status_lines=status_lines,
            )
            if not ok:
                if move_status == "visual_abort":
                    raise UserAbortRequested("mobile alignment aborted from the live view")
                # The base may have partially moved before the cancel, so the
                # pre-move measurement is stale; fall through to the post-move
                # recapture instead of acting on it.
                print_stage(
                    stage,
                    f"mobility command not ok ({move_status}); re-measuring before deciding",
                )

            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                print_stage(stage, "timeout after move; recapture required")
                return None
            measure_timeout = min(float(vision_timeout_sec), max(0.1, remaining))
            if live_view is not None:
                refreshed = live_view.collect_measurement(
                    robot,
                    dyn_model,
                    dyn_state,
                    view_rotation,
                    frames_needed=vision_frames_needed,
                    timeout_sec=measure_timeout,
                    max_center_spread_m=max_center_spread_m,
                    stage=stage,
                    status_lines=status_lines,
                )
            else:
                refreshed = capture_live_box_measurement(
                    robot,
                    dyn_model,
                    dyn_state,
                    view_rotation,
                    frames_needed=vision_frames_needed,
                    timeout_sec=measure_timeout,
                    max_center_spread_m=max_center_spread_m,
                    visualize=False,
                    hold_after_capture=False,
                )
            if refreshed is None:
                print_stage(stage, "post-move vision failed; recapture required")
                return None
            latest = refreshed

        center_base = np.asarray(latest["center_base_m"], dtype=np.float64)
        print_stage(
            stage,
            f"max iterations reached; continuing with x={center_base[0]:+.3f}m y={center_base[1]:+.3f}m",
        )
        return latest
    finally:
        if live_view is not None:
            live_view.close()


def run_mobile_base_combined_servo_alignment(
    robot: Any,
    dyn_model: Any,
    dyn_state: Any,
    view_rotation: str,
    measurement: dict[str, Any],
    *,
    target_x_m: float,
    x_tolerance_m: float,
    y_tolerance_m: float,
    yaw_tolerance_deg: float,
    coarse_yaw_threshold_deg: float,
    max_speed_mps: float,
    max_angular_speed_radps: float,
    max_step_m: float,
    max_step_deg: float,
    total_timeout_sec: float,
    xy_move_duration_sec: float,
    yaw_move_duration_sec: float,
    vision_frames_needed: int,
    stage: str = "3-4/7 mobile_base_se2_align",
) -> dict[str, Any] | None:
    """Continuously update M-base SE(2) velocity from live vision frames."""
    initial_plan = mobile_base_combined_alignment_plan(
        measurement,
        target_x_m=target_x_m,
        x_tolerance_m=x_tolerance_m,
        y_tolerance_m=y_tolerance_m,
        yaw_tolerance_deg=yaw_tolerance_deg,
        max_step_m=max_step_m,
        max_step_deg=max_step_deg,
        max_speed_mps=max_speed_mps,
        max_angular_speed_radps=max_angular_speed_radps,
        xy_move_duration_sec=xy_move_duration_sec,
        yaw_move_duration_sec=yaw_move_duration_sec,
        coarse_yaw_threshold_deg=coarse_yaw_threshold_deg,
    )
    if initial_plan is None:
        print_stage(stage, "FAILED: no usable long-axis yaw in initial vision measurement")
        return None
    if bool(initial_plan["aligned"]):
        center_base = np.asarray(initial_plan["center_base_m"], dtype=np.float64)
        print_stage(
            stage,
            f"already aligned; x={center_base[0]:+.3f}m y={center_base[1]:+.3f}m "
            f"yaw_error={float(initial_plan['yaw_error_deg']):+.2f}deg",
        )
        return measurement

    live_view = ContinuousLiveBoxView(view_rotation)
    latest = measurement
    latest_plan = initial_plan
    velocity_xy = np.zeros(2, dtype=np.float64)
    angular_velocity = 0.0
    aligned_frames = 0
    required_aligned_frames = max(1, min(3, int(vision_frames_needed)))
    period = max(0.01, float(MOBILE_BASE_STREAM_PERIOD_SEC))
    minimum_time = period * 1.01
    deadline = time.monotonic() + float(total_timeout_sec)
    next_log = time.monotonic() + COMMAND_WAIT_LOG_INTERVAL_SEC
    sends = 0
    updates = 0

    try:
        stream = robot.create_command_stream(priority=MOBILE_BASE_STREAM_PRIORITY)
    except Exception as exc:  # pragma: no cover - SDK/hardware dependent
        print_stage(stage, f"failed to create mobile command stream: {exc}")
        live_view.close()
        return None

    def send_velocity(v_xy: Any, omega: float) -> None:
        stream.send_command(
            build_mobile_base_velocity_command(
                v_xy,
                angular_velocity_radps=float(omega),
                minimum_time=minimum_time,
                control_hold_time=MOBILE_BASE_COMMAND_HOLD_TIME_SEC,
            )
        )

    try:
        print_stage(
            stage,
            "servo streaming live vision -> SE(2) velocity "
            f"(timeout={float(total_timeout_sec):.1f}s, aligned_frames={required_aligned_frames})",
        )
        while time.monotonic() < deadline:
            error_xy = np.asarray(latest_plan["error_xy_m"], dtype=np.float64)
            yaw_error = float(latest_plan["yaw_error_deg"])
            status_lines = [
                (
                    f"servo target x={target_x_m * 100:+.1f}cm y=+0.0cm yaw=base y",
                    (255, 255, 255),
                ),
                (
                    f"error x={error_xy[0] * 100:+.1f}cm y={error_xy[1] * 100:+.1f}cm "
                    f"yaw={yaw_error:+.1f}deg aligned={aligned_frames}/{required_aligned_frames}",
                    (0, 220, 255),
                ),
                (
                    f"cmd vel=[{velocity_xy[0]:+.3f},{velocity_xy[1]:+.3f}]m/s "
                    f"omega={angular_velocity:+.3f}rad/s",
                    (0, 220, 255),
                ),
            ]
            q = robot.get_state().position
            camera_to_base = compute_camera_to_base_for_view_rotation(
                view_rotation,
                dyn_model,
                dyn_state,
                q,
            )
            try:
                usable, mode, estimate, _, abort_requested = live_view.process_next_frame(
                    camera_to_base=camera_to_base,
                    status_lines=status_lines,
                )
            except StopIteration:
                print_stage(stage, "live vision stream ended during servo alignment")
                return latest
            if abort_requested:
                raise UserAbortRequested("combined mobile servo alignment aborted from the live view")

            if usable:
                updates += 1
                latest = live_estimate_to_measurement(
                    robot,
                    dyn_model,
                    dyn_state,
                    view_rotation,
                    estimate,
                    mode,
                )
                plan = mobile_base_combined_alignment_plan(
                    latest,
                    target_x_m=target_x_m,
                    x_tolerance_m=x_tolerance_m,
                    y_tolerance_m=y_tolerance_m,
                    yaw_tolerance_deg=yaw_tolerance_deg,
                    max_step_m=max_step_m,
                    max_step_deg=max_step_deg,
                    max_speed_mps=max_speed_mps,
                    max_angular_speed_radps=max_angular_speed_radps,
                    xy_move_duration_sec=xy_move_duration_sec,
                    yaw_move_duration_sec=yaw_move_duration_sec,
                    coarse_yaw_threshold_deg=coarse_yaw_threshold_deg,
                )
                if plan is None:
                    aligned_frames = 0
                    velocity_xy = np.zeros(2, dtype=np.float64)
                    angular_velocity = 0.0
                else:
                    latest_plan = plan
                    if bool(plan["aligned"]):
                        aligned_frames += 1
                        velocity_xy = np.zeros(2, dtype=np.float64)
                        angular_velocity = 0.0
                        if aligned_frames >= required_aligned_frames:
                            print_stage(
                                stage,
                                f"servo aligned after {updates} usable updates; "
                                f"yaw_error={float(plan['yaw_error_deg']):+.2f}deg",
                            )
                            return latest
                    else:
                        aligned_frames = 0
                        velocity_xy = np.asarray(plan["velocity_xy_mps"], dtype=np.float64)
                        angular_velocity = float(plan["angular_velocity_radps"])
            else:
                aligned_frames = 0
                velocity_xy = np.zeros(2, dtype=np.float64)
                angular_velocity = 0.0
                if updates == 0 or updates % 5 == 0:
                    print_stage(stage, f"servo frame rejected: {mode}; sending zero velocity")

            send_velocity(velocity_xy, angular_velocity)
            sends += 1
            now = time.monotonic()
            if now >= next_log:
                elapsed = float(total_timeout_sec) - max(0.0, deadline - now)
                print_stage(
                    stage,
                    f"servo streaming ({elapsed:.1f}/{float(total_timeout_sec):.1f}s, "
                    f"updates={updates}, sends={sends})",
                )
                next_log = now + COMMAND_WAIT_LOG_INTERVAL_SEC

        print_stage(stage, "servo timeout; returning latest vision measurement for residual gate")
        return latest
    finally:
        zero = np.zeros(2, dtype=np.float64)
        for _ in range(max(1, int(MOBILE_BASE_STOP_REPEATS))):
            send_velocity(zero, 0.0)
            time.sleep(period)
        print_stage(stage, f"servo stop sent; motion sends={sends}, usable updates={updates}")
        live_view.close()


def run_mobile_base_combined_alignment(
    robot: Any,
    dyn_model: Any,
    dyn_state: Any,
    view_rotation: str,
    measurement: dict[str, Any],
    *,
    visualize: bool,
    target_x_m: float,
    x_tolerance_m: float,
    y_tolerance_m: float,
    yaw_tolerance_deg: float,
    coarse_yaw_threshold_deg: float,
    max_speed_mps: float,
    max_angular_speed_radps: float,
    max_step_m: float,
    max_step_deg: float,
    max_iterations: int,
    total_timeout_sec: float,
    xy_move_duration_sec: float,
    yaw_move_duration_sec: float,
    vision_frames_needed: int,
    vision_timeout_sec: float,
    max_center_spread_m: float,
    stage: str = "3-4/7 mobile_base_se2_align",
) -> dict[str, Any] | None:
    """Closed-loop mobile-base yaw/x/y alignment using one SE(2) stage."""
    if visualize:
        return run_mobile_base_combined_servo_alignment(
            robot,
            dyn_model,
            dyn_state,
            view_rotation,
            measurement,
            target_x_m=target_x_m,
            x_tolerance_m=x_tolerance_m,
            y_tolerance_m=y_tolerance_m,
            yaw_tolerance_deg=yaw_tolerance_deg,
            coarse_yaw_threshold_deg=coarse_yaw_threshold_deg,
            max_speed_mps=max_speed_mps,
            max_angular_speed_radps=max_angular_speed_radps,
            max_step_m=max_step_m,
            max_step_deg=max_step_deg,
            total_timeout_sec=total_timeout_sec,
            xy_move_duration_sec=xy_move_duration_sec,
            yaw_move_duration_sec=yaw_move_duration_sec,
            vision_frames_needed=vision_frames_needed,
            stage=stage,
        )

    deadline = time.monotonic() + float(total_timeout_sec)
    latest = measurement
    live_view: ContinuousLiveBoxView | None = None

    try:
        for iteration in range(1, int(max_iterations) + 1):
            plan = mobile_base_combined_alignment_plan(
                latest,
                target_x_m=target_x_m,
                x_tolerance_m=x_tolerance_m,
                y_tolerance_m=y_tolerance_m,
                yaw_tolerance_deg=yaw_tolerance_deg,
                max_step_m=max_step_m,
                max_step_deg=max_step_deg,
                max_speed_mps=max_speed_mps,
                max_angular_speed_radps=max_angular_speed_radps,
                xy_move_duration_sec=xy_move_duration_sec,
                yaw_move_duration_sec=yaw_move_duration_sec,
                coarse_yaw_threshold_deg=coarse_yaw_threshold_deg,
            )
            if plan is None:
                print_stage(stage, "FAILED: no usable long-axis yaw in latest vision measurement")
                return None

            error_xy = np.asarray(plan["error_xy_m"], dtype=np.float64)
            yaw_error = float(plan["yaw_error_deg"])
            center_base = np.asarray(plan["center_base_m"], dtype=np.float64)
            if bool(plan["aligned"]):
                print_stage(
                    stage,
                    f"aligned at iter={iteration - 1}; "
                    f"x={center_base[0]:+.3f}m y={center_base[1]:+.3f}m "
                    f"error=[{error_xy[0]:+.3f}, {error_xy[1]:+.3f}]m "
                    f"yaw_error={yaw_error:+.2f}deg",
                )
                return latest

            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                print_stage(stage, "timeout; returning latest vision measurement for residual gate")
                return latest

            velocity_xy = np.asarray(plan["velocity_xy_mps"], dtype=np.float64)
            angular_velocity = float(plan["angular_velocity_radps"])
            if float(np.linalg.norm(velocity_xy)) <= 1e-6 and abs(angular_velocity) <= 1e-6:
                print_stage(stage, "zero correction; returning latest vision measurement")
                return latest

            move_time = min(float(plan["duration_sec"]), max(0.1, remaining))
            translation_text = "enabled" if bool(plan["translation_enabled"]) else "suppressed"
            status_lines = [
                (
                    f"se2 iter {iteration}/{int(max_iterations)} "
                    f"target x={target_x_m * 100:+.1f}cm y=+0.0cm yaw=base y",
                    (255, 255, 255),
                ),
                (
                    f"error x={error_xy[0] * 100:+.1f}cm y={error_xy[1] * 100:+.1f}cm "
                    f"yaw={yaw_error:+.1f}deg; xy {translation_text}",
                    (0, 220, 255),
                ),
            ]
            step_xy = np.asarray(plan["step_xy_m"], dtype=np.float64)
            print_stage(
                stage,
                f"iter={iteration}/{int(max_iterations)} "
                f"center=[{center_base[0]:+.3f}, {center_base[1]:+.3f}]m "
                f"error_xy=[{error_xy[0]:+.3f}, {error_xy[1]:+.3f}]m "
                f"yaw_error={yaw_error:+.2f}deg "
                f"xy={translation_text} "
                f"step_xy=[{step_xy[0]:+.3f}, {step_xy[1]:+.3f}]m "
                f"step_yaw={float(plan['step_yaw_deg']):+.2f}deg "
                f"vel=[{velocity_xy[0]:+.3f}, {velocity_xy[1]:+.3f}]m/s "
                f"omega={angular_velocity:+.3f}rad/s duration={move_time:.2f}s",
            )
            ok, move_status = stream_mobile_base_velocity_stage(
                robot,
                velocity_xy,
                angular_velocity_radps=angular_velocity,
                duration_sec=move_time,
                stage=stage,
                live_view=live_view,
                dyn_model=dyn_model,
                dyn_state=dyn_state,
                view_rotation=view_rotation,
                status_lines=status_lines,
            )
            if not ok:
                if move_status == "visual_abort":
                    raise UserAbortRequested("combined mobile alignment aborted from the live view")
                print_stage(stage, f"SE(2) command not ok ({move_status}); re-measuring before deciding")

            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                print_stage(stage, "timeout after SE(2) move; recapture required")
                return None
            measure_timeout = min(float(vision_timeout_sec), max(0.1, remaining))
            if live_view is not None:
                refreshed = live_view.collect_measurement(
                    robot,
                    dyn_model,
                    dyn_state,
                    view_rotation,
                    frames_needed=vision_frames_needed,
                    timeout_sec=measure_timeout,
                    max_center_spread_m=max_center_spread_m,
                    stage=stage,
                    status_lines=status_lines,
                )
            else:
                refreshed = capture_live_box_measurement(
                    robot,
                    dyn_model,
                    dyn_state,
                    view_rotation,
                    frames_needed=vision_frames_needed,
                    timeout_sec=measure_timeout,
                    max_center_spread_m=max_center_spread_m,
                    visualize=False,
                    hold_after_capture=False,
                )
            if refreshed is None:
                print_stage(stage, "post-SE(2) vision failed; recapture required")
                return None
            latest = refreshed

        plan = mobile_base_combined_alignment_plan(
            latest,
            target_x_m=target_x_m,
            x_tolerance_m=x_tolerance_m,
            y_tolerance_m=y_tolerance_m,
            yaw_tolerance_deg=yaw_tolerance_deg,
            max_step_m=max_step_m,
            max_step_deg=max_step_deg,
            max_speed_mps=max_speed_mps,
            max_angular_speed_radps=max_angular_speed_radps,
            xy_move_duration_sec=xy_move_duration_sec,
            yaw_move_duration_sec=yaw_move_duration_sec,
            coarse_yaw_threshold_deg=coarse_yaw_threshold_deg,
        )
        if plan is None:
            print_stage(stage, "max iterations reached with no usable yaw")
        else:
            center_base = np.asarray(plan["center_base_m"], dtype=np.float64)
            error_xy = np.asarray(plan["error_xy_m"], dtype=np.float64)
            print_stage(
                stage,
                f"max iterations reached; x={center_base[0]:+.3f}m y={center_base[1]:+.3f}m "
                f"error=[{error_xy[0]:+.3f}, {error_xy[1]:+.3f}]m "
                f"yaw_error={float(plan['yaw_error_deg']):+.2f}deg",
            )
        return latest
    finally:
        if live_view is not None:
            live_view.close()


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


def build_impedance_push_command(
    dyn_model: Any,
    dyn_state: Any,
    q: Any,
    *,
    inward: float = PUSH_DISTANCE,
    hold_time: float = PUSH_HOLD_TIME,
) -> Any:
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
        inward=float(inward),
        lift=0.0,
        translation_weight=IMPEDANCE_TRANSLATION_WEIGHT,
        hold_time=float(hold_time),
        label="push",
    )


def stream_impedance_push_stage(
    robot: Any,
    dyn_model: Any,
    dyn_state: Any,
    q: Any,
    *,
    ramp_time_sec: float = PUSH_RAMP_TIME,
    hold_time_sec: float = PUSH_HOLD_TIME,
    stream_period_sec: float = PUSH_STREAM_PERIOD_SEC,
    stage: str = "6/7 inward_push",
) -> bool:
    """Ramp the inward impedance target instead of jumping it in one command."""
    q0 = np.asarray(q, dtype=np.float64).copy()
    period = max(0.01, float(stream_period_sec))
    ramp_time = max(0.0, float(ramp_time_sec))
    hold_time = max(0.0, float(hold_time_sec))
    ramp_command_hold_time = max(period * 2.0, float(PUSH_STREAM_COMMAND_HOLD_TIME))

    try:
        stream = robot.create_command_stream(priority=PUSH_STREAM_PRIORITY)
    except Exception as exc:  # pragma: no cover - SDK/hardware dependent
        print_stage(stage, f"failed to create push command stream: {exc}")
        return False

    def send_progress(progress: float, *, command_hold_time: float) -> float:
        clamped = float(np.clip(progress, 0.0, 1.0))
        stream.send_command(
            build_impedance_push_command(
                dyn_model,
                dyn_state,
                q0,
                inward=PUSH_DISTANCE * clamped,
                hold_time=command_hold_time,
            )
        )
        return clamped

    print_stage(
        stage,
        f"streaming impedance target ramp over {ramp_time:.2f}s "
        f"to inward={PUSH_DISTANCE:.3f}m at {1.0 / period:.1f}Hz",
    )
    start = time.monotonic()
    next_send = start
    next_log = start + COMMAND_WAIT_LOG_INTERVAL_SEC
    sends = 0
    last_progress = 0.0

    if ramp_time <= 0.0:
        last_progress = send_progress(1.0, command_hold_time=hold_time)
        sends += 1
    else:
        last_progress = send_progress(0.0, command_hold_time=ramp_command_hold_time)
        sends += 1
        next_send = start + period
        deadline = start + ramp_time
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_send:
                progress = min(1.0, max(0.0, (now - start) / ramp_time))
                last_progress = send_progress(
                    progress,
                    command_hold_time=ramp_command_hold_time,
                )
                sends += 1
                next_send = now + period

            now = time.monotonic()
            if now >= next_log:
                elapsed = min(ramp_time, max(0.0, now - start))
                print_stage(
                    stage,
                    f"ramping ({elapsed:.1f}/{ramp_time:.1f}s, "
                    f"progress={last_progress * 100.0:.0f}%, sends={sends})",
                )
                next_log = now + COMMAND_WAIT_LOG_INTERVAL_SEC

            sleep_sec = min(0.005, max(0.0, min(next_send, deadline) - time.monotonic()))
            if sleep_sec > 0.0:
                time.sleep(sleep_sec)

    if last_progress < 1.0:
        send_progress(1.0, command_hold_time=hold_time)
        sends += 1

    print_stage(stage, f"ramp complete; holding final push target for {hold_time:.2f}s")
    if hold_time > 0.0:
        time.sleep(hold_time)
    print_stage(stage, f"done; stream sends={sends}")
    return True


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
    push_ramp_time_sec: float,
    visualize: bool,
    visualize_only: bool,
    gripper_open: bool,
    mobile_base_yaw_align: bool,
    mobile_base_yaw_tolerance_deg: float,
    mobile_base_yaw_max_speed_radps: float,
    mobile_base_yaw_max_step_deg: float,
    mobile_base_yaw_max_iterations: int,
    mobile_base_yaw_total_timeout_sec: float,
    mobile_base_yaw_move_duration_sec: float,
    mobile_base_yaw_vision_frames_needed: int,
    mobile_base_yaw_vision_timeout_sec: float,
    mobile_base_combined_coarse_yaw_threshold_deg: float,
    mobile_base_align: bool,
    mobile_base_target_x_m: float,
    mobile_base_x_tolerance_m: float,
    mobile_base_y_tolerance_m: float,
    mobile_base_max_speed_mps: float,
    mobile_base_max_step_m: float,
    mobile_base_max_iterations: int,
    mobile_base_total_timeout_sec: float,
    mobile_base_move_duration_sec: float,
    mobile_base_vision_frames_needed: int,
    mobile_base_vision_timeout_sec: float,
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
    gripper_device: MaxOpenGripper | None = None
    try:
        if not visualize_only and gripper_open:
            gripper_device = setup_max_open_gripper(robot)
            if gripper_device is None:
                print_stage("0/7 gripper_home_open", "FAILED; aborting before arm motion")
                return False
        elif not visualize_only:
            print_stage("0/7 gripper_home_open", "skipped by --no-gripper-open")

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
            stage = f"1/7 {name}"
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

        using_live_vision = box_center_camera_m is None
        if box_center_camera_m is None:
            # Live capture AT this pose, so the FK-based camera->base above and
            # the measurement share the exact same posture.
            print_stage("2/7 live_vision", "capturing D405 frames")
            measurement = capture_live_box_measurement(
                robot,
                dyn_model,
                dyn_state,
                view_rotation,
                frames_needed=live_vision_frames_needed,
                timeout_sec=live_vision_timeout_sec,
                max_center_spread_m=live_center_spread_m,
                visualize=visualize,
                hold_after_capture=visualize,
            )
            if measurement is None:
                print_stage(
                    "2/7 live_vision",
                    "FAILED: not enough usable center frames within "
                    f"{live_vision_timeout_sec:.0f}s; aborting before contact",
                )
                return done
            print_stage(
                "2/7 live_vision",
                f"done; usable frames={measurement['frames_used']} "
                f"| center spread={measurement['center_spread_m'] * 1000.0:.1f} mm "
                f"| modes={sorted(set(measurement['modes']))}",
            )
            if mobile_base_yaw_align and mobile_base_align and str(model).lower() == "m":
                if mobile_base_combined_coarse_yaw_threshold_deg >= 90.0:
                    xy_policy = "xy enabled during yaw alignment"
                else:
                    xy_policy = (
                        f"xy suppressed while |yaw|>{mobile_base_combined_coarse_yaw_threshold_deg:.1f}deg"
                    )
                print_stage(
                    "3-4/7 mobile_base_se2_align",
                    "closed-loop combined target "
                    f"x={mobile_base_target_x_m:.3f}m±{mobile_base_x_tolerance_m:.3f}m, "
                    f"y=0.000m±{mobile_base_y_tolerance_m:.3f}m, "
                    f"yaw≤{mobile_base_yaw_tolerance_deg:.1f}deg ({xy_policy})",
                )
                combined_measurement = run_mobile_base_combined_alignment(
                    robot,
                    dyn_model,
                    dyn_state,
                    view_rotation,
                    measurement,
                    visualize=visualize,
                    target_x_m=mobile_base_target_x_m,
                    x_tolerance_m=mobile_base_x_tolerance_m,
                    y_tolerance_m=mobile_base_y_tolerance_m,
                    yaw_tolerance_deg=mobile_base_yaw_tolerance_deg,
                    coarse_yaw_threshold_deg=mobile_base_combined_coarse_yaw_threshold_deg,
                    max_speed_mps=mobile_base_max_speed_mps,
                    max_angular_speed_radps=mobile_base_yaw_max_speed_radps,
                    max_step_m=mobile_base_max_step_m,
                    max_step_deg=mobile_base_yaw_max_step_deg,
                    max_iterations=max(mobile_base_max_iterations, mobile_base_yaw_max_iterations),
                    total_timeout_sec=max(mobile_base_total_timeout_sec, mobile_base_yaw_total_timeout_sec),
                    xy_move_duration_sec=mobile_base_move_duration_sec,
                    yaw_move_duration_sec=mobile_base_yaw_move_duration_sec,
                    vision_frames_needed=max(mobile_base_vision_frames_needed, mobile_base_yaw_vision_frames_needed),
                    vision_timeout_sec=max(mobile_base_vision_timeout_sec, mobile_base_yaw_vision_timeout_sec),
                    max_center_spread_m=live_center_spread_m,
                )
                if combined_measurement is None:
                    print_stage(
                        "3-4/7 mobile_base_se2_align",
                        "alignment recapture failed; trying one normal live_vision recovery",
                    )
                    combined_measurement = capture_live_box_measurement(
                        robot,
                        dyn_model,
                        dyn_state,
                        view_rotation,
                        frames_needed=live_vision_frames_needed,
                        timeout_sec=live_vision_timeout_sec,
                        max_center_spread_m=live_center_spread_m,
                        visualize=visualize,
                        hold_after_capture=False,
                    )
                if combined_measurement is None:
                    print_stage(
                        "3-4/7 mobile_base_se2_align",
                        "FAILED: no fresh vision after base motion; aborting before contact",
                    )
                    return done
                residual_yaw = mobile_base_yaw_alignment_error_deg(combined_measurement)
                if residual_yaw is None:
                    print_stage(
                        "3-4/7 mobile_base_se2_align",
                        "FAILED: no usable long-axis yaw after alignment; aborting before contact",
                    )
                    return done
                if abs(float(residual_yaw)) > float(mobile_base_yaw_tolerance_deg):
                    print_stage(
                        "3-4/7 mobile_base_se2_align",
                        "FAILED: residual yaw error "
                        f"{residual_yaw:+.2f}deg exceeds {mobile_base_yaw_tolerance_deg:.1f}deg; "
                        "aborting before contact",
                    )
                    return done
                xy_status = mobile_base_xy_residual_status(
                    combined_measurement["center_base_m"],
                    target_x_m=mobile_base_target_x_m,
                    x_tolerance_m=mobile_base_x_tolerance_m,
                    y_tolerance_m=mobile_base_y_tolerance_m,
                )
                residual_xy = np.asarray(xy_status["residual_xy_m"], dtype=np.float64)
                if not bool(xy_status["within_target_band"]):
                    if bool(xy_status["within_safety_band"]):
                        reason = (
                            f"outside target band x±{mobile_base_x_tolerance_m * 100:.1f}cm "
                            f"y±{mobile_base_y_tolerance_m * 100:.1f}cm"
                        )
                    else:
                        reason = (
                            f"exceeds safety band {MOBILE_BASE_MAX_RESIDUAL_XY_M * 100:.0f}cm; "
                            "re-approach with the mobile base"
                        )
                    print_stage(
                        "3-4/7 mobile_base_se2_align",
                        "FAILED: residual error "
                        f"x={residual_xy[0] * 100:+.1f}cm y={residual_xy[1] * 100:+.1f}cm "
                        f"{reason}; aborting before contact",
                    )
                    return done
                measurement = combined_measurement
                mobile_base_yaw_align = False
                mobile_base_align = False
            if mobile_base_yaw_align and str(model).lower() == "m":
                print_stage(
                    "3/7 mobile_base_yaw_align",
                    "closed-loop target box long-axis -> base y "
                    f"(tolerance={mobile_base_yaw_tolerance_deg:.1f}deg)",
                )
                yaw_measurement = run_mobile_base_yaw_alignment(
                    robot,
                    dyn_model,
                    dyn_state,
                    view_rotation,
                    measurement,
                    visualize=visualize,
                    tolerance_deg=mobile_base_yaw_tolerance_deg,
                    max_angular_speed_radps=mobile_base_yaw_max_speed_radps,
                    max_step_deg=mobile_base_yaw_max_step_deg,
                    max_iterations=mobile_base_yaw_max_iterations,
                    total_timeout_sec=mobile_base_yaw_total_timeout_sec,
                    move_duration_sec=mobile_base_yaw_move_duration_sec,
                    vision_frames_needed=mobile_base_yaw_vision_frames_needed,
                    vision_timeout_sec=mobile_base_yaw_vision_timeout_sec,
                    max_center_spread_m=live_center_spread_m,
                )
                if yaw_measurement is None:
                    print_stage(
                        "3/7 mobile_base_yaw_align",
                        "FAILED: no fresh yaw measurement after base rotation; aborting before contact",
                    )
                    return done
                residual_yaw = mobile_base_yaw_alignment_error_deg(yaw_measurement)
                if residual_yaw is None:
                    print_stage(
                        "3/7 mobile_base_yaw_align",
                        "FAILED: no usable long-axis yaw after alignment; aborting before contact",
                    )
                    return done
                if abs(float(residual_yaw)) > float(mobile_base_yaw_tolerance_deg):
                    print_stage(
                        "3/7 mobile_base_yaw_align",
                        "FAILED: residual yaw error "
                        f"{residual_yaw:+.2f}deg exceeds {mobile_base_yaw_tolerance_deg:.1f}deg; "
                        "aborting before contact",
                    )
                    return done
                measurement = yaw_measurement
            elif mobile_base_yaw_align:
                print_stage(
                    "3/7 mobile_base_yaw_align",
                    f"FAILED: model={model!r} cannot run M mobile-base yaw alignment; aborting before contact",
                )
                return done

            if mobile_base_align and str(model).lower() == "m":
                print_stage(
                    "4/7 mobile_base_xy_align",
                    "closed-loop target "
                    f"x={mobile_base_target_x_m:.3f}m±{mobile_base_x_tolerance_m:.3f}m, "
                    f"y=0.000m±{mobile_base_y_tolerance_m:.3f}m",
                )
                aligned_measurement = run_mobile_base_alignment(
                    robot,
                    dyn_model,
                    dyn_state,
                    view_rotation,
                    measurement,
                    visualize=visualize,
                    target_x_m=mobile_base_target_x_m,
                    x_tolerance_m=mobile_base_x_tolerance_m,
                    y_tolerance_m=mobile_base_y_tolerance_m,
                    max_speed_mps=mobile_base_max_speed_mps,
                    max_step_m=mobile_base_max_step_m,
                    max_iterations=mobile_base_max_iterations,
                    total_timeout_sec=mobile_base_total_timeout_sec,
                    move_duration_sec=mobile_base_move_duration_sec,
                    vision_frames_needed=mobile_base_vision_frames_needed,
                    vision_timeout_sec=mobile_base_vision_timeout_sec,
                    max_center_spread_m=live_center_spread_m,
                    min_command_timeout_sec=min_command_timeout_sec,
                    command_timeout_margin_sec=command_timeout_margin_sec,
                    stage="4/7 mobile_base_xy_align",
                )
                if aligned_measurement is None:
                    print_stage(
                        "4/7 mobile_base_xy_align",
                        "alignment recapture failed; trying one normal live_vision recovery",
                    )
                    aligned_measurement = capture_live_box_measurement(
                        robot,
                        dyn_model,
                        dyn_state,
                        view_rotation,
                        frames_needed=live_vision_frames_needed,
                        timeout_sec=live_vision_timeout_sec,
                        max_center_spread_m=live_center_spread_m,
                        visualize=visualize,
                        hold_after_capture=False,
                    )
                if aligned_measurement is None:
                    print_stage(
                        "4/7 mobile_base_xy_align",
                        "FAILED: no fresh vision after base motion; aborting before contact",
                    )
                    return done
                measurement = aligned_measurement

                # Contact gate: the arms only move if alignment actually ended
                # near the target. A large residual (timeout / max iterations)
                # would reproduce the far-reach IK failure this script avoids.
                xy_status = mobile_base_xy_residual_status(
                    measurement["center_base_m"],
                    target_x_m=mobile_base_target_x_m,
                    x_tolerance_m=mobile_base_x_tolerance_m,
                    y_tolerance_m=mobile_base_y_tolerance_m,
                )
                residual_xy = np.asarray(xy_status["residual_xy_m"], dtype=np.float64)
                if not bool(xy_status["within_target_band"]):
                    if bool(xy_status["within_safety_band"]):
                        reason = (
                            f"outside target band x±{mobile_base_x_tolerance_m * 100:.1f}cm "
                            f"y±{mobile_base_y_tolerance_m * 100:.1f}cm"
                        )
                    else:
                        reason = (
                            f"exceeds safety band {MOBILE_BASE_MAX_RESIDUAL_XY_M * 100:.0f}cm; "
                            "re-approach with the mobile base"
                        )
                    print_stage(
                        "4/7 mobile_base_xy_align",
                        "FAILED: residual error "
                        f"x={residual_xy[0] * 100:+.1f}cm y={residual_xy[1] * 100:+.1f}cm "
                        f"{reason}; aborting before contact",
                    )
                    return done
            elif mobile_base_align:
                print_stage(
                    "4/7 mobile_base_xy_align",
                    f"skipped because model={model!r}; M model is required for y-axis SE(2)",
                )
        else:
            print_stage("2/7 live_vision", "skipped; using supplied center_top_camera_m")
            q = robot.get_state().position
            camera_to_base = compute_camera_to_base_for_view_rotation(
                view_rotation,
                dyn_model,
                dyn_state,
                q,
            )
            measurement = {
                "q": q,
                "camera_to_base": camera_to_base,
                "center_camera_m": np.asarray(box_center_camera_m, dtype=np.float64),
                "center_base_m": transform_camera_point_to_base(
                    box_center_camera_m,
                    camera_to_base,
                ),
                "long_axis_camera": None,
                "long_axis_unconstrained": False,
                "drop_delta_axis_base_xy": None,
            }
            if mobile_base_align:
                print_stage(
                    "4/7 mobile_base_xy_align",
                    "skipped because --box-center-camera/--vision-json is not a live feedback source",
                )
            if mobile_base_yaw_align:
                print_stage(
                    "3/7 mobile_base_yaw_align",
                    "FAILED: --box-center-camera/--vision-json has no live yaw feedback; aborting before contact",
                )
                return done

        box_center_camera_m = np.asarray(measurement["center_camera_m"], dtype=np.float64)
        box_center_base_m = np.asarray(measurement["center_base_m"], dtype=np.float64)
        camera_to_base = np.asarray(measurement["camera_to_base"], dtype=np.float64)
        long_axis_camera = measurement["long_axis_camera"]
        drop_delta_axis_base_xy = measurement["drop_delta_axis_base_xy"]
        if measurement["long_axis_unconstrained"] and long_axis_camera is None:
            print_stage(
                "2/7 live_vision",
                "FAILED: long axis underconstrained and no axis direction; aborting",
            )
            return done
        if drop_delta_axis_base_xy is not None:
            print_stage(
                "2/7 live_vision",
                "long axis underconstrained (both short sides cropped); its "
                "correction component will be dropped",
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
                "-- after optional mobile-base yaw alignment."
            )

        if continue_pick and not verify_yaw_safe_before_contact(
            measurement,
            tolerance_deg=mobile_base_yaw_tolerance_deg,
            stage="pre_contact_yaw_gate",
        ):
            return done

        print_stage("5/7 vision_pre_push", "building target")
        q = robot.get_state().position
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
            "5/7 vision_pre_push",
            timeout_sec=stage_timeout_sec(
                approach_time,
                hold_time,
                min_timeout_sec=min_command_timeout_sec,
                margin_sec=command_timeout_margin_sec,
            ),
        ):
            print_stage("5/7 vision_pre_push", "FAILED; aborting")
            return done
        print_stage("5/7 vision_pre_push", "reached")

        if not continue_pick:
            done = True
            print_stage("5/7 vision_pre_push", "pre-push-only stop; done=True")
            return done

        print_stage("6/7 inward_push", "building ramped target stream")
        q = robot.get_state().position
        if not stream_impedance_push_stage(
            robot,
            dyn_model,
            dyn_state,
            q,
            ramp_time_sec=push_ramp_time_sec,
            hold_time_sec=PUSH_HOLD_TIME,
            stage="6/7 inward_push",
        ):
            print_stage("6/7 inward_push", "FAILED; aborting")
            return done

        print_stage("7/7 lift", "building target")
        q = robot.get_state().position
        if not send_stage(
            robot,
            build_impedance_lift_command(dyn_model, dyn_state, q),
            "7/7 lift",
            timeout_sec=stage_timeout_sec(
                LIFT_MINIMUM_TIME,
                LIFT_HOLD_TIME,
                min_timeout_sec=min_command_timeout_sec,
                margin_sec=command_timeout_margin_sec,
            ),
        ):
            print_stage("7/7 lift", "FAILED; aborting")
            return done
        print_stage("7/7 lift", "done")

        done = True
        print("=" * 60)
        print(f"[picking] picking motion COMPLETED. done = {done}")
        print("=" * 60)
        return done
    except UserAbortRequested as abort:
        print_stage("aborted", f"operator abort: {abort}; no further motion")
        return done
    except KeyboardInterrupt:
        print_stage("interrupted", "KeyboardInterrupt while waiting in the last printed stage")
        raise
    finally:
        if gripper_device is not None:
            print_stage("0/7 gripper_home_open", "stopping max-open background loop")
            gripper_device.stop()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vision-adjusted picking_box_5 with combined M SE(2) alignment")
    parser.add_argument("--address", type=str, required=True, help="Robot address")
    parser.add_argument("--model", type=str, default="m", help="Robot Model Name")
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
    parser.set_defaults(gripper_open=GRIPPER_OPEN_DEFAULT)
    parser.add_argument(
        "--gripper-open",
        dest="gripper_open",
        action="store_true",
        help="Home both gripper fingers at startup and keep them max-open. This is the default.",
    )
    parser.add_argument(
        "--no-gripper-open",
        dest="gripper_open",
        action="store_false",
        help="Debug only: skip gripper homing/opening before arm motion.",
    )
    parser.set_defaults(mobile_base_yaw_align=MOBILE_BASE_YAW_ALIGN_DEFAULT)
    parser.add_argument(
        "--mobile-base-yaw-align",
        dest="mobile_base_yaw_align",
        action="store_true",
        help=(
            "Enable M mobile-base theta alignment so box long axis is parallel to base y. "
            "In picking_box_5 this is combined with x/y alignment when both are enabled."
        ),
    )
    parser.add_argument(
        "--no-mobile-base-yaw-align",
        dest="mobile_base_yaw_align",
        action="store_false",
        help="Disable mobile-base yaw alignment. Use only for debugging or already-aligned boxes.",
    )
    parser.add_argument(
        "--mobile-base-yaw-tolerance-deg",
        type=float,
        default=MOBILE_BASE_YAW_TOLERANCE_DEG,
        help="Required final yaw error before contact; box long axis must align to base y within this.",
    )
    parser.add_argument(
        "--mobile-base-yaw-max-speed-radps",
        type=float,
        default=MOBILE_BASE_YAW_MAX_SPEED_RADPS,
        help="Maximum angular velocity norm for mobile-base yaw alignment.",
    )
    parser.add_argument(
        "--mobile-base-yaw-max-step-deg",
        type=float,
        default=MOBILE_BASE_YAW_MAX_STEP_DEG,
        help="Maximum requested yaw correction per alignment iteration.",
    )
    parser.add_argument(
        "--mobile-base-yaw-max-iterations",
        type=int,
        default=MOBILE_BASE_YAW_MAX_ITERATIONS,
        help="Maximum closed-loop yaw alignment iterations before aborting at the residual gate.",
    )
    parser.add_argument(
        "--mobile-base-yaw-timeout-sec",
        type=float,
        default=MOBILE_BASE_YAW_TOTAL_TIMEOUT_SEC,
        help="Total yaw-alignment wall-clock budget before aborting at the residual gate.",
    )
    parser.add_argument(
        "--mobile-base-yaw-move-duration-sec",
        type=float,
        default=MOBILE_BASE_YAW_MOVE_DURATION_SEC,
        help="Minimum time for each small theta correction command.",
    )
    parser.add_argument(
        "--mobile-base-yaw-vision-frames",
        type=int,
        default=MOBILE_BASE_YAW_VISION_FRAMES_NEEDED,
        help="Usable live vision frames per post-yaw recapture.",
    )
    parser.add_argument(
        "--mobile-base-yaw-vision-timeout-sec",
        type=float,
        default=MOBILE_BASE_YAW_VISION_TIMEOUT_SEC,
        help="Timeout for each post-yaw recapture.",
    )
    parser.add_argument(
        "--mobile-base-combined-coarse-yaw-threshold-deg",
        type=float,
        default=MOBILE_BASE_COMBINED_COARSE_YAW_THRESHOLD_DEG,
        help=(
            "In picking_box_5 combined SE(2) alignment, suppress x/y translation "
            "while absolute yaw error is larger than this threshold."
        ),
    )
    parser.set_defaults(mobile_base_align=MOBILE_BASE_ALIGN_DEFAULT)
    parser.add_argument(
        "--mobile-base-align",
        dest="mobile_base_align",
        action="store_true",
        help="Enable closed-loop M mobile-base x/y alignment before vision_pre_push. This is the default.",
    )
    parser.add_argument(
        "--no-mobile-base-align",
        dest="mobile_base_align",
        action="store_false",
        help="Disable mobile-base alignment and use arm-only vision_pre_push like picking_box_2.",
    )
    parser.add_argument(
        "--mobile-base-target-x-m",
        type=float,
        default=MOBILE_BASE_TARGET_X_M,
        help="Target box center x in base frame after mobile-base alignment.",
    )
    parser.add_argument(
        "--mobile-base-x-tolerance-m",
        type=float,
        default=MOBILE_BASE_X_TOLERANCE_M,
        help="Accepted x tolerance around --mobile-base-target-x-m.",
    )
    parser.add_argument(
        "--mobile-base-y-tolerance-m",
        type=float,
        default=MOBILE_BASE_Y_TOLERANCE_M,
        help="Accepted y tolerance around 0.0 m.",
    )
    parser.add_argument(
        "--mobile-base-max-speed-mps",
        type=float,
        default=MOBILE_BASE_MAX_SPEED_MPS,
        help="Maximum SE(2) linear speed norm for alignment moves.",
    )
    parser.add_argument(
        "--mobile-base-max-step-m",
        type=float,
        default=MOBILE_BASE_MAX_STEP_M,
        help="Maximum requested x/y correction norm per alignment iteration.",
    )
    parser.add_argument(
        "--mobile-base-max-iterations",
        type=int,
        default=MOBILE_BASE_MAX_ITERATIONS,
        help="Maximum closed-loop align iterations before continuing with vision_pre_push.",
    )
    parser.add_argument(
        "--mobile-base-timeout-sec",
        type=float,
        default=MOBILE_BASE_TOTAL_TIMEOUT_SEC,
        help="Total alignment-loop wall-clock budget before continuing with vision_pre_push.",
    )
    parser.add_argument(
        "--mobile-base-move-duration-sec",
        type=float,
        default=MOBILE_BASE_MOVE_DURATION_SEC,
        help="Minimum time for each small SE(2) correction command.",
    )
    parser.add_argument(
        "--mobile-base-vision-frames",
        type=int,
        default=MOBILE_BASE_VISION_FRAMES_NEEDED,
        help="Usable live vision frames per post-base-move alignment measurement.",
    )
    parser.add_argument(
        "--mobile-base-vision-timeout-sec",
        type=float,
        default=MOBILE_BASE_VISION_TIMEOUT_SEC,
        help="Timeout for each post-base-move alignment vision measurement.",
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
    parser.add_argument(
        "--push-ramp-time-sec",
        type=float,
        default=PUSH_RAMP_TIME,
        help=(
            "Time to ramp the inward impedance push target from 0 to full PUSH_DISTANCE. "
            "Increase this if the grippers hit the box too hard."
        ),
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
    if args.mobile_base_yaw_tolerance_deg <= 0.0:
        raise SystemExit("--mobile-base-yaw-tolerance-deg must be positive")
    if args.mobile_base_yaw_max_speed_radps <= 0.0:
        raise SystemExit("--mobile-base-yaw-max-speed-radps must be positive")
    if args.mobile_base_yaw_max_step_deg <= 0.0:
        raise SystemExit("--mobile-base-yaw-max-step-deg must be positive")
    if args.mobile_base_yaw_max_iterations < 0:
        raise SystemExit("--mobile-base-yaw-max-iterations must be non-negative")
    if args.mobile_base_yaw_timeout_sec <= 0.0:
        raise SystemExit("--mobile-base-yaw-timeout-sec must be positive")
    if args.mobile_base_yaw_move_duration_sec <= 0.0:
        raise SystemExit("--mobile-base-yaw-move-duration-sec must be positive")
    if args.mobile_base_yaw_vision_frames <= 0:
        raise SystemExit("--mobile-base-yaw-vision-frames must be positive")
    if args.mobile_base_yaw_vision_timeout_sec <= 0.0:
        raise SystemExit("--mobile-base-yaw-vision-timeout-sec must be positive")
    if args.mobile_base_combined_coarse_yaw_threshold_deg <= args.mobile_base_yaw_tolerance_deg:
        raise SystemExit(
            "--mobile-base-combined-coarse-yaw-threshold-deg must be larger than "
            "--mobile-base-yaw-tolerance-deg"
        )
    if args.mobile_base_x_tolerance_m <= 0.0:
        raise SystemExit("--mobile-base-x-tolerance-m must be positive")
    if args.mobile_base_y_tolerance_m <= 0.0:
        raise SystemExit("--mobile-base-y-tolerance-m must be positive")
    if args.mobile_base_max_speed_mps <= 0.0:
        raise SystemExit("--mobile-base-max-speed-mps must be positive")
    if args.mobile_base_max_step_m <= 0.0:
        raise SystemExit("--mobile-base-max-step-m must be positive")
    if args.mobile_base_max_iterations < 0:
        raise SystemExit("--mobile-base-max-iterations must be non-negative")
    if args.mobile_base_timeout_sec <= 0.0:
        raise SystemExit("--mobile-base-timeout-sec must be positive")
    if args.mobile_base_move_duration_sec <= 0.0:
        raise SystemExit("--mobile-base-move-duration-sec must be positive")
    if args.mobile_base_vision_frames <= 0:
        raise SystemExit("--mobile-base-vision-frames must be positive")
    if args.mobile_base_vision_timeout_sec <= 0.0:
        raise SystemExit("--mobile-base-vision-timeout-sec must be positive")
    if args.push_ramp_time_sec < 0.0:
        raise SystemExit("--push-ramp-time-sec must be non-negative")
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
            push_ramp_time_sec=float(args.push_ramp_time_sec),
            visualize=bool(args.visualize or args.visualize_only),
            visualize_only=bool(args.visualize_only),
            gripper_open=bool(args.gripper_open),
            mobile_base_yaw_align=bool(args.mobile_base_yaw_align),
            mobile_base_yaw_tolerance_deg=float(args.mobile_base_yaw_tolerance_deg),
            mobile_base_yaw_max_speed_radps=float(args.mobile_base_yaw_max_speed_radps),
            mobile_base_yaw_max_step_deg=float(args.mobile_base_yaw_max_step_deg),
            mobile_base_yaw_max_iterations=int(args.mobile_base_yaw_max_iterations),
            mobile_base_yaw_total_timeout_sec=float(args.mobile_base_yaw_timeout_sec),
            mobile_base_yaw_move_duration_sec=float(args.mobile_base_yaw_move_duration_sec),
            mobile_base_yaw_vision_frames_needed=int(args.mobile_base_yaw_vision_frames),
            mobile_base_yaw_vision_timeout_sec=float(args.mobile_base_yaw_vision_timeout_sec),
            mobile_base_combined_coarse_yaw_threshold_deg=float(
                args.mobile_base_combined_coarse_yaw_threshold_deg
            ),
            mobile_base_align=bool(args.mobile_base_align),
            mobile_base_target_x_m=float(args.mobile_base_target_x_m),
            mobile_base_x_tolerance_m=float(args.mobile_base_x_tolerance_m),
            mobile_base_y_tolerance_m=float(args.mobile_base_y_tolerance_m),
            mobile_base_max_speed_mps=float(args.mobile_base_max_speed_mps),
            mobile_base_max_step_m=float(args.mobile_base_max_step_m),
            mobile_base_max_iterations=int(args.mobile_base_max_iterations),
            mobile_base_total_timeout_sec=float(args.mobile_base_timeout_sec),
            mobile_base_move_duration_sec=float(args.mobile_base_move_duration_sec),
            mobile_base_vision_frames_needed=int(args.mobile_base_vision_frames),
            mobile_base_vision_timeout_sec=float(args.mobile_base_vision_timeout_sec),
            command_timeout_margin_sec=float(args.command_timeout_margin_sec),
            min_command_timeout_sec=float(args.min_command_timeout_sec),
        )
        else 1
    )
