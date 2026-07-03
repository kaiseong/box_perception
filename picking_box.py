# Picking Box Demo
# This example moves the robot through three recorded poses to perform a
# "pick up an object" motion. Each pose is a joint set (torso / right arm /
# left arm / head), commanded with a Joint Position Controller and sent to the
# robot with the one-shot "send once" pattern (robot.send_command(...).get()).
#
# In addition, force/torque (FT) sensor data on both arms is monitored in real
# time through a state-update callback (robot.start_state_update), which runs in
# a background thread while the picking motion is executing.
#
# Motion sequence:
#   1. ready            -> home/ready posture         (joint position control)
#   2. ready_to_picking -> approach posture           (joint position control)
#   3. start_to_picking -> push both hands inward     (Cartesian impedance control)
#   4. ready_to_move    -> raise both hands straight up (Cartesian impedance, timed)
#
# Steps 3-4 use Cartesian impedance control. In step 3 (ImpedanceControlCommand)
# the right hand is pushed toward +y and the left toward -y (both inward) so the
# end-effectors compliantly press into the object. In step 4
# (CartesianImpedanceControlCommand) the hands are raised STRAIGHT UP along the
# base-frame +z axis following a TIMED trajectory (set_minimum_time + velocity
# limits) for a smooth, gentle rise, while joint-space impedance holds the box.
#
# When the whole sequence finishes successfully, a completion flag is raised.
#
# Usage example:
#     python picking_box.py --address 192.168.30.1:50051 --model a
#
# Copyright (c) 2025 Rainbow Robotics. All rights reserved.
#
# DISCLAIMER:
# This is a sample code provided for educational and reference purposes only.
# Rainbow Robotics shall not be held liable for any damages or malfunctions resulting from
# the use or misuse of this demo code. Please use with caution and at your own discretion.

import argparse

import numpy as np
import rby1_sdk as rby

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

# Time (seconds) the robot takes to reach each pose. Larger = slower/safer.
MINIMUM_TIME = 2.0

# Rate (Hz) at which the FT-sensor monitoring callback is invoked.
FT_MONITOR_RATE = 10.0

# ---- Cartesian impedance "push inward" parameters (for start_to_picking) ----
# Link indices into the dynamics state (order MUST match DYN_LINK_NAMES below).
DYN_LINK_NAMES = ["base", "link_torso_5", "ee_right", "ee_left", "link_head_2"]
BASE_INDEX, TORSO_INDEX, EE_RIGHT_INDEX, EE_LEFT_INDEX, HEAD2_INDEX = 0, 1, 2, 3, 4
# Reference link the EEF target pose is expressed in (torso tip = arm root).
IMPEDANCE_REFERENCE_LINK = "link_torso_5"
# Distance [m] each hand moves inward along the reference-frame y-axis.
# Right hand -> +y, left hand -> -y (both toward the center).
PUSH_DISTANCE = 0.1
# Task-space (Cartesian) stiffness diag(Kx, Ky, Kz) [N/m]. Keep y firm so the
# hands actually push; x/z softer for compliance. TUNE for your object/robot.
IMPEDANCE_TRANSLATION_WEIGHT = [500.0, 1000.0, 500.0]
# Rotational stiffness diag [Nm/rad].
IMPEDANCE_ROTATION_WEIGHT = [50.0, 50.0, 50.0]
# ---- Cartesian IMPEDANCE lift (ready_to_move): time-controlled & smooth ----
# The lift uses CartesianImpedanceControlCommandBuilder so it can follow a TIMED
# trajectory (set_minimum_time + velocity limits) -> smooth, gentle rise, while
# joint-space impedance holds the box firmly (compliant, not a rigid jerk).
# Target is expressed in the BASE frame so +z is TRUE vertical ("하늘 방향").
IMPEDANCE_LIFT_REFERENCE_LINK = "base"
# Height [m] to raise the EEF straight up (+z, base frame).
LIFT_HEIGHT = 0.1
# Trajectory time [s] for the lift. Larger = slower / smoother rise (like the
# joint-position minimum_time, but for this compliant Cartesian move).
LIFT_MINIMUM_TIME = 5.0
# Cartesian velocity / acceleration caps for the lift trajectory (safety bounds;
# with LIFT_MINIMUM_TIME large, the time governs and these rarely bind).
LIFT_LINEAR_VELOCITY_LIMIT = 0.1                # m/s
LIFT_ANGULAR_VELOCITY_LIMIT = float(np.pi / 4)  # rad/s
LIFT_LINEAR_ACCELERATION_LIMIT = 0.5            # m/s^2
LIFT_ANGULAR_ACCELERATION_LIMIT = float(np.pi)  # rad/s^2
# Per-arm (7-joint) joint-space impedance that holds the box while lifting. TUNE
# for your robot/payload: stiffness firm enough to hold, torque above the load.
LIFT_JOINT_STIFFNESS = [100.0] * 7              # Nm/rad
LIFT_JOINT_DAMPING_RATIO = 1.0                  # critically damped -> smooth, no overshoot
LIFT_JOINT_TORQUE_LIMIT = [50.0] * 7            # Nm (must exceed the holding torque)
# control_hold_time [s] for the push (short: settle grip) and the final lift
# (long: keep the box raised at the end). With the blocking "send once" pattern
# a command must finish before the next runs.
PUSH_HOLD_TIME = 3.0
LIFT_HOLD_TIME = 100.0

# ---- Vision extrinsic: D405 camera frame into T5/link_torso_5 frame ----
# Matrix naming below uses target_from_source convention:
#   p_target = T_TARGET_FROM_SOURCE @ [p_source_x, p_source_y, p_source_z, 1]
#
# User-measured D405 extrinsic: x,y,z [m], roll,pitch,yaw [deg], Euler ZYX.
# The mount frame is link_head_2, i.e. AFTER the head_1 pitch joint: with the
# recorded D405 data the box rim-plane normal maps to world vertical within
# 0.3 deg through link_head_2, but is off by exactly the head_1 angle (25.2 deg
# at head_1 = 0.436 rad) through link_head_1 -- the camera pitches with head_1.
# (link_head_1 and link_head_2 coincide at head_1 = 0, so a mount measurement
# taken at zero pitch is identical in both frames.)
HEAD2_TO_CAMERA_XYZ_RPY_ZYX_DEG = np.array(
    [0.023, 0.0, 0.066, 0.0, 90.0, 0.0],
    dtype=np.float64,
)

# Static URDF fallback for model=rby1a/model.urdf when head_0 = 0:
#   link_torso_5 -> link_head_0: xyz=(0.022, 0.0, 0.120073451525)
#   link_head_0  -> link_head_1: xyz=(0.0,   0.0, 0.080)   (head_0 revolute, z)
#   link_head_1  -> link_head_2: xyz=(0.0,   0.0, 0.0)     (head_1 revolute, y)
T5_TO_HEAD1_ZERO_HEAD0_XYZ_M = np.array([0.022, 0.0, 0.200073451525], dtype=np.float64)
# head_1 pitch used by every recorded pose below ("head": [0.000, 0.436]).
HEAD_1_PITCH_RAD_STATIC = 0.436


def _rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def rotation_from_euler_zyx_deg(roll_deg, pitch_deg, yaw_deg):
    """Return R = Rz(yaw) @ Ry(pitch) @ Rx(roll), angles in degrees."""
    roll, pitch, yaw = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    return _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)


def make_transform(translation, rotation=None):
    T = np.eye(4, dtype=np.float64)
    if rotation is not None:
        T[:3, :3] = np.asarray(rotation, dtype=np.float64)
    T[:3, 3] = np.asarray(translation, dtype=np.float64)
    return T


def transform_from_xyz_rpy_zyx_deg(xyz_rpy_deg):
    x, y, z, roll_deg, pitch_deg, yaw_deg = np.asarray(xyz_rpy_deg, dtype=np.float64)
    return make_transform(
        [x, y, z],
        rotation_from_euler_zyx_deg(roll_deg, pitch_deg, yaw_deg),
    )


def invert_transform(T):
    T = np.asarray(T, dtype=np.float64)
    T_inv = np.eye(4, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


# Pose of camera frame expressed in link_head_2 coordinates. It maps
# camera-frame points into head_2: p_head2 = T_HEAD2_FROM_CAMERA @ p_camera.
T_HEAD2_FROM_CAMERA = transform_from_xyz_rpy_zyx_deg(HEAD2_TO_CAMERA_XYZ_RPY_ZYX_DEG)

# Static camera->T5 transform for the recorded posture (head_0 = 0,
# head_1 = HEAD_1_PITCH_RAD_STATIC): p_t5 = CAMERA_TO_T5_STATIC @ p_camera.
T_T5_FROM_HEAD2_STATIC = make_transform(T5_TO_HEAD1_ZERO_HEAD0_XYZ_M) @ make_transform(
    [0.0, 0.0, 0.0], _rot_y(HEAD_1_PITCH_RAD_STATIC)
)
CAMERA_TO_T5_STATIC = T_T5_FROM_HEAD2_STATIC @ T_HEAD2_FROM_CAMERA
T5_TO_CAMERA_STATIC = invert_transform(CAMERA_TO_T5_STATIC)


def raw_camera_from_view_rotation_transform(view_rotation):
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


def camera_to_t5_for_view_rotation(camera_to_t5, view_rotation):
    """Adapt a raw camera->T5 transform for inference output in a rotated view.

    `inference.py` rotates RGB/depth and intrinsics before backprojection, so
    center_top_camera_m is expressed in that rotated analysis camera frame.
    """
    return np.asarray(camera_to_t5, dtype=np.float64) @ raw_camera_from_view_rotation_transform(view_rotation)


# Transform to use directly with inference.py's default --view-rotation cw90
# output: p_t5 = CAMERA_CW90_VIEW_TO_T5_STATIC @ p_center_top_camera_m.
CAMERA_CW90_VIEW_TO_T5_STATIC = camera_to_t5_for_view_rotation(CAMERA_TO_T5_STATIC, "cw90")


def compute_camera_to_t5_transform(dyn_model=None, dyn_state=None, q=None):
    """Return camera->T5 transform for the current robot state.

    When dynamics inputs are provided, link_torso_5 -> link_head_2 comes from FK
    so the head_0 yaw and head_1 pitch are respected. Without dynamics inputs,
    the static URDF transform for the recorded posture (head_0 = 0, head_1 =
    HEAD_1_PITCH_RAD_STATIC) is returned.
    """
    if dyn_model is None and dyn_state is None and q is None:
        return CAMERA_TO_T5_STATIC.copy()
    if dyn_model is None or dyn_state is None or q is None:
        raise ValueError("dyn_model, dyn_state, and q must be provided together.")

    dyn_state.set_q(q)
    dyn_model.compute_forward_kinematics(dyn_state)
    T_t5_from_head2 = dyn_model.compute_transformation(dyn_state, TORSO_INDEX, HEAD2_INDEX)
    return np.asarray(T_t5_from_head2, dtype=np.float64) @ T_HEAD2_FROM_CAMERA


def compute_camera_to_t5_for_view_rotation(view_rotation, dyn_model=None, dyn_state=None, q=None):
    return camera_to_t5_for_view_rotation(
        compute_camera_to_t5_transform(dyn_model, dyn_state, q),
        view_rotation,
    )


def transform_camera_point_to_t5(point_camera_m, camera_to_t5=None):
    """Transform one 3D camera-frame point into link_torso_5/T5 coordinates."""
    T = CAMERA_TO_T5_STATIC if camera_to_t5 is None else np.asarray(camera_to_t5, dtype=np.float64)
    p_camera = np.ones(4, dtype=np.float64)
    p_camera[:3] = np.asarray(point_camera_m, dtype=np.float64)
    return (T @ p_camera)[:3]


# ========================================================================================
# Recorded joint sets [rad], grouped by component (from measured robot states).
# Order per component follows model.torso_idx / right_arm_idx / left_arm_idx / head_idx.
# ========================================================================================

# --- Pose 1: "ready" (measured @ 15:46:30) ---
READY = {
    "torso":     [0.000,  0.000,  0.000,  0.349,  0.000,  0.000],
    "right_arm": [-0.175, -1.309, -0.262, -1.571, -2.618,  0.000, -0.175],
    "left_arm":  [-0.175,  1.309,  0.262, -1.571,  2.618,  0.000,  0.175],
    "head":      [0.000,  0.436],
}

# --- Pose 2: "ready_to_picking" (measured @ 15:44:42) ---
READY_TO_PICKING = {
    "torso":     [0.000,  0.000,  0.000,  0.349,  0.000,  0.000],
    "right_arm": [-0.111, -0.987, -0.205, -1.463, -2.454,  1.744,  0.30],
    "left_arm":  [-0.111,  0.987,  0.205, -1.463,  2.454,  1.744, -0.30],
    "head":      [0.000,  0.436],
}

# --- Pose 3: "start_to_picking" (measured @ 15:45:12) ---
# NOTE: This pose is now reached with a Cartesian impedance "push inward" motion
#       (see build_impedance_push_command) instead of a joint position command.
#       The values below are kept only as a reference for the approximate target.
START_TO_PICKING = {
    "torso":     [0.000,  0.000,  0.000,  0.349,  0.000,  0.000],
    "right_arm": [-0.171, -0.841, -0.153, -1.511, -2.403,  1.743,  3.2],
    "left_arm":  [-0.171,  0.841,  0.153, -1.511,  2.403,  1.743, -3.2],
    "head":      [0.000,  0.436],
}

# Poses executed with the joint position controller (ready -> ready_to_picking).
# 'start_to_picking' is handled afterwards by the Cartesian impedance push.
JOINT_SEQUENCE = [
    ("ready", READY),
    ("ready_to_picking", READY_TO_PICKING),
]


def build_pose_command(pose, minimum_time):
    """Build a RobotCommandBuilder that drives every component to `pose` using
    Joint Position Command builders (torso / right arm / left arm / head)."""
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


def send_once(robot, builder):
    """Send a single (one-shot) command and block until it finishes.

    This is the "send once" pattern: hand one command to the robot, wait for the
    handler to complete, and check the finish code (no persistent stream)."""
    feedback = robot.send_command(builder).get()
    return feedback.finish_code == rby.RobotCommandFeedback.FinishCode.Ok


def offset_translation(T, dy=0.0, dz=0.0):
    """Return a copy of the 4x4 transform T translated by `dy` along y and `dz`
    along z (in the reference frame; orientation unchanged)."""
    T_offset = T.copy()
    T_offset[1, 3] += dy
    T_offset[2, 3] += dz
    return T_offset


def build_dual_arm_impedance_command(
    dyn_model, dyn_state, q, reference_link, ref_index,
    inward, lift, translation_weight, hold_time, label,
):
    """Build a dual-arm Cartesian (task-space) impedance command.

    Targets are expressed in `reference_link` (index `ref_index`). Each hand
    target = its CURRENT EEF pose offset in that frame by:
      - y: right hand +inward, left hand -inward  -> grip toward the center
      - z: both hands +lift                        -> raise the object

    NOTE: the offsets are relative to the CURRENT pose, so passing inward=0 keeps
    the hands exactly where they are in y and only moves them by `lift` in z.

    Cartesian impedance provides compliance: the hands press toward the target
    with a bounded, spring-like force (F = K * pose_error) instead of rigidly
    tracking a position. `translation_weight` sets the task-space stiffness K
    (tighter = firmer grip / lift)."""
    # Current EEF poses expressed in the reference link, via FK.
    dyn_state.set_q(q)
    dyn_model.compute_forward_kinematics(dyn_state)
    T_ref2right = dyn_model.compute_transformation(dyn_state, ref_index, EE_RIGHT_INDEX)
    T_ref2left = dyn_model.compute_transformation(dyn_state, ref_index, EE_LEFT_INDEX)

    # Offset targets: inward along y (right +y / left -y) and up along z.
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

    def arm_impedance(link_name, T_target):
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


def build_impedance_push_command(dyn_model, dyn_state, q):
    """start_to_picking: push both hands inward to grip the object (no lift).
    Expressed in the torso-tip frame."""
    return build_dual_arm_impedance_command(
        dyn_model, dyn_state, q,
        reference_link=IMPEDANCE_REFERENCE_LINK, ref_index=TORSO_INDEX,
        inward=PUSH_DISTANCE, lift=0.0,
        translation_weight=IMPEDANCE_TRANSLATION_WEIGHT,
        hold_time=PUSH_HOLD_TIME, label="push",
    )


def build_impedance_lift_command(dyn_model, dyn_state, q):
    """ready_to_move: raise both hands STRAIGHT UP by LIFT_HEIGHT along base +z,
    SMOOTHLY. Uses a Cartesian IMPEDANCE controller that follows a timed
    trajectory (set_minimum_time + velocity limits) so the box rises gently over
    LIFT_MINIMUM_TIME instead of jerking up, while joint-space impedance
    (stiffness / damping / torque limit) holds the box firmly."""
    # Current EEF poses in the BASE frame (true vertical), via FK.
    dyn_state.set_q(q)
    dyn_model.compute_forward_kinematics(dyn_state)
    T_base2right = dyn_model.compute_transformation(dyn_state, BASE_INDEX, EE_RIGHT_INDEX)
    T_base2left = dyn_model.compute_transformation(dyn_state, BASE_INDEX, EE_LEFT_INDEX)

    # Targets: straight up by LIFT_HEIGHT (no horizontal change).
    T_right_target = offset_translation(T_base2right, 0.0, +LIFT_HEIGHT)
    T_left_target = offset_translation(T_base2left, 0.0, +LIFT_HEIGHT)

    print(
        f"[lift] right EEF z: {T_base2right[2, 3]:+.3f} -> {T_right_target[2, 3]:+.3f} m"
        f"  |  left EEF z: {T_base2left[2, 3]:+.3f} -> {T_left_target[2, 3]:+.3f} m"
        f"  (over {LIFT_MINIMUM_TIME:.0f}s)"
    )

    def arm_cartesian_impedance(link_name, T_target):
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


class FTMonitor:
    """Monitors force/torque sensor data through a state-update callback.

    `callback` is passed to robot.start_state_update() and is invoked in a
    background thread at FT_MONITOR_RATE Hz. It prints the latest right/left
    FT readings and keeps track of the peak force magnitude seen on each arm."""

    def __init__(self):
        self.samples = 0
        self.peak_force_right = 0.0
        self.peak_force_left = 0.0

    def callback(self, robot_state):
        ft_right = robot_state.ft_sensor_right
        ft_left = robot_state.ft_sensor_left

        # Force magnitude (Euclidean norm of the 3-axis force vector) in Newtons.
        force_right = float(np.linalg.norm(ft_right.force))
        force_left = float(np.linalg.norm(ft_left.force))

        self.samples += 1
        self.peak_force_right = max(self.peak_force_right, force_right)
        self.peak_force_left = max(self.peak_force_left, force_left)

        print(
            f"[FT] right | force {ft_right.force} |F|={force_right:6.2f}N "
            f"  ||  left | force {ft_left.force} |F|={force_left:6.2f}N "
        )


def main(address, model, power):
    robot = rby.create_robot(address, model)
    robot.connect()
    if not robot.is_connected():
        print("Robot is not connected")
        exit(1)

    # Bring the robot up: power, servo, clear faults, enable the control manager.
    if not robot.is_power_on(power):
        if not robot.power_on(power):
            print("Failed to power on")
            exit(1)
    if not robot.is_servo_on(".*"):
        if not robot.servo_on(".*"):
            print("Failed to servo on")
            exit(1)
    robot.reset_fault_control_manager()
    if not robot.enable_control_manager():
        print("Failed to enable control manager")
        exit(1)

    # Dynamics model used to compute current EEF poses (forward kinematics) for
    # the Cartesian impedance push. Joint order follows model.robot_joint_names.
    robot_model = robot.model()
    dyn_model = robot.get_dynamics()
    dyn_state = dyn_model.make_state(DYN_LINK_NAMES, robot_model.robot_joint_names)

    q_initial = robot.get_state().position
    camera_to_t5 = compute_camera_to_t5_transform(dyn_model, dyn_state, q_initial)
    print("[vision] camera->T5 transform (p_t5 = T @ p_camera):")
    print(camera_to_t5)

    # Start monitoring FT sensor data in the background (callback @ FT_MONITOR_RATE Hz).
    ft_monitor = FTMonitor()
    robot.start_state_update(ft_monitor.callback, FT_MONITOR_RATE)

    picking_done = False  # completion flag
    try:
        # 1) Joint position control: ready -> ready_to_picking.
        for name, pose in JOINT_SEQUENCE:
            print(f"[picking] moving to '{name}' (joint position) ...")
            if not send_once(robot, build_pose_command(pose, MINIMUM_TIME)):
                print(f"[picking] FAILED while moving to '{name}'. Aborting.")
                return picking_done
            print(f"[picking] reached '{name}'.")

        # 2) start_to_picking: Cartesian impedance push (both hands inward on y).
        print("[picking] pushing EEF inward with Cartesian impedance control ...")
        q = robot.get_state().position
        if not send_once(robot, build_impedance_push_command(dyn_model, dyn_state, q)):
            print("[picking] FAILED during Cartesian impedance push. Aborting.")
            return picking_done
        print("[picking] Cartesian impedance push done (start_to_picking).")

        # 3) ready_to_move: keep the inward grip and lift the box up by
        #    LIFT_HEIGHT, with a tighter stiffness so it is held firmly.
        print(f"[picking] lifting the box ~{LIFT_HEIGHT * 100:.0f} cm (ready_to_move) ...")
        q = robot.get_state().position
        if not send_once(robot, build_impedance_lift_command(dyn_model, dyn_state, q)):
            print("[picking] FAILED during lift (ready_to_move). Aborting.")
            return picking_done
        print("[picking] box lifted (ready_to_move).")

        # All steps done -> raise the completion flag.
        picking_done = True
        print("=" * 60)
        print(f"[picking] picking motion COMPLETED. done = {picking_done}")
        print("=" * 60)
    finally:
        # Always stop the background monitoring and report a short summary.
        robot.stop_state_update()
        print(
            f"[FT] monitoring stopped. samples={ft_monitor.samples}, "
            f"peak |F| right={ft_monitor.peak_force_right:.2f}N, "
            f"left={ft_monitor.peak_force_left:.2f}N"
        )

    return picking_done


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="picking_box")
    parser.add_argument("--address", type=str, required=True, help="Robot address")
    parser.add_argument(
        "--model", type=str, default="a", help="Robot Model Name (default: 'a')"
    )
    parser.add_argument(
        "--power",
        type=str,
        default=".*",
        help="Power device name regex pattern (default: '.*')",
    )
    args = parser.parse_args()

    main(address=args.address, model=args.model, power=args.power)
