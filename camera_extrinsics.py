"""Single source of truth for the D405 camera extrinsic on RB-Y1.

Both picking scripts import from here so a calibration change (remeasured
mount, different head pitch) can never leave one of them stale. Matrix naming
uses the target_from_source convention:

    p_target = T_TARGET_FROM_SOURCE @ [x, y, z, 1]
"""

from __future__ import annotations

from typing import Any

import numpy as np

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
# head_1 pitch used by every recorded picking pose ("head": [0.000, 0.436]).
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


# Pose of camera frame expressed in link_head_2 coordinates.
T_HEAD2_FROM_CAMERA = transform_from_xyz_rpy_zyx_deg(HEAD2_TO_CAMERA_XYZ_RPY_ZYX_DEG)

# Static camera->T5 transform for the recorded posture (head_0 = 0,
# head_1 = HEAD_1_PITCH_RAD_STATIC): p_t5 = CAMERA_TO_T5_STATIC @ p_camera.
T_T5_FROM_HEAD2_STATIC = make_transform(T5_TO_HEAD1_ZERO_HEAD0_XYZ_M) @ make_transform(
    [0.0, 0.0, 0.0],
    _rot_y(HEAD_1_PITCH_RAD_STATIC),
)
CAMERA_TO_T5_STATIC = T_T5_FROM_HEAD2_STATIC @ T_HEAD2_FROM_CAMERA
T5_TO_CAMERA_STATIC = invert_transform(CAMERA_TO_T5_STATIC)


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
    *,
    torso_index: int = 1,
    head2_index: int = 4,
) -> np.ndarray:
    """Return camera->T5 for the current robot state.

    With dynamics inputs, link_torso_5 -> link_head_2 comes from FK so the
    head_0 yaw and head_1 pitch are respected; the index defaults match the
    DYN_LINK_NAMES order used by the picking scripts. Without dynamics inputs,
    the static URDF transform for the recorded posture is returned.
    """
    if dyn_model is None and dyn_state is None and q is None:
        return CAMERA_TO_T5_STATIC.copy()
    if dyn_model is None or dyn_state is None or q is None:
        raise ValueError("dyn_model, dyn_state, and q must be provided together.")

    dyn_state.set_q(q)
    dyn_model.compute_forward_kinematics(dyn_state)
    T_t5_from_head2 = dyn_model.compute_transformation(dyn_state, torso_index, head2_index)
    return np.asarray(T_t5_from_head2, dtype=np.float64) @ T_HEAD2_FROM_CAMERA


def compute_camera_to_t5_for_view_rotation(
    view_rotation: str,
    dyn_model: Any = None,
    dyn_state: Any = None,
    q: Any = None,
    *,
    torso_index: int = 1,
    head2_index: int = 4,
) -> np.ndarray:
    return camera_to_t5_for_view_rotation(
        compute_camera_to_t5_transform(
            dyn_model, dyn_state, q, torso_index=torso_index, head2_index=head2_index
        ),
        view_rotation,
    )


def transform_camera_point_to_t5(point_camera_m: Any, camera_to_t5: Any = None) -> np.ndarray:
    """Transform one 3D camera-frame point into link_torso_5/T5 coordinates."""
    T = CAMERA_TO_T5_STATIC if camera_to_t5 is None else np.asarray(camera_to_t5, dtype=np.float64)
    p_camera = np.ones(4, dtype=np.float64)
    p_camera[:3] = np.asarray(point_camera_m, dtype=np.float64)
    return (T @ p_camera)[:3]
