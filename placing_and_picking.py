#!/usr/bin/env python3
"""Target-based place and regrasp after picking_box_5 has lifted a box.

This script is intentionally narrower than place_and_picking.py:
it does not run vision, mobile-base alignment, initial picking, or gripper
homing. It starts from the final lift target exported by picking_box_5.py and
replays only the destination-table sequence:

  1. lower from the exported lift target by 0.08 m in base z
  2. release by moving both hands outward by the exact push distance
  3. wait on the table
  4. regrasp by returning both hands to the lowered push target
  5. lift back to the exported lift target

The lower/release/regrasp/lift targets are target-derived, not current-FK-
derived. Runtime FK is used only to verify that each target was reached.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

try:  # Keep parser/unit tests runnable on machines without the RBY1 SDK.
    import rby1_sdk as rby
except ModuleNotFoundError:  # pragma: no cover - exercised through tests
    rby = None


np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

DYN_LINK_NAMES = ["base", "link_torso_5", "ee_right", "ee_left", "link_head_2"]
BASE_INDEX, EE_RIGHT_INDEX, EE_LEFT_INDEX = 0, 2, 3

LIFT_TARGET_RECORD_VERSION = "box-perception-lift-target-v1"
DEFAULT_LIFT_TARGET_JSON = ".omx/runtime/latest_pick_lift_target.json"

PUSH_DISTANCE = 0.10
PLACE_LOWER_DELTA_M = 0.08
PLACE_WAIT_AFTER_RELEASE_SEC = 1.0

REFERENCE_LINK = "base"
COMMAND_STREAM_PRIORITY = 1
STREAM_PERIOD_SEC = 1.0 / 30.0
STREAM_COMMAND_HOLD_TIME_SEC = 1.0
STREAM_FINAL_HOLD_SEC = 3.0
FINAL_LIFT_HOLD_SEC = 100.0
STREAM_SEND_ATTEMPTS = 2
STREAM_RETRY_IDLE_SLEEP_SEC = 0.3
STREAM_CANCEL_SLEEP_SEC = 0.3

LOWER_RAMP_TIME_SEC = 1.0
RELEASE_RAMP_TIME_SEC = 0.5
REGRASP_RAMP_TIME_SEC = 0.5
LIFT_RAMP_TIME_SEC = 1.0

LIFT_LINEAR_VELOCITY_LIMIT = 0.4
LIFT_ANGULAR_VELOCITY_LIMIT = float(np.pi / 4)
LIFT_LINEAR_ACCELERATION_LIMIT = 0.8
LIFT_ANGULAR_ACCELERATION_LIMIT = float(np.pi)
LIFT_JOINT_STIFFNESS = [300.0] * 7
LIFT_JOINT_DAMPING_RATIO = 1.0
LIFT_JOINT_TORQUE_LIMIT = [70.0, 70.0, 70.0, 40.0, 10.0, 10.0, 8.0]

EEF_POSITION_TOLERANCE_M = 0.02
EEF_ROTATION_TOLERANCE_DEG = 8.0
EEF_WAIT_TIMEOUT_SEC = 4.0
EEF_POLL_PERIOD_SEC = 0.02


@dataclass(frozen=True)
class TargetPair:
    right: np.ndarray
    left: np.ndarray


@dataclass(frozen=True)
class PlaceRegraspTargets:
    lifted: TargetPair
    lowered: TargetPair
    released: TargetPair
    regrasped: TargetPair


def print_stage(stage: str, message: str) -> None:
    print(f"[stage] {stage} | {message}", flush=True)


def as_transform(value: Any, *, name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 transform, got shape={transform.shape}")
    if not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} contains non-finite values")
    return transform.copy()


def load_lift_target_record(path: str | Path) -> TargetPair:
    record_path = Path(path)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("format_version") != LIFT_TARGET_RECORD_VERSION:
        raise ValueError(
            f"Unsupported lift target format {record.get('format_version')!r}; "
            f"expected {LIFT_TARGET_RECORD_VERSION!r}"
        )
    if record.get("reference_link", REFERENCE_LINK) != REFERENCE_LINK:
        raise ValueError(
            f"Lift target reference_link must be {REFERENCE_LINK!r}, "
            f"got {record.get('reference_link')!r}"
        )
    return TargetPair(
        right=as_transform(record["right_target"], name="right_target"),
        left=as_transform(record["left_target"], name="left_target"),
    )


def offset_z(pair: TargetPair, dz_m: float) -> TargetPair:
    right = pair.right.copy()
    left = pair.left.copy()
    right[2, 3] += float(dz_m)
    left[2, 3] += float(dz_m)
    return TargetPair(right=right, left=left)


def hand_outward_y_directions(pair: TargetPair) -> tuple[float, float]:
    midpoint_y = 0.5 * (float(pair.right[1, 3]) + float(pair.left[1, 3]))
    right_dir = float(np.sign(float(pair.right[1, 3]) - midpoint_y))
    left_dir = float(np.sign(float(pair.left[1, 3]) - midpoint_y))
    if abs(right_dir) < 1e-9:
        right_dir = -1.0
    if abs(left_dir) < 1e-9:
        left_dir = +1.0
    if right_dir == left_dir:
        right_dir, left_dir = -1.0, +1.0
    return right_dir, left_dir


def offset_outward_y(pair: TargetPair, distance_m: float) -> TargetPair:
    right_dir, left_dir = hand_outward_y_directions(pair)
    right = pair.right.copy()
    left = pair.left.copy()
    right[1, 3] += right_dir * float(distance_m)
    left[1, 3] += left_dir * float(distance_m)
    return TargetPair(right=right, left=left)


def build_place_regrasp_target_chain(
    lifted: TargetPair,
    *,
    lower_delta_m: float = PLACE_LOWER_DELTA_M,
    push_distance_m: float = PUSH_DISTANCE,
) -> PlaceRegraspTargets:
    """Build all place/regrasp targets from the exported pick-lift target."""
    lowered = offset_z(lifted, -float(lower_delta_m))
    released = offset_outward_y(lowered, float(push_distance_m))
    return PlaceRegraspTargets(
        lifted=lifted,
        lowered=lowered,
        released=released,
        regrasped=lowered,
    )


def interpolate_transform(start: np.ndarray, end: np.ndarray, progress: float) -> np.ndarray:
    """Interpolate translation linearly and keep the target rotation fixed."""
    clamped = float(np.clip(progress, 0.0, 1.0))
    result = np.asarray(end, dtype=np.float64).copy()
    result[:3, 3] = (
        np.asarray(start, dtype=np.float64)[:3, 3] * (1.0 - clamped)
        + np.asarray(end, dtype=np.float64)[:3, 3] * clamped
    )
    return result


def build_dual_arm_impedance_target_command(
    right_target: np.ndarray,
    left_target: np.ndarray,
    *,
    hold_time_sec: float,
    minimum_time_sec: float = STREAM_PERIOD_SEC,
) -> Any:
    if rby is None:
        raise RuntimeError("rby1_sdk is required for robot motion")

    def arm_cartesian_impedance(link_name: str, target: np.ndarray) -> Any:
        return (
            rby.CartesianImpedanceControlCommandBuilder()
            .set_command_header(
                rby.CommandHeaderBuilder().set_control_hold_time(float(hold_time_sec))
            )
            .add_target(
                REFERENCE_LINK,
                link_name,
                np.asarray(target, dtype=np.float64),
                LIFT_LINEAR_VELOCITY_LIMIT,
                LIFT_ANGULAR_VELOCITY_LIMIT,
                LIFT_LINEAR_ACCELERATION_LIMIT,
                LIFT_ANGULAR_ACCELERATION_LIMIT,
            )
            .set_joint_stiffness(LIFT_JOINT_STIFFNESS)
            .set_joint_damping_ratio(LIFT_JOINT_DAMPING_RATIO)
            .set_joint_torque_limit(LIFT_JOINT_TORQUE_LIMIT)
            .set_minimum_time(max(0.01, float(minimum_time_sec)))
        )

    return rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_body_command(
            rby.BodyComponentBasedCommandBuilder()
            .set_right_arm_command(arm_cartesian_impedance("ee_right", right_target))
            .set_left_arm_command(arm_cartesian_impedance("ee_left", left_target))
        )
    )


def stream_target_ramp_stage(
    robot: Any,
    *,
    start: TargetPair,
    end: TargetPair,
    stage: str,
    ramp_time_sec: float,
    stream_period_sec: float = STREAM_PERIOD_SEC,
    final_hold_sec: float = STREAM_FINAL_HOLD_SEC,
) -> bool:
    """Stream a target-space ramp with the same Cartesian impedance family as lift."""
    period = max(0.01, float(stream_period_sec))
    ramp_time = max(0.0, float(ramp_time_sec))
    ramp_hold = max(period * 2.0, STREAM_COMMAND_HOLD_TIME_SEC)
    final_hold = max(float(final_hold_sec), ramp_hold)

    last_successful_sends = 0

    def run_ramp_once(stream: Any) -> int:
        nonlocal last_successful_sends
        last_successful_sends = 0

        def send_progress(progress: float, *, hold_time_sec: float) -> None:
            nonlocal last_successful_sends
            right_target = interpolate_transform(start.right, end.right, progress)
            left_target = interpolate_transform(start.left, end.left, progress)
            stream.send_command(
                build_dual_arm_impedance_target_command(
                    right_target,
                    left_target,
                    hold_time_sec=hold_time_sec,
                    minimum_time_sec=period,
                )
            )
            last_successful_sends += 1

        print_stage(stage, f"streaming target ramp over {ramp_time:.2f}s")
        start_time = time.monotonic()
        if ramp_time <= 0.0:
            send_progress(1.0, hold_time_sec=final_hold)
        else:
            send_progress(0.0, hold_time_sec=ramp_hold)
            next_send = start_time + period
            deadline = start_time + ramp_time
            while time.monotonic() < deadline:
                now = time.monotonic()
                if now >= next_send:
                    progress = min(1.0, max(0.0, (now - start_time) / ramp_time))
                    send_progress(progress, hold_time_sec=ramp_hold)
                    next_send = now + period
                sleep_sec = max(0.0, min(next_send, deadline) - time.monotonic())
                time.sleep(min(0.005, sleep_sec))
        send_progress(1.0, hold_time_sec=final_hold)
        return last_successful_sends

    for attempt in range(1, STREAM_SEND_ATTEMPTS + 1):
        try:
            stream = robot.create_command_stream(priority=COMMAND_STREAM_PRIORITY)
        except Exception as exc:  # pragma: no cover - SDK/hardware dependent
            print_stage(stage, f"failed to create command stream: {exc}")
            return False
        try:
            sends = run_ramp_once(stream)
        except Exception as exc:  # pragma: no cover - SDK/hardware dependent
            print_stage(stage, f"FAILED: stream send failed on attempt {attempt}: {exc}")
            if attempt >= STREAM_SEND_ATTEMPTS or last_successful_sends > 0:
                return False
            print_stage(stage, "cancel_control; retrying target stream from idle")
            if not cancel_control_for_next_stream(
                robot,
                stage,
                sleep_sec=STREAM_RETRY_IDLE_SLEEP_SEC,
            ):
                return False
            continue
        print_stage(stage, f"ramp sent; stream sends={sends}")
        return True
    return False


def cancel_control_for_next_stream(
    robot: Any,
    stage: str,
    *,
    sleep_sec: float = STREAM_CANCEL_SLEEP_SEC,
) -> bool:
    """Release the held body stream so the next body stream starts from idle."""
    try:
        robot.cancel_control()
    except Exception as exc:  # pragma: no cover - SDK/hardware dependent
        print_stage(stage, f"FAILED: cancel_control before next stream failed: {exc}")
        return False
    time.sleep(max(0.0, float(sleep_sec)))
    return True


def rotation_error_deg(current: np.ndarray, target: np.ndarray) -> float:
    delta = (
        np.asarray(target, dtype=np.float64)[:3, :3].T
        @ np.asarray(current, dtype=np.float64)[:3, :3]
    )
    trace = float(np.trace(delta))
    angle = np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(angle))


def wait_for_eef_targets(
    robot: Any,
    dyn_model: Any,
    dyn_state: Any,
    target: TargetPair,
    *,
    stage: str,
    timeout_sec: float = EEF_WAIT_TIMEOUT_SEC,
    position_tolerance_m: float = EEF_POSITION_TOLERANCE_M,
    rotation_tolerance_deg: float = EEF_ROTATION_TOLERANCE_DEG,
) -> bool:
    deadline = time.monotonic() + max(0.1, float(timeout_sec))
    best_pos_m = float("inf")
    best_rot_deg = float("inf")
    while time.monotonic() < deadline:
        q = robot.get_state().position
        dyn_state.set_q(q)
        dyn_model.compute_forward_kinematics(dyn_state)
        current_right = np.asarray(
            dyn_model.compute_transformation(dyn_state, BASE_INDEX, EE_RIGHT_INDEX),
            dtype=np.float64,
        )
        current_left = np.asarray(
            dyn_model.compute_transformation(dyn_state, BASE_INDEX, EE_LEFT_INDEX),
            dtype=np.float64,
        )
        pos_m = max(
            float(np.linalg.norm(current_right[:3, 3] - target.right[:3, 3])),
            float(np.linalg.norm(current_left[:3, 3] - target.left[:3, 3])),
        )
        rot_deg = max(
            rotation_error_deg(current_right, target.right),
            rotation_error_deg(current_left, target.left),
        )
        best_pos_m = min(best_pos_m, pos_m)
        best_rot_deg = min(best_rot_deg, rot_deg)
        if pos_m <= float(position_tolerance_m) and rot_deg <= float(rotation_tolerance_deg):
            print_stage(stage, f"FK target reached (pos={pos_m * 100:.1f}cm rot={rot_deg:.1f}deg)")
            return True
        time.sleep(EEF_POLL_PERIOD_SEC)

    print_stage(
        stage,
        f"FAILED: FK target not reached (best pos={best_pos_m * 100:.1f}cm "
        f"rot={best_rot_deg:.1f}deg)",
    )
    return False


def perform_place_regrasp_sequence(
    robot: Any,
    dyn_model: Any,
    dyn_state: Any,
    lifted: TargetPair,
    *,
    place_lower_delta_m: float = PLACE_LOWER_DELTA_M,
    place_wait_sec: float = PLACE_WAIT_AFTER_RELEASE_SEC,
    lower_ramp_time_sec: float = LOWER_RAMP_TIME_SEC,
    release_ramp_time_sec: float = RELEASE_RAMP_TIME_SEC,
    regrasp_ramp_time_sec: float = REGRASP_RAMP_TIME_SEC,
    lift_ramp_time_sec: float = LIFT_RAMP_TIME_SEC,
    eef_wait_timeout_sec: float = EEF_WAIT_TIMEOUT_SEC,
) -> bool:
    targets = build_place_regrasp_target_chain(
        lifted,
        lower_delta_m=float(place_lower_delta_m),
    )

    stages = [
        ("1/5 place_lower", targets.lifted, targets.lowered, lower_ramp_time_sec),
        ("2/5 release_open", targets.lowered, targets.released, release_ramp_time_sec),
        ("4/5 regrasp_push", targets.released, targets.regrasped, regrasp_ramp_time_sec),
        ("5/5 regrasp_lift", targets.regrasped, targets.lifted, lift_ramp_time_sec),
    ]
    wait_after_stage = "2/5 release_open"

    print_stage("placing_and_picking", "cancel_control before first target stream")
    if not cancel_control_for_next_stream(robot, "placing_and_picking"):
        return False

    for index, (stage, start, end, ramp_time) in enumerate(stages):
        print_stage(stage, "building target ramp")
        if not stream_target_ramp_stage(
            robot,
            start=start,
            end=end,
            stage=stage,
            ramp_time_sec=float(ramp_time),
            final_hold_sec=(
                FINAL_LIFT_HOLD_SEC
                if stage == "5/5 regrasp_lift"
                else STREAM_FINAL_HOLD_SEC
            ),
        ):
            return False
        if not wait_for_eef_targets(
            robot,
            dyn_model,
            dyn_state,
            end,
            stage=stage,
            timeout_sec=float(eef_wait_timeout_sec),
        ):
            return False
        if stage == wait_after_stage:
            print_stage("3/5 place_wait", f"waiting {float(place_wait_sec):.2f}s before regrasp")
            if place_wait_sec > 0.0:
                time.sleep(float(place_wait_sec))
        if index < len(stages) - 1:
            print_stage(stage, "cancel_control for next target stream")
            if not cancel_control_for_next_stream(robot, stage):
                return False

    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Place a lifted box on the destination table and regrasp it from exported targets."
    )
    parser.add_argument("--address", type=str, required=True, help="Robot address")
    parser.add_argument("--model", type=str, default="m", help="Robot model name")
    parser.add_argument("--power", type=str, default=".*", help="Power device name regex")
    parser.add_argument(
        "--lift-target-json",
        type=str,
        default=DEFAULT_LIFT_TARGET_JSON,
        help="Target JSON exported by picking_box_5.py --lift-target-output.",
    )
    parser.add_argument(
        "--place-wait-sec",
        type=float,
        default=PLACE_WAIT_AFTER_RELEASE_SEC,
        help="Seconds to wait after opening/releasing before regrasp.",
    )
    parser.add_argument(
        "--place-lower-delta-m",
        type=float,
        default=PLACE_LOWER_DELTA_M,
        help="Base-frame z distance to lower from the exported lift target.",
    )
    parser.add_argument("--lower-ramp-time-sec", type=float, default=LOWER_RAMP_TIME_SEC)
    parser.add_argument("--release-ramp-time-sec", type=float, default=RELEASE_RAMP_TIME_SEC)
    parser.add_argument("--regrasp-ramp-time-sec", type=float, default=REGRASP_RAMP_TIME_SEC)
    parser.add_argument("--lift-ramp-time-sec", type=float, default=LIFT_RAMP_TIME_SEC)
    parser.add_argument(
        "--eef-wait-timeout-sec",
        type=float,
        default=EEF_WAIT_TIMEOUT_SEC,
        help="FK target arrival timeout per streamed stage.",
    )
    return parser.parse_args(argv)


def main(
    *,
    address: str,
    model: str,
    power: str,
    lift_target_json: str | Path,
    place_lower_delta_m: float,
    place_wait_sec: float,
    lower_ramp_time_sec: float,
    release_ramp_time_sec: float,
    regrasp_ramp_time_sec: float,
    lift_ramp_time_sec: float,
    eef_wait_timeout_sec: float,
) -> bool:
    if rby is None:
        print("rby1_sdk is required for robot motion")
        return False

    try:
        lifted = load_lift_target_record(lift_target_json)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Failed to load lift target JSON {lift_target_json}: {exc}")
        return False
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

    print_stage("placing_and_picking", f"loaded target: {lift_target_json}")
    done = perform_place_regrasp_sequence(
        robot,
        dyn_model,
        dyn_state,
        lifted,
        place_lower_delta_m=float(place_lower_delta_m),
        place_wait_sec=float(place_wait_sec),
        lower_ramp_time_sec=float(lower_ramp_time_sec),
        release_ramp_time_sec=float(release_ramp_time_sec),
        regrasp_ramp_time_sec=float(regrasp_ramp_time_sec),
        lift_ramp_time_sec=float(lift_ramp_time_sec),
        eef_wait_timeout_sec=float(eef_wait_timeout_sec),
    )
    if done:
        print("[placing_and_picking] completed")
    return done


def run_cli(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.place_wait_sec < 0.0:
        raise SystemExit("--place-wait-sec must be non-negative")
    if args.place_lower_delta_m <= 0.0:
        raise SystemExit("--place-lower-delta-m must be positive")
    for attr in (
        "lower_ramp_time_sec",
        "release_ramp_time_sec",
        "regrasp_ramp_time_sec",
        "lift_ramp_time_sec",
    ):
        if float(getattr(args, attr)) < 0.0:
            raise SystemExit(f"--{attr.replace('_', '-')} must be non-negative")
    if args.eef_wait_timeout_sec <= 0.0:
        raise SystemExit("--eef-wait-timeout-sec must be positive")

    ok = main(
        address=args.address,
        model=args.model,
        power=args.power,
        lift_target_json=args.lift_target_json,
        place_lower_delta_m=float(args.place_lower_delta_m),
        place_wait_sec=float(args.place_wait_sec),
        lower_ramp_time_sec=float(args.lower_ramp_time_sec),
        release_ramp_time_sec=float(args.release_ramp_time_sec),
        regrasp_ramp_time_sec=float(args.regrasp_ramp_time_sec),
        lift_ramp_time_sec=float(args.lift_ramp_time_sec),
        eef_wait_timeout_sec=float(args.eef_wait_timeout_sec),
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run_cli())
