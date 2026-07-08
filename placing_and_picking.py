#!/usr/bin/env python3
"""Place the currently held pallet box and open both arms outward.

This script intentionally reuses only the destination-table place/release
pieces from place_and_picking.py:

  1. lower both hands by the same reverse-lift Cartesian impedance command
  2. move both hands outward along the grasp y-axis by reversing inward_push

It does not run vision, mobile-base alignment, gripper homing, regrasp, or lift.
The sequence starts from the current held-box posture.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

import place_and_picking as _place


np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

rby = _place.rby

DYN_LINK_NAMES = _place.DYN_LINK_NAMES

PUSH_DISTANCE = _place.PUSH_DISTANCE
PUSH_RAMP_TIME = _place.PUSH_RAMP_TIME
PUSH_HOLD_TIME = _place.PUSH_HOLD_TIME
RELEASE_RAMP_TIME_SEC = PUSH_RAMP_TIME

LIFT_HEIGHT = _place.LIFT_HEIGHT
LIFT_MINIMUM_TIME = _place.LIFT_MINIMUM_TIME
VERTICAL_MOVE_HOLD_TIME = _place.VERTICAL_MOVE_HOLD_TIME
PLACE_LOWER_DELTA_M = LIFT_HEIGHT
LOWER_RAMP_TIME_SEC = LIFT_MINIMUM_TIME

COMMAND_TIMEOUT_MARGIN_SEC = _place.COMMAND_TIMEOUT_MARGIN_SEC
COMMAND_TIMEOUT_MIN_SEC = _place.COMMAND_TIMEOUT_MIN_SEC

# Legacy import compatibility for older compare.py versions. The current
# placing_and_picking.py does not read a lift-target JSON.
DEFAULT_LIFT_TARGET_JSON = ".omx/runtime/latest_pick_lift_target.json"
EEF_WAIT_TIMEOUT_SEC = COMMAND_TIMEOUT_MIN_SEC

print_stage = _place.print_stage
stage_timeout_sec = _place.stage_timeout_sec
build_impedance_lower_command = _place.build_impedance_lower_command
stream_impedance_push_stage = _place.stream_impedance_push_stage
send_stage = _place.send_stage


def perform_place_release_sequence(
    robot: Any,
    dyn_model: Any,
    dyn_state: Any,
    *,
    push_ramp_time_sec: float = PUSH_RAMP_TIME,
    command_timeout_margin_sec: float = COMMAND_TIMEOUT_MARGIN_SEC,
    min_command_timeout_sec: float = COMMAND_TIMEOUT_MIN_SEC,
) -> bool:
    """Lower the held box, then open both arms outward along the grasp y-axis."""
    print_stage("1/2 place_lower", "building reverse-lift target")
    q = robot.get_state().position
    if not send_stage(
        robot,
        build_impedance_lower_command(dyn_model, dyn_state, q),
        "1/2 place_lower",
        timeout_sec=stage_timeout_sec(
            LIFT_MINIMUM_TIME,
            VERTICAL_MOVE_HOLD_TIME,
            min_timeout_sec=float(min_command_timeout_sec),
            margin_sec=float(command_timeout_margin_sec),
        ),
    ):
        print_stage("1/2 place_lower", "FAILED; aborting")
        return False

    print_stage("2/2 release_open", "ramping inward target back to zero")
    q = robot.get_state().position
    if not stream_impedance_push_stage(
        robot,
        dyn_model,
        dyn_state,
        q,
        target_inward=-PUSH_DISTANCE,
        ramp_time_sec=float(push_ramp_time_sec),
        hold_time_sec=PUSH_HOLD_TIME,
        stage="2/2 release_open",
    ):
        print_stage("2/2 release_open", "FAILED; aborting")
        return False

    print_stage("2/2 release_open", "done")
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Place the currently held box and open both arms outward. "
            "No vision, gripper homing, regrasp, or lift is run."
        )
    )
    parser.add_argument("--address", type=str, required=True, help="Robot address")
    parser.add_argument("--model", type=str, default="m", help="Robot model name")
    parser.add_argument("--power", type=str, default=".*", help="Power device name regex")
    parser.add_argument(
        "--push-ramp-time-sec",
        type=float,
        default=PUSH_RAMP_TIME,
        help="Time to ramp the y-axis outward release motion.",
    )
    parser.add_argument(
        "--command-timeout-margin-sec",
        type=float,
        default=COMMAND_TIMEOUT_MARGIN_SEC,
        help="Additional timeout margin around expected place motion time.",
    )
    parser.add_argument(
        "--min-command-timeout-sec",
        type=float,
        default=COMMAND_TIMEOUT_MIN_SEC,
        help="Minimum timeout for the place-lower command.",
    )
    return parser.parse_args(argv)


def main(
    *,
    address: str,
    model: str,
    power: str,
    push_ramp_time_sec: float,
    command_timeout_margin_sec: float,
    min_command_timeout_sec: float,
) -> bool:
    if rby is None:
        print("Failed to import rby1_sdk. Run this script in the RBY1 SDK environment.")
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

    ok = perform_place_release_sequence(
        robot,
        dyn_model,
        dyn_state,
        push_ramp_time_sec=float(push_ramp_time_sec),
        command_timeout_margin_sec=float(command_timeout_margin_sec),
        min_command_timeout_sec=float(min_command_timeout_sec),
    )
    if ok:
        print("[placing_and_picking] completed")
    return ok


def run_cli(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.push_ramp_time_sec < 0.0:
        raise SystemExit("--push-ramp-time-sec must be non-negative")
    if args.command_timeout_margin_sec < 0.0:
        raise SystemExit("--command-timeout-margin-sec must be non-negative")
    if args.min_command_timeout_sec <= 0.0:
        raise SystemExit("--min-command-timeout-sec must be positive")

    ok = main(
        address=args.address,
        model=args.model,
        power=args.power,
        push_ramp_time_sec=float(args.push_ramp_time_sec),
        command_timeout_margin_sec=float(args.command_timeout_margin_sec),
        min_command_timeout_sec=float(args.min_command_timeout_sec),
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run_cli())
