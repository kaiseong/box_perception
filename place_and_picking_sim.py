#!/usr/bin/env python3
"""Simulator entrypoint for the destination place/regrasp motion.

This intentionally runs only the no-vision, no-gripper destination sequence:

  place_lower -> release_push_reverse -> release_wait -> regrasp_push -> regrasp_lift

Use it to check the arm motion in the RBY1 simulator after the robot is already
in the held-box posture. The physical Dynamixel gripper setup is not exposed or
called here.
"""

from __future__ import annotations

import argparse
from urllib.parse import urlparse

import place_and_picking as pap

try:
    import rby1_sdk as rby
except ModuleNotFoundError:  # Allows --help and parser tests on non-robot dev hosts.
    rby = None


LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
DEFAULT_ADDRESS = "localhost:50051"
DEFAULT_MODEL = "m"
DEFAULT_POWER = ".*"
DEFAULT_SERVO = ".*"


def _host_from_address(address: str) -> str:
    parsed = urlparse(address if "://" in address else f"grpc://{address}")
    return parsed.hostname or address.split(":", 1)[0]


def is_local_address(address: str) -> bool:
    return _host_from_address(address) in LOCAL_HOSTS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="run the place_and_picking destination motion in the RBY1 simulator"
    )
    parser.add_argument(
        "--address",
        default=DEFAULT_ADDRESS,
        help="RBY1 simulator address. Non-local addresses require --allow-real.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Robot model name.")
    parser.add_argument("--power", default=DEFAULT_POWER, help="Power device regex.")
    parser.add_argument("--servo", default=DEFAULT_SERVO, help="Servo regex.")
    parser.add_argument(
        "--place-wait-sec",
        type=float,
        default=pap.PLACE_WAIT_AFTER_RELEASE_SEC,
        help="Wait after reverse release before regrasp. Default is 1.0 s.",
    )
    parser.add_argument(
        "--push-ramp-time-sec",
        type=float,
        default=pap.PUSH_RAMP_TIME,
        help="Time to ramp the inward/release impedance push target.",
    )
    parser.add_argument(
        "--command-timeout-margin-sec",
        type=float,
        default=pap.COMMAND_TIMEOUT_MARGIN_SEC,
        help="Extra watchdog slack added to each command's expected motion/hold time.",
    )
    parser.add_argument(
        "--min-command-timeout-sec",
        type=float,
        default=pap.COMMAND_TIMEOUT_MIN_SEC,
        help="Minimum watchdog timeout for any robot command stage.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sequence and defaults without connecting.",
    )
    parser.add_argument(
        "--skip-enable",
        action="store_true",
        help="Connect only; do not power/servo/enable control manager.",
    )
    parser.add_argument(
        "--allow-real",
        action="store_true",
        help="Allow non-local addresses. Use only when intentionally moving hardware.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.place_wait_sec < 0.0:
        raise SystemExit("--place-wait-sec must be non-negative")
    if args.push_ramp_time_sec < 0.0:
        raise SystemExit("--push-ramp-time-sec must be non-negative")
    if args.command_timeout_margin_sec < 0.0:
        raise SystemExit("--command-timeout-margin-sec must be non-negative")
    if args.min_command_timeout_sec <= 0.0:
        raise SystemExit("--min-command-timeout-sec must be positive")
    if not args.allow_real and not is_local_address(args.address):
        raise SystemExit(
            "Refusing non-local address for a simulator motion demo. "
            "Use simulator localhost:50051 or pass --allow-real intentionally."
        )


def connect_and_enable_robot(
    *,
    address: str,
    model: str,
    power: str,
    servo: str,
    skip_enable: bool,
) -> object | None:
    if rby is None:
        print("Failed to import rby1_sdk. Run this script in the RBY1 SDK/sim environment.")
        return None

    robot = rby.create_robot(address, model)
    connected = robot.connect()
    if connected is False:
        print(f"Failed to connect robot/sim: {address}")
        return None
    is_connected = getattr(robot, "is_connected", None)
    if is_connected is not None and not is_connected():
        print(f"Robot/sim is not connected: {address}")
        return None

    if skip_enable:
        return robot

    if not robot.is_power_on(power) and not robot.power_on(power):
        print(f"Failed to power on: {power}")
        return None
    if not robot.is_servo_on(servo) and not robot.servo_on(servo):
        print(f"Failed to servo on: {servo}")
        return None

    state_getter = getattr(robot, "get_control_manager_state", None)
    if state_getter is not None and rby is not None:
        state = state_getter().state
        control_state = getattr(getattr(rby, "ControlManagerState", None), "State", None)
        if control_state is not None and state in (
            getattr(control_state, "MajorFault", object()),
            getattr(control_state, "MinorFault", object()),
        ):
            if not robot.reset_fault_control_manager():
                print("Failed to reset control manager fault")
                return None
    else:
        resetter = getattr(robot, "reset_fault_control_manager", None)
        if resetter is not None:
            resetter()

    if not robot.enable_control_manager():
        print("Failed to enable control manager")
        return None
    return robot


def run(args: argparse.Namespace) -> bool:
    print(
        "[place_and_picking_sim] "
        f"address={args.address} model={args.model} "
        f"place_wait={args.place_wait_sec:.2f}s "
        f"push_ramp={args.push_ramp_time_sec:.2f}s"
    )
    print(
        "[sequence] "
        "place_lower -> release_push_reverse -> release_wait -> regrasp_push -> regrasp_lift"
    )
    print("[mode] no live vision, no initial pick, no Dynamixel gripper setup")
    if args.dry_run:
        return True

    robot = connect_and_enable_robot(
        address=args.address,
        model=args.model,
        power=args.power,
        servo=args.servo,
        skip_enable=args.skip_enable,
    )
    if robot is None:
        return False

    if pap.rby is None and rby is not None:
        pap.rby = rby

    robot_model = robot.model()
    dyn_model = robot.get_dynamics()
    dyn_state = dyn_model.make_state(pap.DYN_LINK_NAMES, robot_model.robot_joint_names)

    ok = pap.perform_place_regrasp_sequence(
        robot,
        dyn_model,
        dyn_state,
        push_ramp_time_sec=float(args.push_ramp_time_sec),
        place_wait_sec=float(args.place_wait_sec),
        command_timeout_margin_sec=float(args.command_timeout_margin_sec),
        min_command_timeout_sec=float(args.min_command_timeout_sec),
    )
    if ok:
        print("[place_and_picking_sim] completed")
    else:
        print("[place_and_picking_sim] failed")
    return ok


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    return 0 if run(args) else 1


if __name__ == "__main__":
    raise SystemExit(main())
