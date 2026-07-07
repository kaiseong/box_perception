#!/usr/bin/env python3
"""Simulator entrypoint for the destination place/regrasp motion.

This intentionally runs the no-vision, no-gripper simulator sequence:

  ready -> start_to_picking -> inward_push_pick -> lift_pick ->
  place_lower -> release_push_reverse -> release_wait -> regrasp_push -> regrasp_lift

The first four stages put the simulated robot into the held-box posture that
place_and_picking.py normally reaches after picking. The physical Dynamixel
gripper setup is not exposed or called here.
"""

from __future__ import annotations

import argparse
import time
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
DEFAULT_SETUP_HOLD_SEC = 0.2


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
        "--setup-minimum-time",
        type=float,
        default=pap.MINIMUM_TIME,
        help="Minimum time for ready/start_to_picking setup joint moves.",
    )
    parser.add_argument(
        "--setup-hold-sec",
        type=float,
        default=DEFAULT_SETUP_HOLD_SEC,
        help="Small pause after each setup stage.",
    )
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
        "--skip-held-start",
        action="store_true",
        help="Skip ready/start_to_picking/push/lift setup and start from the current posture.",
    )
    parser.add_argument(
        "--allow-real",
        action="store_true",
        help="Allow non-local addresses. Use only when intentionally moving hardware.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.setup_minimum_time <= 0.0:
        raise SystemExit("--setup-minimum-time must be positive")
    if args.setup_hold_sec < 0.0:
        raise SystemExit("--setup-hold-sec must be non-negative")
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


def send_setup_pose(
    robot: object,
    pose_name: str,
    pose: dict[str, list[float]],
    *,
    minimum_time: float,
    hold_sec: float,
    command_timeout_margin_sec: float,
    min_command_timeout_sec: float,
) -> bool:
    stage = f"setup:{pose_name}"
    pap.print_stage(stage, f"joint position move over {minimum_time:.2f}s")
    ok = pap.send_stage(
        robot,
        pap.build_pose_command(pose, minimum_time),
        stage,
        timeout_sec=pap.stage_timeout_sec(
            minimum_time,
            min_timeout_sec=min_command_timeout_sec,
            margin_sec=command_timeout_margin_sec,
        ),
    )
    if not ok:
        pap.print_stage(stage, "FAILED; aborting")
        return False
    pap.print_stage(stage, "reached")
    if hold_sec > 0.0:
        time.sleep(hold_sec)
    return True


def prepare_held_box_pose(
    robot: object,
    dyn_model: object,
    dyn_state: object,
    *,
    setup_minimum_time: float,
    setup_hold_sec: float,
    push_ramp_time_sec: float,
    command_timeout_margin_sec: float,
    min_command_timeout_sec: float,
) -> bool:
    """Move the simulator to the post-pick held-box posture without gripper IO."""
    if not send_setup_pose(
        robot,
        "ready",
        pap.READY,
        minimum_time=setup_minimum_time,
        hold_sec=setup_hold_sec,
        command_timeout_margin_sec=command_timeout_margin_sec,
        min_command_timeout_sec=min_command_timeout_sec,
    ):
        return False
    if not send_setup_pose(
        robot,
        "start_to_picking",
        pap.START_TO_PICKING,
        minimum_time=setup_minimum_time,
        hold_sec=setup_hold_sec,
        command_timeout_margin_sec=command_timeout_margin_sec,
        min_command_timeout_sec=min_command_timeout_sec,
    ):
        return False

    pap.print_stage("setup:inward_push_pick", "building ramped target stream")
    q = robot.get_state().position
    if not pap.stream_impedance_push_stage(
        robot,
        dyn_model,
        dyn_state,
        q,
        target_inward=+pap.PUSH_DISTANCE,
        ramp_time_sec=push_ramp_time_sec,
        hold_time_sec=pap.PUSH_HOLD_TIME,
        stage="setup:inward_push_pick",
    ):
        pap.print_stage("setup:inward_push_pick", "FAILED; aborting")
        return False
    if setup_hold_sec > 0.0:
        time.sleep(setup_hold_sec)

    pap.print_stage("setup:lift_pick", "building target")
    q = robot.get_state().position
    if not pap.send_stage(
        robot,
        pap.build_impedance_lift_command(dyn_model, dyn_state, q),
        "setup:lift_pick",
        timeout_sec=pap.stage_timeout_sec(
            pap.LIFT_MINIMUM_TIME,
            pap.VERTICAL_MOVE_HOLD_TIME,
            min_timeout_sec=min_command_timeout_sec,
            margin_sec=command_timeout_margin_sec,
        ),
    ):
        pap.print_stage("setup:lift_pick", "FAILED; aborting")
        return False
    pap.print_stage("setup:lift_pick", "held-box posture reached")
    if setup_hold_sec > 0.0:
        time.sleep(setup_hold_sec)
    return True


def run(args: argparse.Namespace) -> bool:
    sequence = (
        "place_lower -> release_push_reverse -> release_wait -> regrasp_push -> regrasp_lift"
        if args.skip_held_start
        else "ready -> start_to_picking -> inward_push_pick -> lift_pick -> "
        "place_lower -> release_push_reverse -> release_wait -> regrasp_push -> regrasp_lift"
    )
    print(
        "[place_and_picking_sim] "
        f"address={args.address} model={args.model} "
        f"place_wait={args.place_wait_sec:.2f}s "
        f"push_ramp={args.push_ramp_time_sec:.2f}s"
    )
    print(f"[sequence] {sequence}")
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

    if not args.skip_held_start:
        if not prepare_held_box_pose(
            robot,
            dyn_model,
            dyn_state,
            setup_minimum_time=float(args.setup_minimum_time),
            setup_hold_sec=float(args.setup_hold_sec),
            push_ramp_time_sec=float(args.push_ramp_time_sec),
            command_timeout_margin_sec=float(args.command_timeout_margin_sec),
            min_command_timeout_sec=float(args.min_command_timeout_sec),
        ):
            print("[place_and_picking_sim] failed during held-box setup")
            return False
    else:
        print("[place_and_picking_sim] held-box setup skipped; using current posture")

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
