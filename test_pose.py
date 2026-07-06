#!/usr/bin/env python3
"""Pose-only simulation demo for picking_box_2 postures.

This script is for visual motion checks in the RBY1 simulator. It does not run
vision or FT monitoring. Default mode runs:
ready -> vision_pre_push_demo -> inward impedance push -> lift.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import time
from urllib.parse import urlparse

import numpy as np
import rby1_sdk as rby

from picking_box_2 import (
    DYN_LINK_NAMES,
    COMMAND_TIMEOUT_MARGIN_SEC,
    COMMAND_TIMEOUT_MIN_SEC,
    IMPEDANCE_LIFT_REFERENCE_LINK,
    IMPEDANCE_REFERENCE_LINK,
    JOINT_SEQUENCE,
    LIFT_HOLD_TIME,
    LIFT_MINIMUM_TIME,
    MINIMUM_TIME,
    PUSH_HOLD_TIME,
    READY,
    START_TO_PICKING,
    VISION_APPROACH_HOLD_TIME,
    VISION_APPROACH_MINIMUM_TIME,
    VISION_PRE_PUSH_REFERENCE_LINK,
    build_impedance_lift_command,
    build_impedance_push_command,
    build_pose_command,
    build_vision_pre_push_command,
    cancel_timed_out_command,
    start_to_picking_reference_targets,
    stage_timeout_sec,
    wait_for_command_feedback,
)


LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
DEFAULT_ADDRESS = "localhost:50051"
DEFAULT_MODEL = "a"
DEFAULT_POWER = ".*"
DEFAULT_SERVO = ".*"
DEFAULT_MODE = "full-pick"
DEFAULT_HOLD_SEC = 1.0
DEFAULT_PRIORITY = 1
DEFAULT_BOX_X_OFFSET = 0.0


@dataclass(frozen=True)
class NamedPose:
    name: str
    pose: dict[str, list[float]]


def _host_from_address(address: str) -> str:
    parsed = urlparse(address if "://" in address else f"grpc://{address}")
    return parsed.hostname or address.split(":", 1)[0]


def is_local_address(address: str) -> bool:
    return _host_from_address(address) in LOCAL_HOSTS


def pose_sequence(mode: str) -> list[NamedPose]:
    if mode == "full-pick":
        return [NamedPose(name, pose) for name, pose in JOINT_SEQUENCE]
    if mode == "picking2":
        return [NamedPose(name, pose) for name, pose in JOINT_SEQUENCE]
    if mode == "start-to-picking":
        return [
            NamedPose("ready", READY),
            NamedPose("start_to_picking", START_TO_PICKING),
        ]
    if mode == "start-only":
        return [NamedPose("start_to_picking", START_TO_PICKING)]
    raise ValueError(f"unsupported mode: {mode}")


def mode_runs_push_lift(mode: str) -> bool:
    return mode == "full-pick"


def print_pose_values(sequence: list[NamedPose]) -> None:
    print("[pose defaults]")
    for item in sequence:
        print(f"  {item.name}")
        for group in ("torso", "right_arm", "left_arm", "head"):
            values = np.asarray(item.pose[group], dtype=np.float64)
            print(f"    {group:>9}: {np.array2string(values, precision=3, suppress_small=False)}")


def connect_and_enable_robot(
    *,
    address: str,
    model: str,
    power: str,
    servo: str,
    skip_enable: bool,
) -> object:
    robot = rby.create_robot(address, model)
    if not robot.connect():
        raise RuntimeError(f"failed to connect robot/sim: {address}")

    if skip_enable:
        return robot

    if not robot.is_power_on(power) and not robot.power_on(power):
        raise RuntimeError(f"failed to power on: {power}")
    if not robot.is_servo_on(servo) and not robot.servo_on(servo):
        raise RuntimeError(f"failed to servo on: {servo}")

    state = robot.get_control_manager_state().state
    if state in (
        rby.ControlManagerState.State.MajorFault,
        rby.ControlManagerState.State.MinorFault,
    ):
        if not robot.reset_fault_control_manager():
            raise RuntimeError("failed to reset control manager fault")
    if not robot.enable_control_manager():
        raise RuntimeError("failed to enable control manager")
    return robot


def send_builder_stage(
    robot: object,
    builder: object,
    name: str,
    priority: int,
    *,
    timeout_sec: float,
) -> None:
    print(f"[stage] {name} | sending command")
    command = robot.send_command(builder, priority)
    print(f"[stage] {name} | waiting for finish_code (timeout={timeout_sec:.1f}s)")
    feedback = wait_for_command_feedback(command, name, timeout_sec=timeout_sec)
    if feedback is None:
        cancel_timed_out_command(robot, command, name)
        raise RuntimeError(f"command timed out: {name}")
    print(f"[stage] {name} | finish_code={feedback.finish_code}")
    if feedback.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
        raise RuntimeError(f"command failed: {name}")


def send_pose(
    robot: object,
    named_pose: NamedPose,
    minimum_time: float,
    priority: int,
    *,
    timeout_sec: float,
) -> None:
    print(f"[pose] moving to {named_pose.name} over {minimum_time:.2f}s")
    send_builder_stage(
        robot,
        build_pose_command(named_pose.pose, minimum_time),
        f"pose:{named_pose.name}",
        priority,
        timeout_sec=timeout_sec,
    )
    print(f"[pose] {named_pose.name} reached")


def send_cartesian_stage(
    robot: object,
    builder: object,
    name: str,
    priority: int,
    *,
    timeout_sec: float,
) -> None:
    print(f"[cartesian] running {name}")
    send_builder_stage(robot, builder, name, priority, timeout_sec=timeout_sec)


def build_demo_vision_pre_push_command(
    dyn_model: object,
    dyn_state: object,
    robot_model: object,
    q: np.ndarray,
    box_x_offset: float,
) -> object:
    right_ref, left_ref = start_to_picking_reference_targets(
        dyn_model,
        dyn_state,
        robot_model,
        q,
    )
    midpoint_xy = 0.5 * (right_ref[:2, 3] + left_ref[:2, 3])
    box_center_base_m = np.array(
        [midpoint_xy[0] + box_x_offset, midpoint_xy[1], 0.0],
        dtype=np.float64,
    )
    print(
        f"[vision_pre_push] demo reference={VISION_PRE_PUSH_REFERENCE_LINK} "
        "center_base_xy="
        f"{np.array2string(box_center_base_m[:2], precision=3)} "
        f"(reference_midpoint_xy={np.array2string(midpoint_xy, precision=3)}, "
        f"box_x_offset={box_x_offset:+.3f} m)"
    )
    return build_vision_pre_push_command(
        dyn_model,
        dyn_state,
        robot_model,
        q,
        box_center_base_m,
        approach_time=VISION_APPROACH_MINIMUM_TIME,
        hold_time=VISION_APPROACH_HOLD_TIME,
        max_reference_xy_shift_m=max(0.001, abs(box_x_offset) + 0.001),
        midpoint_offset_xy_m=(0.0, 0.0),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="picking_box_2 pose simulation demo")
    parser.add_argument(
        "--address",
        default=DEFAULT_ADDRESS,
        help="RBY1 simulator address. Non-local addresses require --allow-real.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Robot model name.")
    parser.add_argument("--power", default=DEFAULT_POWER, help="Power device regex.")
    parser.add_argument("--servo", default=DEFAULT_SERVO, help="Servo regex.")
    parser.add_argument(
        "--mode",
        choices=(
            "full-pick",
            "start-to-picking",
            "picking2",
            "start-only",
        ),
        default=DEFAULT_MODE,
        help=(
            "full-pick: picking_box_2 ready -> vision_pre_push_demo -> inward push -> lift. "
            "start-to-picking: ready -> start_to_picking preview. "
            "picking2: current picking_box_2 ready sequence. "
            "start-only: start_to_picking pose only."
        ),
    )
    parser.add_argument("--minimum-time", type=float, default=MINIMUM_TIME)
    parser.add_argument("--hold-sec", type=float, default=DEFAULT_HOLD_SEC)
    parser.add_argument("--priority", type=int, default=DEFAULT_PRIORITY)
    parser.add_argument(
        "--command-timeout-margin-sec",
        type=float,
        default=COMMAND_TIMEOUT_MARGIN_SEC,
        help="Extra watchdog slack added to each demo command's expected motion/hold time.",
    )
    parser.add_argument(
        "--min-command-timeout-sec",
        type=float,
        default=COMMAND_TIMEOUT_MIN_SEC,
        help="Minimum watchdog timeout for any demo command stage.",
    )
    parser.add_argument(
        "--box-x-offset",
        "--box-center-offset-x-m",
        dest="box_x_offset",
        type=float,
        default=DEFAULT_BOX_X_OFFSET,
        help=(
            "Base-frame x offset added to the demo box center used by full-pick "
            "vision_pre_push. Positive values move both hand targets along +x."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sequence and deltas without connecting.",
    )
    parser.add_argument(
        "--skip-enable",
        action="store_true",
        help="Connect only; do not power/servo/enable control manager.",
    )
    parser.add_argument(
        "--allow-real",
        action="store_true",
        help="Allow non-localhost addresses. Use only when intentionally moving hardware.",
    )
    args = parser.parse_args()
    if args.command_timeout_margin_sec < 0.0:
        raise SystemExit("--command-timeout-margin-sec must be non-negative")
    if args.min_command_timeout_sec <= 0.0:
        raise SystemExit("--min-command-timeout-sec must be positive")

    sequence = pose_sequence(args.mode)
    print(
        "[defaults] "
        f"address={args.address} model={args.model} mode={args.mode} "
        f"minimum_time={args.minimum_time} hold_sec={args.hold_sec} "
        f"box_x_offset={args.box_x_offset:+.3f} "
        f"timeout_min={args.min_command_timeout_sec:.1f}s "
        f"timeout_margin={args.command_timeout_margin_sec:.1f}s"
    )
    print("[sequence]", " -> ".join(item.name for item in sequence))
    if mode_runs_push_lift(args.mode):
        print(
            "[cartesian sequence] "
            f"vision_pre_push_demo({VISION_PRE_PUSH_REFERENCE_LINK}) -> "
            f"inward_push({IMPEDANCE_REFERENCE_LINK}) -> "
            f"lift({IMPEDANCE_LIFT_REFERENCE_LINK})"
        )
    print_pose_values(sequence)

    if args.dry_run:
        return 0
    if not args.allow_real and not is_local_address(args.address):
        raise SystemExit(
            "Refusing non-local address for a pose demo. "
            "Use simulator localhost:50051 or pass --allow-real intentionally."
        )

    robot = connect_and_enable_robot(
        address=args.address,
        model=args.model,
        power=args.power,
        servo=args.servo,
        skip_enable=args.skip_enable,
    )
    for item in sequence:
        send_pose(
            robot,
            item,
            float(args.minimum_time),
            int(args.priority),
            timeout_sec=stage_timeout_sec(
                float(args.minimum_time),
                min_timeout_sec=float(args.min_command_timeout_sec),
                margin_sec=float(args.command_timeout_margin_sec),
            ),
        )
        if args.hold_sec > 0:
            time.sleep(float(args.hold_sec))
    if mode_runs_push_lift(args.mode):
        robot_model = robot.model()
        dyn_model = robot.get_dynamics()
        dyn_state = dyn_model.make_state(DYN_LINK_NAMES, robot_model.robot_joint_names)

        q = robot.get_state().position
        send_cartesian_stage(
            robot,
            build_demo_vision_pre_push_command(
                dyn_model,
                dyn_state,
                robot_model,
                q,
                float(args.box_x_offset),
            ),
            "vision_pre_push_demo",
            int(args.priority),
            timeout_sec=stage_timeout_sec(
                VISION_APPROACH_MINIMUM_TIME,
                VISION_APPROACH_HOLD_TIME,
                min_timeout_sec=float(args.min_command_timeout_sec),
                margin_sec=float(args.command_timeout_margin_sec),
            ),
        )
        if args.hold_sec > 0:
            time.sleep(float(args.hold_sec))

        q = robot.get_state().position
        send_cartesian_stage(
            robot,
            build_impedance_push_command(dyn_model, dyn_state, q),
            "inward_push",
            int(args.priority),
            timeout_sec=stage_timeout_sec(
                PUSH_HOLD_TIME,
                min_timeout_sec=float(args.min_command_timeout_sec),
                margin_sec=float(args.command_timeout_margin_sec),
            ),
        )
        if args.hold_sec > 0:
            time.sleep(float(args.hold_sec))

        q = robot.get_state().position
        send_cartesian_stage(
            robot,
            build_impedance_lift_command(dyn_model, dyn_state, q),
            "lift",
            int(args.priority),
            timeout_sec=stage_timeout_sec(
                LIFT_MINIMUM_TIME,
                LIFT_HOLD_TIME,
                min_timeout_sec=float(args.min_command_timeout_sec),
                margin_sec=float(args.command_timeout_margin_sec),
            ),
        )
    print("[done] pose demo complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
