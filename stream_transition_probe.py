#!/usr/bin/env python3
"""Probe rby1-sdk command-stream transitions WITHOUT moving the robot.

Answers the four unknowns behind the smooth mobile-align -> pre-push -> push
handoff before rebuilding it inside picking_box_5_debug.py:

  A. mobility-only SE(2) velocity streaming on one stream (known-good baseline)
  B. same stream, then ONE composite command: body Cartesian hold + zero mobility
     (can a stream gain a body component mid-life without a control drop?)
  C. same stream, then body Impedance hold + zero mobility
     (can the body controller type switch Cartesian -> Impedance on a stream?)
  D. let control_hold_time expire with no further sends, then try to send again
     (does the stream die on hold expiry, and with which exception?)

Every command in this script targets the CURRENT pose with zero velocity, so a
healthy robot should not visibly move at any point. Watch the LED during each
phase; the script timestamps phase boundaries so you can correlate:
  - control active (command running) vs idle between phases
  - whether B/C keep control engaged with no blue->green->blue cycle

Run:  python stream_transition_probe.py --address <ROBOT_IP:PORT>
Skip phases with e.g. --skip D. Abort any time with Ctrl+C (cancels control).
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np
import rby1_sdk as rby

DYN_LINK_NAMES = ["base", "link_torso_5", "ee_right", "ee_left"]
BASE_INDEX, TORSO_INDEX, EE_RIGHT_INDEX, EE_LEFT_INDEX = 0, 1, 2, 3

STREAM_PRIORITY = 1
SEND_PERIOD_SEC = 1.0 / 30.0
SERVO_HOLD_SEC = 0.25
BRIDGE_HOLD_SEC = 3.0
EXPIRY_HOLD_SEC = 0.3

CARTESIAN_LINEAR_VELOCITY_LIMIT = 0.4
CARTESIAN_ANGULAR_VELOCITY_LIMIT = float(np.pi / 4)
CARTESIAN_ACCELERATION_LIMIT_SCALING = 0.5
IMPEDANCE_TRANSLATION_WEIGHT = [500.0, 1000.0, 500.0]
IMPEDANCE_ROTATION_WEIGHT = [50.0, 50.0, 50.0]

_START = time.monotonic()


def log(message: str) -> None:
    print(f"[probe +{time.monotonic() - _START:7.3f}s] {message}", flush=True)


def log_control_manager_state(robot: Any, label: str) -> None:
    """Best-effort dump of the control manager state around a transition."""
    try:
        state = robot.get_control_manager_state()
    except Exception as exc:  # pragma: no cover - SDK/hardware dependent
        log(f"{label}: get_control_manager_state failed: {exc}")
        return
    fields = {}
    for name in ("state", "control_state", "time_scale", "enabled_joint_idx"):
        value = getattr(state, name, None)
        if value is not None:
            fields[name] = value
    log(f"{label}: control_manager {fields}")


def build_zero_se2_command(*, minimum_time: float, hold_sec: float) -> Any:
    return rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_mobility_command(
            rby.SE2VelocityCommandBuilder()
            .set_command_header(
                rby.CommandHeaderBuilder().set_control_hold_time(float(hold_sec))
            )
            .set_minimum_time(float(minimum_time))
            .set_velocity(np.zeros(2, dtype=np.float64), 0.0)
        )
    )


def build_zero_se2_mobility(*, minimum_time: float, hold_sec: float) -> Any:
    return (
        rby.SE2VelocityCommandBuilder()
        .set_command_header(
            rby.CommandHeaderBuilder().set_control_hold_time(float(hold_sec))
        )
        .set_minimum_time(float(minimum_time))
        .set_velocity(np.zeros(2, dtype=np.float64), 0.0)
    )


def current_eef_targets(robot: Any, dyn_model: Any, dyn_state: Any) -> dict[str, np.ndarray]:
    q = robot.get_state().position
    dyn_state.set_q(q)
    dyn_model.compute_forward_kinematics(dyn_state)
    return {
        "base_to_right": np.asarray(
            dyn_model.compute_transformation(dyn_state, BASE_INDEX, EE_RIGHT_INDEX),
            dtype=np.float64,
        ),
        "base_to_left": np.asarray(
            dyn_model.compute_transformation(dyn_state, BASE_INDEX, EE_LEFT_INDEX),
            dtype=np.float64,
        ),
        "torso_to_right": np.asarray(
            dyn_model.compute_transformation(dyn_state, TORSO_INDEX, EE_RIGHT_INDEX),
            dtype=np.float64,
        ),
        "torso_to_left": np.asarray(
            dyn_model.compute_transformation(dyn_state, TORSO_INDEX, EE_LEFT_INDEX),
            dtype=np.float64,
        ),
    }


def build_cartesian_hold_body(targets: dict[str, np.ndarray], *, hold_sec: float) -> Any:
    def arm(link_name: str, target: np.ndarray) -> Any:
        return (
            rby.CartesianCommandBuilder()
            .set_command_header(
                rby.CommandHeaderBuilder().set_control_hold_time(float(hold_sec))
            )
            .add_target(
                "base",
                link_name,
                target,
                CARTESIAN_LINEAR_VELOCITY_LIMIT,
                CARTESIAN_ANGULAR_VELOCITY_LIMIT,
                CARTESIAN_ACCELERATION_LIMIT_SCALING,
            )
            .set_minimum_time(1.0)
        )

    return (
        rby.BodyComponentBasedCommandBuilder()
        .set_right_arm_command(arm("ee_right", targets["base_to_right"]))
        .set_left_arm_command(arm("ee_left", targets["base_to_left"]))
    )


def build_impedance_hold_body(targets: dict[str, np.ndarray], *, hold_sec: float) -> Any:
    def arm(link_name: str, target: np.ndarray) -> Any:
        return (
            rby.ImpedanceControlCommandBuilder()
            .set_command_header(
                rby.CommandHeaderBuilder().set_control_hold_time(float(hold_sec))
            )
            .set_reference_link_name("link_torso_5")
            .set_link_name(link_name)
            .set_translation_weight(IMPEDANCE_TRANSLATION_WEIGHT)
            .set_rotation_weight(IMPEDANCE_ROTATION_WEIGHT)
            .set_transformation(target)
        )

    return (
        rby.BodyComponentBasedCommandBuilder()
        .set_right_arm_command(arm("ee_right", targets["torso_to_right"]))
        .set_left_arm_command(arm("ee_left", targets["torso_to_left"]))
    )


def build_composite(body: Any, *, hold_sec: float) -> Any:
    return rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder()
        .set_body_command(body)
        .set_mobility_command(
            build_zero_se2_mobility(minimum_time=SEND_PERIOD_SEC, hold_sec=hold_sec)
        )
    )


def try_send(stream: Any, command: Any, label: str) -> bool:
    try:
        stream.send_command(command)
    except Exception as exc:
        log(f"{label}: send_command RAISED {type(exc).__name__}: {exc}")
        return False
    log(f"{label}: send_command accepted")
    return True


def phase_a_mobility_stream(stream: Any, duration_sec: float) -> bool:
    log(f"phase A: streaming zero SE(2) velocity at 30Hz for {duration_sec:.1f}s "
        f"(hold {SERVO_HOLD_SEC}s each). LED should show control ACTIVE now.")
    deadline = time.monotonic() + duration_sec
    sends = 0
    while time.monotonic() < deadline:
        if not try_send(
            stream,
            build_zero_se2_command(minimum_time=SEND_PERIOD_SEC * 1.01, hold_sec=SERVO_HOLD_SEC),
            label=f"A send#{sends}",
        ):
            return False
        sends += 1
        time.sleep(SEND_PERIOD_SEC)
    log(f"phase A: done ({sends} sends). Sending one bridge command (hold {BRIDGE_HOLD_SEC}s).")
    return try_send(
        stream,
        build_zero_se2_command(minimum_time=SEND_PERIOD_SEC, hold_sec=BRIDGE_HOLD_SEC),
        label="A bridge",
    )


def phase_b_add_body(stream: Any, targets: dict[str, np.ndarray]) -> bool:
    log("phase B: ONE composite command on the SAME stream: "
        "body Cartesian hold (current pose) + zero mobility. Watch the LED: "
        "any green flash here means the stream cannot gain a body component seamlessly.")
    ok = try_send(
        stream,
        build_composite(
            build_cartesian_hold_body(targets, hold_sec=BRIDGE_HOLD_SEC),
            hold_sec=BRIDGE_HOLD_SEC,
        ),
        label="B composite",
    )
    time.sleep(2.0)
    return ok


def phase_c_switch_to_impedance(stream: Any, targets: dict[str, np.ndarray]) -> bool:
    log("phase C: body Impedance hold (current pose) + zero mobility on the SAME stream. "
        "Tests Cartesian -> Impedance controller switch mid-stream.")
    ok = try_send(
        stream,
        build_composite(
            build_impedance_hold_body(targets, hold_sec=BRIDGE_HOLD_SEC),
            hold_sec=BRIDGE_HOLD_SEC,
        ),
        label="C composite",
    )
    time.sleep(2.0)
    return ok


def phase_d_hold_expiry(stream: Any) -> None:
    log(f"phase D: sending zero SE(2) with SHORT hold ({EXPIRY_HOLD_SEC}s), then sleeping "
        "1.5s so it expires. Watch when the LED drops to idle.")
    if not try_send(
        stream,
        build_zero_se2_command(minimum_time=SEND_PERIOD_SEC, hold_sec=EXPIRY_HOLD_SEC),
        label="D short-hold",
    ):
        return
    time.sleep(1.5)
    log("phase D: attempting a send AFTER hold expiry (expected to fail if the "
        "stream dies with its command).")
    try_send(
        stream,
        build_zero_se2_command(minimum_time=SEND_PERIOD_SEC, hold_sec=SERVO_HOLD_SEC),
        label="D post-expiry",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--address", type=str, required=True, help="Robot address")
    parser.add_argument("--model", type=str, default="m", help="Robot Model Name")
    parser.add_argument("--power", type=str, default=".*", help="Power device name regex")
    parser.add_argument("--phase-a-sec", type=float, default=2.0)
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        choices=["B", "C", "D"],
        help="Skip a phase (repeatable)",
    )
    args = parser.parse_args()

    robot = rby.create_robot(args.address, args.model)
    robot.connect()
    if not robot.is_connected():
        print("Robot is not connected")
        return 1
    if not robot.is_power_on(args.power):
        if not robot.power_on(args.power):
            print("Failed to power on")
            return 1
    if not robot.is_servo_on(".*"):
        if not robot.servo_on(".*"):
            print("Failed to servo on")
            return 1
    robot.reset_fault_control_manager()
    if not robot.enable_control_manager():
        print("Failed to enable control manager")
        return 1

    dyn_model = robot.get_dynamics()
    dyn_state = dyn_model.make_state(DYN_LINK_NAMES, robot.model().robot_joint_names)
    targets = current_eef_targets(robot, dyn_model, dyn_state)
    log("current EEF poses captured; every command below targets these (no motion expected)")
    log_control_manager_state(robot, "before stream")

    stream = robot.create_command_stream(priority=STREAM_PRIORITY)
    log("command stream created")

    try:
        if not phase_a_mobility_stream(stream, args.phase_a_sec):
            return 1
        log_control_manager_state(robot, "after A")

        if "B" not in args.skip:
            if not phase_b_add_body(stream, targets):
                log("phase B FAILED -> the rebuild must stream composite commands from "
                    "the start of servoing (body hold + mobility velocity together)")
                return 1
            log_control_manager_state(robot, "after B")

        if "C" not in args.skip:
            if not phase_c_switch_to_impedance(stream, targets):
                log("phase C FAILED -> pre-push -> push cannot switch controller type on "
                    "one stream; push/lift need their own bridged transition")
                return 1
            log_control_manager_state(robot, "after C")

        if "D" not in args.skip:
            phase_d_hold_expiry(stream)
            log_control_manager_state(robot, "after D")

        return 0
    except KeyboardInterrupt:
        log("interrupted by operator")
        return 130
    finally:
        try:
            robot.cancel_control()
            log("robot.cancel_control() requested (cleanup)")
        except Exception as exc:  # pragma: no cover - SDK/hardware dependent
            log(f"cleanup cancel_control failed: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
