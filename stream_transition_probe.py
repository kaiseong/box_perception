#!/usr/bin/env python3
"""Probe rby1-sdk command-stream transitions WITHOUT moving the robot.

Answers the four unknowns behind the smooth mobile-align -> pre-push -> push
handoff before rebuilding it inside picking_box_5_debug.py:

  A. mobility-only SE(2) velocity streaming on one stream (known-good baseline)
  B. same stream, then ONE composite command: body Cartesian micro-move
     (+3cm z, then revert) + zero mobility, FK-VERIFIED. Acceptance alone was a
     false positive on hardware: the composite was accepted while the arms
     never moved, so this phase checks actual EEF motion.
  C. same stream, then body Impedance hold + zero mobility
     (can the body controller type switch Cartesian -> Impedance on a stream?)
  D. let control_hold_time expire with no further sends, then try to send again
     (does the stream die on hold expiry, and with which exception?)

Phase B RAISES AND LOWERS BOTH HANDS BY 3cm; everything else targets the
current pose with zero velocity. Watch the LED during each phase; the script
timestamps phase boundaries so you can correlate:
  - control active (command running) vs idle between phases
  - whether B/C keep control engaged without dropping to idle

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


def build_cartesian_hold_body(
    targets: dict[str, np.ndarray],
    *,
    hold_sec: float,
    minimum_time_sec: float = 1.0,
) -> Any:
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
            .set_minimum_time(float(minimum_time_sec))
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


ARM_PROBE_DZ_M = 0.03
ARM_PROBE_MIN_TIME_SEC = 1.5
ARM_PROBE_WAIT_SEC = 3.0
ARM_PROBE_MOVED_THRESHOLD_M = 0.02


def read_eef_z(robot: Any, dyn_model: Any, dyn_state: Any) -> tuple[float, float]:
    q = robot.get_state().position
    dyn_state.set_q(q)
    dyn_model.compute_forward_kinematics(dyn_state)
    right = dyn_model.compute_transformation(dyn_state, BASE_INDEX, EE_RIGHT_INDEX)
    left = dyn_model.compute_transformation(dyn_state, BASE_INDEX, EE_LEFT_INDEX)
    return float(np.asarray(right)[2, 3]), float(np.asarray(left)[2, 3])


def shifted_targets(targets: dict[str, np.ndarray], dz_m: float) -> dict[str, np.ndarray]:
    shifted = dict(targets)
    for key in ("base_to_right", "base_to_left"):
        lifted = np.asarray(targets[key], dtype=np.float64).copy()
        lifted[2, 3] += float(dz_m)
        shifted[key] = lifted
    return shifted


def phase_b_add_body(
    stream: Any,
    robot: Any,
    dyn_model: Any,
    dyn_state: Any,
    targets: dict[str, np.ndarray],
) -> bool:
    """Send an arm MICRO-MOVE (+3cm z) on the mobility stream and FK-verify it.

    Acceptance alone proved nothing on hardware: a streamed composite body
    command was accepted while the arms never moved 16cm to the pre-push pose.
    This phase measures actual EEF motion, then reverts.
    """
    z0_right, z0_left = read_eef_z(robot, dyn_model, dyn_state)
    log(f"phase B: streamed composite arm micro-move +{ARM_PROBE_DZ_M * 100:.0f}cm z "
        f"(min_time {ARM_PROBE_MIN_TIME_SEC}s) + zero mobility on the SAME stream; "
        "FK-verifying that the arms ACTUALLY track a streamed body command.")
    if not try_send(
        stream,
        build_composite(
            build_cartesian_hold_body(
                shifted_targets(targets, ARM_PROBE_DZ_M),
                hold_sec=BRIDGE_HOLD_SEC,
                minimum_time_sec=ARM_PROBE_MIN_TIME_SEC,
            ),
            hold_sec=BRIDGE_HOLD_SEC,
        ),
        label="B composite (arms +3cm z)",
    ):
        return False

    deadline = time.monotonic() + ARM_PROBE_WAIT_SEC
    best_dz = 0.0
    while time.monotonic() < deadline:
        time.sleep(0.2)
        z_right, z_left = read_eef_z(robot, dyn_model, dyn_state)
        best_dz = max(best_dz, min(z_right - z0_right, z_left - z0_left))
        log(f"B track: dz_right={(z_right - z0_right) * 100:+.1f}cm "
            f"dz_left={(z_left - z0_left) * 100:+.1f}cm")
        if best_dz >= ARM_PROBE_MOVED_THRESHOLD_M:
            break
    moved = best_dz >= ARM_PROBE_MOVED_THRESHOLD_M
    log(f"phase B verdict: streamed body command {'EXECUTES' if moved else 'DOES NOT EXECUTE'} "
        f"(best common dz={best_dz * 100:+.1f}cm)")

    log("phase B: reverting arms to the original pose on the same stream")
    reverted = try_send(
        stream,
        build_composite(
            build_cartesian_hold_body(
                targets,
                hold_sec=BRIDGE_HOLD_SEC,
                minimum_time_sec=ARM_PROBE_MIN_TIME_SEC,
            ),
            hold_sec=BRIDGE_HOLD_SEC,
        ),
        label="B revert",
    )
    time.sleep(ARM_PROBE_WAIT_SEC if moved else 0.5)
    # Liveness: did the composite kill the command stream?
    alive = try_send(
        stream,
        build_zero_se2_command(minimum_time=SEND_PERIOD_SEC, hold_sec=BRIDGE_HOLD_SEC),
        label="B liveness zero-SE2",
    )
    log(f"phase B: stream {'still alive' if alive else 'DIED'} after composite commands")
    return moved and reverted and alive


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
    log("current EEF poses captured; phase B lifts both hands 3cm and reverts, "
        "everything else holds the current pose")
    log_control_manager_state(robot, "before stream")

    stream = robot.create_command_stream(priority=STREAM_PRIORITY)
    log("command stream created")

    try:
        if not phase_a_mobility_stream(stream, args.phase_a_sec):
            return 1
        log_control_manager_state(robot, "after A")

        if "B" not in args.skip:
            if not phase_b_add_body(stream, robot, dyn_model, dyn_state, targets):
                log("phase B FAILED -> streamed composite body commands are accepted but "
                    "not executed; the smooth handoff cannot ride this stream for arm "
                    "motion and needs a different transport for pre-push")
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
