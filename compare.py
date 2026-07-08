#!/usr/bin/env python3
"""Interactive A/B timing harness for legacy vs current picking pipelines.

The comparison timer starts when the operator presses 1 or 2 and stops when
the selected script returns after the lift FK engage gate. Gripper homing and
the initial D405 warmup/center capture are intentionally outside the measured
window.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import select
import sys
import termios
import time
import tty
from typing import Any

import numpy as np

import picking_box_new
import picking_box_legacy
import placing_and_picking


COMPARE_REFERENCE_COMMIT = "0198fa7"
DEFAULT_RESULT_LOG = ".omx/runtime/compare_pick_timing.jsonl"
PLACE_ONLY_KEY = "3"


@dataclass(frozen=True)
class Candidate:
    key: str
    label: str
    module: Any
    source: str


@dataclass(frozen=True)
class Selection:
    kind: str
    candidate: Candidate | None = None
    center_camera_m: np.ndarray | None = None


CANDIDATES = {
    "1": Candidate(
        key="1",
        label="legacy",
        module=picking_box_legacy,
        source=f"picking_box_5.py@{COMPARE_REFERENCE_COMMIT}",
    ),
    "2": Candidate(
        key="2",
        label="new",
        module=picking_box_new,
        source="picking_box_6.py@HEAD copy",
    ),
}


@contextmanager
def cbreak_stdin() -> Any:
    """Read single terminal keys without requiring Enter."""
    if not sys.stdin.isatty():
        yield False
        return
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield True
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def read_key_nonblocking(timeout_sec: float = 0.0) -> str | None:
    """Return one pressed key, or None when no key is available."""
    if not sys.stdin.isatty():
        return None
    ready, _, _ = select.select([sys.stdin], [], [], float(timeout_sec))
    if not ready:
        return None
    return sys.stdin.read(1)


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Warm gripper/D405 once, then press 1 for picking_box_legacy or "
            "2 for picking_box_new, or 3 for place-only. Unknown extra args are "
            "forwarded to both pickers."
        )
    )
    parser.add_argument("--address", type=str, required=True, help="Robot address")
    parser.add_argument("--model", type=str, default="m", help="Robot model name")
    parser.add_argument("--power", type=str, default=".*", help="Power device name regex")
    parser.add_argument(
        "--view-rotation",
        choices=("none", "cw90", "ccw90", "180"),
        default="cw90",
        help="D405 analysis-frame rotation. Must match both pickers.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=picking_box_new.LIVE_VISION_FRAMES_NEEDED,
        help="Stable usable frames required before 1/2 is accepted. 3 does not require vision.",
    )
    parser.add_argument(
        "--warmup-center-spread-m",
        type=float,
        default=picking_box_new.LIVE_VISION_MAX_CENTER_SPREAD_M,
        help="Maximum center spread for warmed measurement acceptance.",
    )
    parser.add_argument(
        "--visualize-warmup",
        action="store_true",
        help="Show the D405 overlay while waiting. The camera warms up even without this.",
    )
    parser.add_argument(
        "--no-gripper-home",
        action="store_true",
        help="Skip the one-time gripper homing/open hold step.",
    )
    parser.add_argument(
        "--result-log",
        type=str,
        default=DEFAULT_RESULT_LOG,
        help="JSONL path for measured A/B timing results. Empty disables logging.",
    )
    parser.add_argument(
        "--lift-target-json",
        type=str,
        default=placing_and_picking.DEFAULT_LIFT_TARGET_JSON,
        help="Lift target JSON used by key 3 place-only.",
    )
    parser.add_argument(
        "--place-lower-delta-m",
        type=float,
        default=placing_and_picking.PLACE_LOWER_DELTA_M,
        help="Key 3: base-frame z distance to lower from the exported lift target.",
    )
    parser.add_argument(
        "--place-release-distance-m",
        type=float,
        default=placing_and_picking.PUSH_DISTANCE,
        help="Key 3: outward y distance per hand for release.",
    )
    parser.add_argument(
        "--lower-ramp-time-sec",
        type=float,
        default=placing_and_picking.LOWER_RAMP_TIME_SEC,
        help="Key 3: lower ramp duration.",
    )
    parser.add_argument(
        "--release-ramp-time-sec",
        type=float,
        default=placing_and_picking.RELEASE_RAMP_TIME_SEC,
        help="Key 3: release/open ramp duration.",
    )
    parser.add_argument(
        "--eef-wait-timeout-sec",
        type=float,
        default=placing_and_picking.EEF_WAIT_TIMEOUT_SEC,
        help="Key 3: FK/gap verification timeout per stage.",
    )
    parser.add_argument(
        "--place-ready-time-sec",
        type=float,
        default=picking_box_new.MINIMUM_TIME,
        help="Key 3: joint-position READY return duration after place-only release.",
    )
    args, unknown = parser.parse_known_args(argv)
    return args, unknown


def prepare_gripper(args: argparse.Namespace) -> Any | None:
    """Home/open the gripper once and keep the max-open loop alive."""
    if args.no_gripper_home:
        print("[compare] gripper home/open skipped by --no-gripper-home", flush=True)
        return None

    print("[compare] connecting once for gripper homing/open hold", flush=True)
    robot = picking_box_new.rby.create_robot(args.address, args.model)
    robot.connect()
    if not robot.is_connected():
        raise RuntimeError("Robot is not connected")
    if not robot.is_power_on(args.power):
        if not robot.power_on(args.power):
            raise RuntimeError("Failed to power on before gripper setup")
    gripper = picking_box_new.setup_max_open_gripper(robot)
    if gripper is None:
        raise RuntimeError("Failed to home/open gripper")
    print("[compare] gripper is homed and held max-open; timing excludes this step", flush=True)
    return gripper


def _stable_live_result_from_candidates(
    module: Any,
    centers: list[Any],
    long_axes: list[Any | None],
    modes: list[str],
    unconstrained_flags: list[bool],
    *,
    frames_needed: int,
    max_center_spread_m: float,
) -> dict[str, Any] | None:
    stable, _ = module.select_stable_live_center_result(
        centers,
        long_axes,
        modes,
        unconstrained_flags,
        frames_needed=frames_needed,
        max_center_spread_m=max_center_spread_m,
    )
    return stable


def wait_for_candidate_with_warm_vision(
    args: argparse.Namespace,
) -> Selection | None:
    """Keep D405 inference alive until the operator selects a candidate."""
    module = picking_box_new
    live_view = module.ContinuousLiveBoxView(args.view_rotation, show=args.visualize_warmup)
    centers: list[Any] = []
    long_axes: list[Any | None] = []
    modes: list[str] = []
    unconstrained_flags: list[bool] = []
    latest_stable: dict[str, Any] | None = None
    last_status_sec = 0.0
    camera_to_display = np.eye(4, dtype=np.float64)

    print(
        "[compare] D405 warmup running. Press 1=legacy pick, 2=new pick, "
        "3=place-only, q=quit. A stable center is required before a pick starts.",
        flush=True,
    )
    try:
        with cbreak_stdin() as raw_keys:
            if not raw_keys:
                print("[compare] stdin is not a TTY; type 1/2/3/q then Enter.", flush=True)
            while True:
                status_lines = [
                    ("1: legacy  2: new  3: place-only  q: quit", (0, 220, 255)),
                ]
                if latest_stable is None:
                    status_lines.append(("warming: waiting for stable center", (40, 40, 220)))
                else:
                    center = np.asarray(latest_stable["center_camera_m"], dtype=np.float64)
                    status_lines.append(
                        (
                            f"READY center camera=[{center[0]:+.3f}, {center[1]:+.3f}, {center[2]:+.3f}]",
                            (30, 180, 30),
                        )
                    )

                try:
                    usable, mode, estimate, _, abort_requested = live_view.process_next_frame(
                        camera_to_base=camera_to_display,
                        status_lines=status_lines,
                    )
                except StopIteration:
                    print("[compare] D405 frame stream ended", flush=True)
                    return None
                if abort_requested:
                    return None

                if usable and estimate.center_top_camera_m is not None:
                    centers.append(estimate.center_top_camera_m)
                    modes.append(mode)
                    long_axes.append(estimate.support.get("long_axis_camera"))
                    unconstrained_flags.append(
                        (
                            not estimate.confidence.ok
                            and "long_axis_center_underconstrained" in estimate.failure_reasons
                        )
                    )
                    # Keep memory bounded while retaining enough candidates for median clustering.
                    max_keep = max(20, int(args.warmup_frames) * 4)
                    centers[:] = centers[-max_keep:]
                    long_axes[:] = long_axes[-max_keep:]
                    modes[:] = modes[-max_keep:]
                    unconstrained_flags[:] = unconstrained_flags[-max_keep:]
                    latest_stable = _stable_live_result_from_candidates(
                        module,
                        centers,
                        long_axes,
                        modes,
                        unconstrained_flags,
                        frames_needed=int(args.warmup_frames),
                        max_center_spread_m=float(args.warmup_center_spread_m),
                    )

                now = time.monotonic()
                if now - last_status_sec >= 1.0:
                    if latest_stable is None:
                        print(
                            f"[compare] warming vision candidates={len(centers)} "
                            f"need={int(args.warmup_frames)}",
                            flush=True,
                        )
                    else:
                        print(
                            "[compare] vision ready; press 1=legacy, 2=new, or 3=place-only "
                            f"(spread={latest_stable['center_spread_m'] * 1000.0:.1f} mm)",
                            flush=True,
                        )
                    last_status_sec = now

                if raw_keys:
                    key = read_key_nonblocking(0.0)
                else:
                    line = input().strip()
                    key = line[:1] if line else None
                if key is None:
                    continue
                if key in ("q", "Q", "\x1b"):
                    return None
                if key == PLACE_ONLY_KEY:
                    return Selection(kind="place_only")
                candidate = CANDIDATES.get(key)
                if candidate is None:
                    print(f"[compare] unknown key {key!r}; use 1, 2, 3, or q", flush=True)
                    continue
                if latest_stable is None:
                    print("[compare] stable warmed center is not ready yet; keep waiting", flush=True)
                    continue
                center = np.asarray(latest_stable["center_camera_m"], dtype=np.float64)
                return Selection(kind="pick", candidate=candidate, center_camera_m=center)
    finally:
        live_view.close()


def build_candidate_argv(
    args: argparse.Namespace,
    extra_candidate_args: list[str],
    center_camera_m: np.ndarray,
) -> list[str]:
    """Build CLI args that skip gripper and initial live capture for fair timing."""
    center = np.asarray(center_camera_m, dtype=np.float64).reshape(3)
    return [
        *extra_candidate_args,
        "--address",
        str(args.address),
        "--model",
        str(args.model),
        "--power",
        str(args.power),
        "--view-rotation",
        str(args.view_rotation),
        "--no-gripper-open",
        "--box-center-camera",
        f"{center[0]:.9f}",
        f"{center[1]:.9f}",
        f"{center[2]:.9f}",
    ]


def append_result(path: str | Path | None, record: dict[str, Any]) -> None:
    if path is None or str(path) == "":
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def run_candidate(
    candidate: Candidate,
    args: argparse.Namespace,
    extra_candidate_args: list[str],
    center_camera_m: np.ndarray,
) -> dict[str, Any]:
    argv = build_candidate_argv(args, extra_candidate_args, center_camera_m)
    print(
        f"[compare] START key={candidate.key} label={candidate.label} source={candidate.source}",
        flush=True,
    )
    print("[compare] timer starts now; gripper homing and warm vision are excluded", flush=True)
    start = time.monotonic()
    exit_code = 1
    error: str | None = None
    try:
        exit_code = int(candidate.module.run_cli(argv))
    except SystemExit as exc:
        exit_code = int(exc.code or 0) if isinstance(exc.code, int) else 1
        if exit_code != 0:
            error = str(exc)
    except Exception as exc:  # pragma: no cover - hardware/runtime dependent
        exit_code = 1
        error = f"{type(exc).__name__}: {exc}"
        print(f"[compare] ERROR: {error}", flush=True)
    elapsed = time.monotonic() - start
    ok = exit_code == 0
    record = {
        "candidate": candidate.label,
        "key": candidate.key,
        "source": candidate.source,
        "ok": ok,
        "exit_code": exit_code,
        "elapsed_sec": elapsed,
        "center_camera_m": np.asarray(center_camera_m, dtype=np.float64).tolist(),
        "unix_time_sec": time.time(),
    }
    if error is not None:
        record["error"] = error
    print(
        f"[compare] DONE key={candidate.key} label={candidate.label} "
        f"ok={ok} elapsed={elapsed:.3f}s",
        flush=True,
    )
    return record


def connect_place_robot(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    """Connect and prepare the robot for target-space place-only motion."""
    if placing_and_picking.rby is None:
        raise RuntimeError("rby1_sdk is required for place-only mode")

    robot = placing_and_picking.rby.create_robot(args.address, args.model)
    robot.connect()
    if not robot.is_connected():
        raise RuntimeError("Robot is not connected")
    if not robot.is_power_on(args.power):
        if not robot.power_on(args.power):
            raise RuntimeError("Failed to power on")
    if not robot.is_servo_on(".*"):
        if not robot.servo_on(".*"):
            raise RuntimeError("Failed to servo on")
    robot.reset_fault_control_manager()
    if not robot.enable_control_manager():
        raise RuntimeError("Failed to enable control manager")

    robot_model = robot.model()
    dyn_model = robot.get_dynamics()
    dyn_state = dyn_model.make_state(
        placing_and_picking.DYN_LINK_NAMES,
        robot_model.robot_joint_names,
    )
    return robot, dyn_model, dyn_state


def perform_place_only_sequence(
    place_module: Any,
    robot: Any,
    dyn_model: Any,
    dyn_state: Any,
    lifted: Any,
    *,
    place_lower_delta_m: float,
    place_release_distance_m: float,
    lower_ramp_time_sec: float,
    release_ramp_time_sec: float,
    eef_wait_timeout_sec: float,
    ready_time_sec: float,
) -> bool:
    """Lower from the exported lift target, open the arms, then return to READY."""
    targets = place_module.build_place_regrasp_target_chain(
        lifted,
        lower_delta_m=float(place_lower_delta_m),
        push_distance_m=float(place_release_distance_m),
    )

    stage = "place_only 1/2 place_lower"
    place_module.print_stage(
        "compare_place_only",
        "cancel_control before first target stream",
    )
    if not place_module.cancel_control_for_next_stream(robot, "compare_place_only"):
        return False
    place_module.print_stage(stage, "building target ramp")
    if not place_module.stream_target_ramp_stage(
        robot,
        start=targets.lifted,
        end=targets.lowered,
        stage=stage,
        ramp_time_sec=float(lower_ramp_time_sec),
    ):
        return False
    if not place_module.wait_for_eef_targets(
        robot,
        dyn_model,
        dyn_state,
        targets.lowered,
        stage=stage,
        timeout_sec=float(eef_wait_timeout_sec),
    ):
        return False

    place_module.print_stage(stage, "cancel_control for release stream")
    if not place_module.cancel_control_for_next_stream(robot, stage):
        return False

    stage = "place_only 2/2 release_open"
    initial_gap = place_module.hand_gap_m(
        place_module.current_eef_pair(robot, dyn_model, dyn_state)
    )
    place_module.print_stage(stage, "building target ramp")
    if not place_module.stream_target_ramp_stage(
        robot,
        start=targets.lowered,
        end=targets.released,
        stage=stage,
        ramp_time_sec=float(release_ramp_time_sec),
    ):
        return False
    if not place_module.wait_for_gap_motion(
        robot,
        dyn_model,
        dyn_state,
        initial_gap_m=initial_gap,
        target_gap_m=place_module.hand_gap_m(targets.released),
        stage=stage,
        timeout_sec=float(eef_wait_timeout_sec),
    ):
        return False

    place_module.print_stage(stage, "cancel_control for READY return")
    if not place_module.cancel_control_for_next_stream(robot, stage):
        return False

    return send_ready_after_place(robot, ready_time_sec=float(ready_time_sec))


def send_ready_after_place(robot: Any, *, ready_time_sec: float) -> bool:
    stage = "place_only 3/3 ready"
    picking_box_new.print_stage(stage, "joint position move")
    ok = picking_box_new.send_stage(
        robot,
        picking_box_new.build_pose_command(
            picking_box_new.READY,
            float(ready_time_sec),
        ),
        stage,
        timeout_sec=picking_box_new.stage_timeout_sec(
            float(ready_time_sec),
            min_timeout_sec=picking_box_new.COMMAND_TIMEOUT_MIN_SEC,
            margin_sec=picking_box_new.COMMAND_TIMEOUT_MARGIN_SEC,
        ),
    )
    if ok:
        picking_box_new.print_stage(stage, "reached")
    else:
        picking_box_new.print_stage(stage, "FAILED")
    return ok


def run_place_only(args: argparse.Namespace) -> dict[str, Any]:
    print(
        f"[compare] START key={PLACE_ONLY_KEY} label=place_only "
        "source=placing_and_picking.py target lower/release/ready",
        flush=True,
    )
    print("[compare] timer starts now; warm vision is excluded", flush=True)
    start = time.monotonic()
    ok = False
    error: str | None = None
    try:
        lifted = placing_and_picking.load_lift_target_record(args.lift_target_json)
        robot, dyn_model, dyn_state = connect_place_robot(args)
        ok = perform_place_only_sequence(
            placing_and_picking,
            robot,
            dyn_model,
            dyn_state,
            lifted,
            place_lower_delta_m=float(args.place_lower_delta_m),
            place_release_distance_m=float(args.place_release_distance_m),
            lower_ramp_time_sec=float(args.lower_ramp_time_sec),
            release_ramp_time_sec=float(args.release_ramp_time_sec),
            eef_wait_timeout_sec=float(args.eef_wait_timeout_sec),
            ready_time_sec=float(args.place_ready_time_sec),
        )
    except Exception as exc:  # pragma: no cover - hardware/runtime dependent
        ok = False
        error = f"{type(exc).__name__}: {exc}"
        print(f"[compare] ERROR: {error}", flush=True)

    elapsed = time.monotonic() - start
    record = {
        "candidate": "place_only",
        "key": PLACE_ONLY_KEY,
        "source": "placing_and_picking.py target lower/release/ready",
        "ok": ok,
        "exit_code": 0 if ok else 1,
        "elapsed_sec": elapsed,
        "lift_target_json": str(args.lift_target_json),
        "place_lower_delta_m": float(args.place_lower_delta_m),
        "place_release_distance_m": float(args.place_release_distance_m),
        "place_ready_time_sec": float(args.place_ready_time_sec),
        "unix_time_sec": time.time(),
    }
    if error is not None:
        record["error"] = error
    print(
        f"[compare] DONE key={PLACE_ONLY_KEY} label=place_only "
        f"ok={ok} elapsed={elapsed:.3f}s",
        flush=True,
    )
    return record


def main(argv: list[str] | None = None) -> int:
    args, extra_candidate_args = parse_args(argv)
    gripper = None
    try:
        gripper = prepare_gripper(args)
        while True:
            selected = wait_for_candidate_with_warm_vision(args)
            if selected is None:
                print("[compare] exiting", flush=True)
                return 0
            if selected.kind == "place_only":
                record = run_place_only(args)
            else:
                if selected.candidate is None or selected.center_camera_m is None:
                    raise RuntimeError(f"Invalid compare selection: {selected}")
                record = run_candidate(
                    selected.candidate,
                    args,
                    extra_candidate_args,
                    selected.center_camera_m,
                )
            append_result(args.result_log, record)
            print(
                "[compare] Result saved. Reset the box/robot as needed, then press 1/2/3 again.",
                flush=True,
            )
    finally:
        if gripper is not None:
            print("[compare] stopping gripper max-open hold", flush=True)
            gripper.stop()


if __name__ == "__main__":
    raise SystemExit(main())
