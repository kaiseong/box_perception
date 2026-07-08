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


COMPARE_REFERENCE_COMMIT = "0198fa7"
DEFAULT_RESULT_LOG = ".omx/runtime/compare_pick_timing.jsonl"


@dataclass(frozen=True)
class Candidate:
    key: str
    label: str
    module: Any
    source: str


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
            "2 for picking_box_new. Unknown extra args are forwarded to both pickers."
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
        help="Stable usable frames required before 1/2 is accepted.",
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
        help="Show the D405 overlay while waiting for 1/2.",
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
) -> tuple[Candidate, np.ndarray] | None:
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
        "[compare] D405 warmup running. Press 1=legacy, 2=new, q=quit. "
        "A stable center is required before a pick starts.",
        flush=True,
    )
    try:
        with cbreak_stdin() as raw_keys:
            if not raw_keys:
                print("[compare] stdin is not a TTY; type 1/2/q then Enter.", flush=True)
            while True:
                status_lines = [
                    ("1: legacy  2: new  q: quit", (0, 220, 255)),
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
                            "[compare] vision ready; press 1=legacy or 2=new "
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
                candidate = CANDIDATES.get(key)
                if candidate is None:
                    print(f"[compare] unknown key {key!r}; use 1, 2, or q", flush=True)
                    continue
                if latest_stable is None:
                    print("[compare] stable warmed center is not ready yet; keep waiting", flush=True)
                    continue
                center = np.asarray(latest_stable["center_camera_m"], dtype=np.float64)
                return candidate, center
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
            candidate, center_camera_m = selected
            record = run_candidate(candidate, args, extra_candidate_args, center_camera_m)
            append_result(args.result_log, record)
            print(
                "[compare] Result saved. Reset the box/robot as needed, then press 1/2 again.",
                flush=True,
            )
    finally:
        if gripper is not None:
            print("[compare] stopping gripper max-open hold", flush=True)
            gripper.stop()


if __name__ == "__main__":
    raise SystemExit(main())
