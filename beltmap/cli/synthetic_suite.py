from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


CASES = {
    "baseline": {"texture": 1.0, "particle_signal": 80.0, "noise": 2.0, "illumination": 0.0, "particles": 1, "velocity": 2.0},
    "weak_texture": {"texture": 0.25, "particle_signal": 80.0, "noise": 2.0, "illumination": 0.0, "particles": 1, "velocity": 2.0},
    "illumination_drift": {
        "texture": 1.0,
        "particle_signal": 80.0,
        "noise": 2.0,
        "illumination": 15.0,
        "particles": 1,
        "velocity": 2.0,
        "photometric": True,
    },
    "faint_particles": {"texture": 1.0, "particle_signal": 25.0, "noise": 3.0, "illumination": 0.0, "particles": 1, "velocity": 2.0},
    "high_density": {"texture": 1.0, "particle_signal": 70.0, "noise": 2.0, "illumination": 0.0, "particles": 5, "velocity": 2.0},
    "negative_velocity": {"texture": 1.0, "particle_signal": 80.0, "noise": 2.0, "illumination": 0.0, "particles": 1, "velocity": -2.0},
}


def make_belt_map(period: int, width: int, texture: float, rng: np.random.Generator) -> np.ndarray:
    y = np.linspace(0, 2 * np.pi, period, endpoint=False)[:, None]
    x = np.linspace(0, 2 * np.pi, width, endpoint=False)[None, :]
    base = 90 + texture * (18 * np.sin(3 * y + 0.7 * np.sin(x)) + 10 * np.cos(2 * x) + 8 * rng.normal(size=(period, width)))
    return np.clip(base, 0, 255).astype(np.float32)


def render_case(case: str, root: Path, *, frames: int, height: int, width: int, period: int, seed: int) -> None:
    params = CASES[case]
    rng = np.random.default_rng(seed)
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    belt_map = make_belt_map(period, width, float(params["texture"]), rng)
    np.save(root / "true_belt_map.npy", belt_map)
    boxes_by_frame: list[list[dict[str, float | str]]] = []
    velocity = float(params["velocity"])
    particle_motion_step_px = math.copysign(
        max(1.0, abs(velocity) * 0.7),
        velocity if velocity != 0 else 1.0,
    )
    particle_vertical_period_px = height - 8
    for frame_index in range(frames):
        phase = (-velocity * frame_index) % period
        rows = (np.arange(height, dtype=np.float64) + phase) % period
        row0 = np.floor(rows).astype(int)
        row1 = (row0 + 1) % period
        w1 = rows - row0
        frame = (1.0 - w1[:, None]) * belt_map[row0] + w1[:, None] * belt_map[row1]
        frame += float(params["illumination"]) * math.sin(2 * math.pi * frame_index / max(frames, 1))
        boxes: list[dict[str, float | str]] = []
        for particle in range(int(params["particles"])):
            vertical_position = 8 + particle * 17 + frame_index * particle_motion_step_px
            top = int(vertical_position % particle_vertical_period_px)
            left = int((12 + particle * 29) % (width - 8))
            bottom = top + 5
            right = left + 6
            frame[top:bottom, left:right] += float(params["particle_signal"])
            pass_index = int(vertical_position // particle_vertical_period_px)
            boxes.append(
                {
                    "top": top,
                    "left": left,
                    "bottom": bottom,
                    "right": right,
                    "event_id": f"{particle}:{pass_index}",
                }
            )
        frame += rng.normal(scale=float(params["noise"]), size=frame.shape)
        Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8)).save(image_dir / f"frame_{frame_index:04d}.png")
        boxes_by_frame.append(boxes)
    metadata = {
        "case": case,
        "height": height,
        "width": width,
        "belt_period_px": period,
        "true_belt_velocity_y_px_per_frame": velocity,
        "particle_shift_y_px_per_frame": particle_motion_step_px,
        "true_particle_velocity_y_px_per_frame": particle_motion_step_px,
        "true_velocity_ratio_y": None if velocity == 0 else particle_motion_step_px / velocity,
        "true_phase_px_by_frame": [float((-velocity * frame_index) % period) for frame_index in range(frames)],
        "true_belt_map_npy": "true_belt_map.npy",
        "frames": [{"frame_index": i, "boxes": boxes} for i, boxes in enumerate(boxes_by_frame)],
    }
    (root / "synthetic_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def write_config(
    root: Path,
    *,
    frames: int,
    velocity: float,
    period: int,
    photometric_enabled: bool = False,
    tracking_max_frame_gap: float = 2.0,
    tracking_velocity_fit_method: str = "theil_sen",
    detection_min_area_px: int = 4,
    track_filter_min_length: int = 3,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    config = f"""[paths]
image_dir = {json.dumps(str(root / "images"))}
output_dir = {json.dumps(str(root / "outputs"))}

[frames]
max_frames = {frames}
stride = 1

[belt]
velocity_px_per_frame = {velocity}
period_px = {period}

[detection]
threshold = 3.0
min_area_px = {detection_min_area_px}

[photometric]
enabled = {str(photometric_enabled).lower()}

[tracking]
min_track_length = 2
max_frame_gap = {tracking_max_frame_gap}
velocity_fit_method = {json.dumps(tracking_velocity_fit_method)}

[track_filter]
min_length = {track_filter_min_length}

[map]
sample_frames = {frames}
mask_iterations = 1
particle_mask_threshold = 3.0
particle_mask_margin_px = 2
particle_mask_min_area_px = 2

[registration]
search_radius_px = 8.0
search_step_px = 0.5
"""
    path = root / "beltmap.toml"
    path.write_text(config, encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="beltmap-synthetic-suite", description="Generate synthetic stress-test BeltMap cases.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/synthetic_suite"))
    parser.add_argument("--case", action="append", choices=sorted(CASES), help="Case to generate. Repeatable. Default: all cases.")
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--period", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--execute", action="store_true", help="Run beltmap-apply and beltmap-benchmark after generating each case.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cases = args.case or sorted(CASES)
    manifest = []
    for case in cases:
        root = args.output_root / case
        root.mkdir(parents=True, exist_ok=True)
        render_case(case, root, frames=args.frames, height=args.height, width=args.width, period=args.period, seed=args.seed)
        config_path = write_config(
            root,
            frames=args.frames,
            velocity=float(CASES[case]["velocity"]),
            period=args.period,
            photometric_enabled=bool(CASES[case].get("photometric", False)),
        )
        manifest.append({"case": case, "root": str(root), "config": str(config_path), "truth": str(root / "synthetic_metadata.json")})
        if args.execute:
            subprocess.run(
                [sys.executable, "-m", "beltmap.cli.apply", "--config", str(config_path)],
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "beltmap.cli.benchmark",
                    "--output-dir",
                    str(root / "outputs"),
                    "--truth-path",
                    str(root / "synthetic_metadata.json"),
                ],
                check=True,
            )
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "synthetic_suite_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
