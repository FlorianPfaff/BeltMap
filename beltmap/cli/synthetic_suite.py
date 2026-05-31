from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


MAP_PARTICLE_MASK_MODES = ("positive", "negative", "absolute", "hysteresis_abs")
PHASE_ESTIMATION_MODES = ("motion_model", "registration", "smoothed_registration")

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
        "map_frame_median_offset_correction": True,
    },
    "local_illumination_drift": {
        "texture": 1.0,
        "particle_signal": 80.0,
        "noise": 2.0,
        "illumination": 0.0,
        "local_illumination": 24.0,
        "particles": 1,
        "velocity": 2.0,
        "photometric": True,
        "map_frame_median_offset_correction": True,
        "map_local_illumination_correction": True,
        "map_local_illumination_tile_px": 16,
    },
    "phase_jitter": {
        "texture": 1.0,
        "particle_signal": 80.0,
        "noise": 2.0,
        "illumination": 0.0,
        "particles": 1,
        "velocity": 2.0,
        "phase_jitter": 2.5,
        "phase_estimation_mode": "smoothed_registration",
        "phase_refinement_iterations": 1,
        "phase_refinement_smoothing_window_frames": 5,
        "phase_refinement_max_abs_correction_px": 4.0,
        "phase_smoothing_window_frames": 5,
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
    phase_corrections = [
        float(params.get("phase_jitter", 0.0))
        * math.sin(2 * math.pi * frame_index / max(frames, 1))
        for frame_index in range(frames)
    ]
    particle_vertical_period_px = height - 8
    for frame_index in range(frames):
        phase = (-velocity * frame_index + phase_corrections[frame_index]) % period
        rows = (np.arange(height, dtype=np.float64) + phase) % period
        row0 = np.floor(rows).astype(int)
        row1 = (row0 + 1) % period
        w1 = rows - row0
        frame = (1.0 - w1[:, None]) * belt_map[row0] + w1[:, None] * belt_map[row1]
        frame += float(params["illumination"]) * math.sin(2 * math.pi * frame_index / max(frames, 1))
        local_illumination = float(params.get("local_illumination", 0.0))
        if local_illumination:
            phase_fraction = 2 * math.pi * frame_index / max(frames, 1)
            yy = np.linspace(-1.0, 1.0, height, dtype=np.float64)[:, None]
            xx = np.linspace(-1.0, 1.0, width, dtype=np.float64)[None, :]
            field_pattern = 0.55 * xx + 0.35 * yy + 0.25 * xx * yy
            frame += local_illumination * math.sin(phase_fraction) * field_pattern
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
        "true_phase_px_by_frame": [
            float((-velocity * frame_index + phase_corrections[frame_index]) % period)
            for frame_index in range(frames)
        ],
        "true_nominal_phase_px_by_frame": [
            float((-velocity * frame_index) % period) for frame_index in range(frames)
        ],
        "true_phase_correction_px_by_frame": phase_corrections,
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
    detection_split_merged_components: bool = True,
    detection_split_min_projection_gap_px: int = 1,
    detection_split_min_component_area_px: int = 4,
    track_filter_min_length: int = 3,
    map_frame_median_offset_correction: bool = False,
    map_local_illumination_correction: bool = False,
    map_local_illumination_tile_px: int = 64,
    map_particle_mask_mode: str = "positive",
    map_particle_mask_grow_threshold: float = 2.0,
    map_particle_mask_margin_px: int = 1,
    phase_estimation_mode: str = "registration",
    phase_refinement_iterations: int = 0,
    phase_refinement_smoothing_window_frames: int = 25,
    phase_refinement_max_abs_correction_px: float = 0.0,
    phase_smoothing_window_frames: int = 0,
) -> Path:
    if map_particle_mask_mode not in MAP_PARTICLE_MASK_MODES:
        choices = ", ".join(MAP_PARTICLE_MASK_MODES)
        raise ValueError(f"map_particle_mask_mode must be one of {choices}")
    if map_particle_mask_grow_threshold < 0:
        raise ValueError("map_particle_mask_grow_threshold must be non-negative")
    if map_particle_mask_margin_px < 0:
        raise ValueError("map_particle_mask_margin_px must be non-negative")
    if map_local_illumination_tile_px < 1:
        raise ValueError("map_local_illumination_tile_px must be positive")
    if phase_estimation_mode not in PHASE_ESTIMATION_MODES:
        choices = ", ".join(PHASE_ESTIMATION_MODES)
        raise ValueError(f"phase_estimation_mode must be one of {choices}")
    if phase_refinement_iterations < 0:
        raise ValueError("phase_refinement_iterations must be non-negative")
    if phase_refinement_smoothing_window_frames < 0:
        raise ValueError("phase_refinement_smoothing_window_frames must be non-negative")
    if phase_refinement_max_abs_correction_px < 0:
        raise ValueError("phase_refinement_max_abs_correction_px must be non-negative")
    if phase_smoothing_window_frames < 0:
        raise ValueError("phase_smoothing_window_frames must be non-negative")
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
split_merged_components = {str(detection_split_merged_components).lower()}
split_min_projection_gap_px = {detection_split_min_projection_gap_px}
split_min_component_area_px = {detection_split_min_component_area_px}

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
frame_median_offset_correction = {str(map_frame_median_offset_correction).lower()}
local_illumination_correction = {str(map_local_illumination_correction).lower()}
local_illumination_tile_px = {map_local_illumination_tile_px}
mask_iterations = 1
particle_mask_threshold = 3.0
particle_mask_mode = {json.dumps(map_particle_mask_mode)}
particle_mask_grow_threshold = {map_particle_mask_grow_threshold}
particle_mask_margin_px = {map_particle_mask_margin_px}
particle_mask_min_area_px = 2

[registration]
search_radius_px = 8.0
search_step_px = 0.5

[phase]
estimation_mode = {json.dumps(phase_estimation_mode)}

[phase_refinement]
iterations = {phase_refinement_iterations}
max_abs_correction_px = {phase_refinement_max_abs_correction_px}
smoothing_window_frames = {phase_refinement_smoothing_window_frames}

[phase_smoothing]
window_frames = {phase_smoothing_window_frames}
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
    parser.add_argument("--map-particle-mask-mode", choices=MAP_PARTICLE_MASK_MODES, default="positive")
    parser.add_argument("--map-particle-mask-grow-threshold", type=float, default=2.0)
    parser.add_argument("--map-particle-mask-margin-px", type=int, default=1)
    parser.add_argument("--map-local-illumination-correction", action="store_true")
    parser.add_argument("--map-local-illumination-tile-px", type=int, default=64)
    parser.add_argument("--phase-estimation-mode", choices=PHASE_ESTIMATION_MODES)
    parser.add_argument("--phase-refinement-iterations", type=int)
    parser.add_argument("--phase-refinement-smoothing-window-frames", type=int)
    parser.add_argument("--phase-refinement-max-abs-correction-px", type=float)
    parser.add_argument("--phase-smoothing-window-frames", type=int)
    parser.add_argument("--execute", action="store_true", help="Run beltmap-apply and beltmap-benchmark after generating each case.")
    return parser


def case_or_arg(params: dict, args, key: str, default):
    value = getattr(args, key)
    if value is not None:
        return value
    return params.get(key, default)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cases = args.case or sorted(CASES)
    manifest = []
    for case in cases:
        params = CASES[case]
        root = args.output_root / case
        root.mkdir(parents=True, exist_ok=True)
        render_case(case, root, frames=args.frames, height=args.height, width=args.width, period=args.period, seed=args.seed)
        config_path = write_config(
            root,
            frames=args.frames,
            velocity=float(params["velocity"]),
            period=args.period,
            photometric_enabled=bool(params.get("photometric", False)),
            map_frame_median_offset_correction=bool(
                params.get("map_frame_median_offset_correction", False)
            ),
            map_local_illumination_correction=(
                args.map_local_illumination_correction
                or bool(params.get("map_local_illumination_correction", False))
            ),
            map_local_illumination_tile_px=int(
                params.get(
                    "map_local_illumination_tile_px",
                    args.map_local_illumination_tile_px,
                )
            ),
            map_particle_mask_mode=args.map_particle_mask_mode,
            map_particle_mask_grow_threshold=args.map_particle_mask_grow_threshold,
            map_particle_mask_margin_px=args.map_particle_mask_margin_px,
            phase_estimation_mode=str(
                case_or_arg(params, args, "phase_estimation_mode", "registration")
            ),
            phase_refinement_iterations=int(
                case_or_arg(params, args, "phase_refinement_iterations", 0)
            ),
            phase_refinement_smoothing_window_frames=int(
                case_or_arg(
                    params,
                    args,
                    "phase_refinement_smoothing_window_frames",
                    25,
                )
            ),
            phase_refinement_max_abs_correction_px=float(
                case_or_arg(
                    params,
                    args,
                    "phase_refinement_max_abs_correction_px",
                    0.0,
                )
            ),
            phase_smoothing_window_frames=int(
                case_or_arg(params, args, "phase_smoothing_window_frames", 0)
            ),
        )
        manifest.append(
            {
                "case": case,
                "root": str(root),
                "config": str(config_path),
                "truth": str(root / "synthetic_metadata.json"),
                "map_particle_mask_mode": args.map_particle_mask_mode,
                "map_particle_mask_grow_threshold": args.map_particle_mask_grow_threshold,
                "map_particle_mask_margin_px": args.map_particle_mask_margin_px,
                "map_local_illumination_correction": (
                    args.map_local_illumination_correction
                    or bool(params.get("map_local_illumination_correction", False))
                ),
                "map_local_illumination_tile_px": int(
                    params.get(
                        "map_local_illumination_tile_px",
                        args.map_local_illumination_tile_px,
                    )
                ),
                "phase_estimation_mode": str(
                    case_or_arg(params, args, "phase_estimation_mode", "registration")
                ),
                "phase_refinement_iterations": int(
                    case_or_arg(params, args, "phase_refinement_iterations", 0)
                ),
                "phase_refinement_smoothing_window_frames": int(
                    case_or_arg(
                        params,
                        args,
                        "phase_refinement_smoothing_window_frames",
                        25,
                    )
                ),
                "phase_refinement_max_abs_correction_px": float(
                    case_or_arg(
                        params,
                        args,
                        "phase_refinement_max_abs_correction_px",
                        0.0,
                    )
                ),
                "phase_smoothing_window_frames": int(
                    case_or_arg(params, args, "phase_smoothing_window_frames", 0)
                ),
            }
        )
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
