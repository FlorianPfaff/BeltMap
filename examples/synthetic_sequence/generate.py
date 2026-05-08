from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a small synthetic conveyor-belt image sequence for BeltMap.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/images"),
        help="Directory where generated PNG frames are written.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=12,
        help="Number of frames to generate.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=64,
        help="Generated frame height in pixels.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=48,
        help="Generated frame width in pixels.",
    )
    parser.add_argument(
        "--belt-shift-px-per-frame",
        type=int,
        default=2,
        help="Integer downward belt-texture shift in pixels per frame.",
    )
    parser.add_argument(
        "--particle-start-y",
        type=int,
        default=8,
        help="Initial top row of the bright synthetic particle.",
    )
    parser.add_argument(
        "--particle-x",
        type=int,
        default=22,
        help="Left column of the bright synthetic particle.",
    )
    parser.add_argument(
        "--particle-size-px",
        type=int,
        default=3,
        help="Square synthetic particle size in pixels.",
    )
    parser.add_argument(
        "--particle-shift-y-px-per-frame",
        type=int,
        default=1,
        help="Downward particle shift in pixels per frame.",
    )
    parser.add_argument(
        "--particle-signal",
        type=float,
        default=120.0,
        help="Brightness added to particle pixels.",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Do not remove existing frame_*.png files in the output directory first.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.frames < 1:
        raise ValueError("--frames must be at least 1")
    if args.height < 1 or args.width < 1:
        raise ValueError("--height and --width must be positive")
    if args.particle_size_px < 1:
        raise ValueError("--particle-size-px must be positive")
    if args.particle_size_px > args.height or args.particle_size_px > args.width:
        raise ValueError("--particle-size-px must fit within the generated frame")


def make_base_texture(height: int, width: int) -> np.ndarray:
    yy = np.arange(height, dtype=np.float64)[:, None]
    xx = np.arange(width, dtype=np.float64)[None, :]
    return (
        100.0
        + 25.0 * np.sin(2.0 * np.pi * yy / 16.0)
        + 10.0 * np.cos(2.0 * np.pi * xx / 12.0)
    )


def to_saved_gray(image: np.ndarray) -> np.ndarray:
    """Return the exact grayscale values written to generated PNG frames."""

    return np.clip(image, 0, 255).astype(np.uint8).astype(np.float32)


def particle_box(args: argparse.Namespace, frame_index: int) -> tuple[int, int, int, int]:
    top = min(
        args.height - args.particle_size_px,
        max(0, args.particle_start_y + args.particle_shift_y_px_per_frame * frame_index),
    )
    left = min(args.width - args.particle_size_px, max(0, args.particle_x))
    bottom = top + args.particle_size_px
    right = left + args.particle_size_px
    return top, left, bottom, right


def particle_centroid(box: tuple[int, int, int, int]) -> dict[str, float]:
    top, left, bottom, right = box
    return {
        "y": 0.5 * (top + bottom - 1),
        "x": 0.5 * (left + right - 1),
    }


def generate_sequence(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_clear:
        for pattern in ("frame_*.png", "true_clean_frame_*.png"):
            for path in args.output_dir.glob(pattern):
                path.unlink()

    base = to_saved_gray(make_base_texture(args.height, args.width))
    np.save(args.output_dir / "true_belt_map.npy", base)
    Image.fromarray(base.astype(np.uint8)).save(args.output_dir / "true_belt_map.png")

    particles: list[dict[str, Any]] = []
    true_phase_px_by_frame: list[float] = []
    true_particle_centroid_by_frame: list[dict[str, float | int]] = []

    for frame_index in range(args.frames):
        phase_px = float((-args.belt_shift_px_per_frame * frame_index) % args.height)
        true_phase_px_by_frame.append(phase_px)

        clean_frame = np.roll(
            base,
            shift=args.belt_shift_px_per_frame * frame_index,
            axis=0,
        )
        frame = clean_frame.copy()
        box = particle_box(args, frame_index)
        top, left, bottom, right = box
        frame[top:bottom, left:right] += args.particle_signal
        frame = to_saved_gray(frame)

        centroid = particle_centroid(box)
        true_particle_centroid_by_frame.append(
            {
                "frame_index": frame_index,
                "y": centroid["y"],
                "x": centroid["x"],
            }
        )
        particles.append(
            {
                "frame_index": frame_index,
                "top": top,
                "left": left,
                "bottom": bottom,
                "right": right,
                "centroid_y": centroid["y"],
                "centroid_x": centroid["x"],
            }
        )

        Image.fromarray(frame.astype(np.uint8)).save(
            args.output_dir / f"frame_{frame_index:03d}.png"
        )
        Image.fromarray(clean_frame.astype(np.uint8)).save(
            args.output_dir / f"true_clean_frame_{frame_index:03d}.png"
        )

    true_velocity_ratio = (
        args.particle_shift_y_px_per_frame / args.belt_shift_px_per_frame
        if args.belt_shift_px_per_frame != 0
        else None
    )
    metadata: dict[str, Any] = {
        "description": "Synthetic periodic conveyor-belt texture with one bright particle.",
        "frames": args.frames,
        "height": args.height,
        "width": args.width,
        "belt_shift_px_per_frame": args.belt_shift_px_per_frame,
        "belt_period_px": args.height,
        "true_belt_velocity_y_px_per_frame": args.belt_shift_px_per_frame,
        "true_phase_px_by_frame": true_phase_px_by_frame,
        "true_belt_map_npy": "true_belt_map.npy",
        "true_belt_map_png": "true_belt_map.png",
        "true_clean_frames_pattern": "true_clean_frame_{frame_index:03d}.png",
        "particle_shift_y_px_per_frame": args.particle_shift_y_px_per_frame,
        "true_particle_velocity_y_px_per_frame": args.particle_shift_y_px_per_frame,
        "true_velocity_ratio_y": true_velocity_ratio,
        "particle_size_px": args.particle_size_px,
        "particle_signal": args.particle_signal,
        "true_particle_centroid_by_frame": true_particle_centroid_by_frame,
        "particles": particles,
    }
    (args.output_dir / "synthetic_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        metadata = generate_sequence(args)
    except ValueError as exc:
        parser.error(str(exc))

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "frames": metadata["frames"],
                "image_shape": [metadata["height"], metadata["width"]],
                "belt_shift_px_per_frame": metadata["belt_shift_px_per_frame"],
                "belt_period_px": metadata["belt_period_px"],
                "true_belt_map": str(args.output_dir / "true_belt_map.npy"),
                "metadata": str(args.output_dir / "synthetic_metadata.json"),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
