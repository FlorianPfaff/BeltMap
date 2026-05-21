from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from beltmap.operational_improvements import list_image_paths, read_gray_image, suggest_belt_region_from_frames


def parse_positive_int(value: str) -> int:
    """Parse a strictly positive integer CLI argument."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_nonnegative_int(value: str) -> int:
    """Parse a non-negative integer CLI argument."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def parse_percentile(value: str) -> float:
    """Parse the motion-energy percentile used for ROI thresholding."""

    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("percentile must be a finite number in [0, 100)") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed < 100.0:
        raise argparse.ArgumentTypeError("percentile must be a finite number in [0, 100)")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Suggest a BeltMap belt region from motion energy.")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("belt_region_suggestion.json"))
    parser.add_argument("--max-frames", type=parse_positive_int, default=50)
    parser.add_argument("--percentile", type=parse_percentile, default=80.0)
    parser.add_argument("--margin-px", type=parse_nonnegative_int, default=16)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = list_image_paths(args.image_dir, max_frames=args.max_frames)
    if not paths:
        raise SystemExit(f"No image files found below {args.image_dir}")
    frames = [read_gray_image(path) for path in paths]
    suggestion = suggest_belt_region_from_frames(frames, percentile=args.percentile, margin_px=args.margin_px)
    payload = suggestion.to_dict()
    payload["region_csv"] = ",".join(str(value) for value in suggestion.region)
    payload["sampled_images"] = [str(path) for path in paths]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
