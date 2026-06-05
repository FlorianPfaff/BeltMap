from __future__ import annotations

import argparse
from pathlib import Path

from beltmap.postrun_improvements import write_label_plan, write_label_template


def parse_positive_int(value: str) -> int:
    """Parse a strictly positive integer CLI argument."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("frame count must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("frame count must be a positive integer")
    return parsed


def parse_non_negative_int(value: str) -> int:
    """Parse a non-negative integer CLI argument."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beltmap-suggest-label-frames",
        description="Suggest a diverse set of frames for sparse real-data annotation.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--frames", type=parse_positive_int, default=50)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--empty-frames",
        type=parse_non_negative_int,
        default=None,
        help=(
            "Minimum number of selected frames to reserve for particle-free or "
            "low-detection empty-frame checks. Default: about 20% of --frames."
        ),
    )
    parser.add_argument(
        "--min-gap-frames",
        type=parse_non_negative_int,
        default=0,
        help="Avoid selecting two different label frames closer than this many frame indices.",
    )
    parser.add_argument(
        "--template-output",
        type=Path,
        default=None,
        help=(
            "Optional CSV label template. Fill bbox columns for particles and "
            "leave blank rows only for frames inspected as empty."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output = args.output or args.output_dir / "label_plan.csv"
    rows = write_label_plan(
        args.output_dir,
        output_path=output,
        frame_count=args.frames,
        empty_frame_count=args.empty_frames,
        min_gap_frames=args.min_gap_frames,
    )
    if args.template_output is not None:
        write_label_template(rows, output_path=args.template_output)
        print(f"wrote label template to {args.template_output}", flush=True)
    print(f"wrote {len(rows)} frame suggestions to {output}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
