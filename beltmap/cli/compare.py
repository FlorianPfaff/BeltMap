from __future__ import annotations

import argparse
import json
from pathlib import Path

from beltmap.compare_runs import generate_comparison_report, parse_run_spec


def parse_frames(value: str) -> list[int]:
    """Parse a comma-separated frame list."""

    frames: list[int] = []
    for part in value.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        frame = int(stripped)
        if frame < 0:
            raise argparse.ArgumentTypeError("frame indices must be non-negative")
        frames.append(frame)
    return frames


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beltmap-compare",
        description="Compare multiple BeltMap output directories with summary metrics and visual contact sheets.",
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run to compare as LABEL=OUTPUT_DIR. May be repeated.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("comparison_report"),
        help="Directory for comparison_report.md, summary.csv, and PNGs. Default: comparison_report",
    )
    parser.add_argument(
        "--frames",
        type=parse_frames,
        default=None,
        help="Comma-separated residual preview frames for the contact sheet, for example 0,248,496.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print generated artifact paths as JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    specs = [parse_run_spec(value) for value in args.run]
    artifacts = generate_comparison_report(
        specs,
        report_dir=args.report_dir,
        frames=args.frames,
    )
    if not args.quiet:
        print(
            json.dumps(
                {
                    "report": str(artifacts.report),
                    "summary_csv": str(artifacts.summary_csv),
                    "plots": {key: str(path) for key, path in artifacts.plots.items()},
                    "images": {key: str(path) for key, path in artifacts.images.items()},
                },
                indent=2,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
