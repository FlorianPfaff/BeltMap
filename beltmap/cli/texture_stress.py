from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from beltmap.compare_runs import parse_run_spec
from beltmap.texture_stress import generate_texture_stress_report


def parse_iou_threshold(value: str) -> float:
    """Parse and validate a detection-match IoU threshold."""

    try:
        threshold = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "truth IoU threshold must be a finite number in [0, 1]"
        ) from exc
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise argparse.ArgumentTypeError(
            "truth IoU threshold must be a finite number in [0, 1]"
        )
    return threshold


def parse_quartiles(value: str) -> int:
    """Parse the number of requested stress subsets."""

    try:
        quartiles = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("quartiles must be an integer >= 2") from exc
    if quartiles < 2:
        raise argparse.ArgumentTypeError("quartiles must be an integer >= 2")
    return quartiles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beltmap-texture-stress",
        description=(
            "Stratify BeltMap output runs by frame-level texture/residual stress "
            "and report detection/tracking metrics inside each subset."
        ),
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run to analyze as LABEL=OUTPUT_DIR. May be repeated.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("texture_stress_report"),
        help=(
            "Directory for texture_stress_report.md, texture_stress_frames.csv, "
            "texture_stress_summary.csv, and PNG plots. Default: texture_stress_report"
        ),
    )
    parser.add_argument(
        "--reference-run",
        default=None,
        help=(
            "Run label or output directory used to compute the stress score. "
            "Default: the first --run."
        ),
    )
    parser.add_argument(
        "--quartiles",
        type=parse_quartiles,
        default=4,
        help="Number of ordered stress subsets to create. Default: 4.",
    )
    parser.add_argument(
        "--truth-path",
        type=Path,
        default=None,
        help=(
            "Optional CSV/JSON file with manually labeled crop-local particle boxes. "
            "When supplied, the report includes precision/recall/F1 per stress subset."
        ),
    )
    parser.add_argument(
        "--truth-iou-threshold",
        type=parse_iou_threshold,
        default=0.25,
        help="IoU threshold used to match detections to labeled boxes. Default: 0.25",
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
    try:
        specs = [parse_run_spec(value) for value in args.run]
        artifacts = generate_texture_stress_report(
            specs,
            report_dir=args.report_dir,
            reference_label=args.reference_run,
            quartiles=args.quartiles,
            truth_path=args.truth_path,
            truth_iou_threshold=args.truth_iou_threshold,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if not args.quiet:
        print(
            json.dumps(
                {
                    "report": str(artifacts.report),
                    "frames_csv": str(artifacts.frames_csv),
                    "summary_csv": str(artifacts.summary_csv),
                    "plots": {key: str(path) for key, path in artifacts.plots.items()},
                    "reference_run": args.reference_run,
                    "quartiles": args.quartiles,
                    "truth_path": None if args.truth_path is None else str(args.truth_path),
                    "truth_iou_threshold": args.truth_iou_threshold,
                },
                indent=2,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
