from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from beltmap.tracklet_evaluation import generate_tracklet_evaluation_report


def parse_iou_threshold(value: str) -> float:
    """Parse and validate an IoU threshold in [0, 1]."""

    try:
        threshold = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("IoU threshold must be a finite number in [0, 1]") from exc
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise argparse.ArgumentTypeError("IoU threshold must be a finite number in [0, 1]")
    return threshold


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beltmap-evaluate-tracklets",
        description=(
            "Evaluate PyRecEst tracks against sparse short real-data tracklet "
            "annotations and write HOTA-style detection/association metrics."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="BeltMap output directory to score. Default: outputs",
    )
    parser.add_argument(
        "--truth-path",
        type=Path,
        required=True,
        help="CSV/JSON sparse tracklet annotation file.",
    )
    parser.add_argument(
        "--prediction-path",
        type=Path,
        default=None,
        help=(
            "Track CSV to score. Default: OUTPUT_DIR/filtered_tracks.csv when present, "
            "otherwise OUTPUT_DIR/tracks.csv."
        ),
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=None,
        help="JSON metrics output path. Default: OUTPUT_DIR/tracklet_metrics.json",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Markdown report path. Default: OUTPUT_DIR/tracklet_report.md",
    )
    parser.add_argument(
        "--matches-path",
        type=Path,
        default=None,
        help="CSV match-inspection path. Default: OUTPUT_DIR/tracklet_matches.csv",
    )
    parser.add_argument(
        "--iou-threshold",
        type=parse_iou_threshold,
        default=0.25,
        help="Frame-level IoU threshold for truth/prediction matches. Default: 0.25",
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
        artifacts = generate_tracklet_evaluation_report(
            output_dir=args.output_dir,
            truth_path=args.truth_path,
            prediction_path=args.prediction_path,
            metrics_path=args.metrics_path,
            report_path=args.report_path,
            matches_path=args.matches_path,
            iou_threshold=args.iou_threshold,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if not args.quiet:
        print(
            json.dumps(
                {
                    "metrics": str(artifacts.metrics),
                    "report": str(artifacts.report),
                    "matches": str(artifacts.matches),
                    "truth_path": str(args.truth_path),
                    "prediction_path": None if args.prediction_path is None else str(args.prediction_path),
                    "iou_threshold": args.iou_threshold,
                },
                indent=2,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
