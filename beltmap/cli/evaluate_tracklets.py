from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from beltmap.tracklet_evaluation import default_prediction_path
from beltmap.tracklet_evaluation import generate_tracklet_evaluation_report


def parse_iou_threshold(value: str) -> float:
    """Parse and validate an IoU threshold in [0, 1]."""

    try:
        threshold = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "IoU threshold must be a finite number in [0, 1]"
        ) from exc
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise argparse.ArgumentTypeError(
            "IoU threshold must be a finite number in [0, 1]"
        )
    return threshold


def _path_key(path: Path) -> str:
    """Return a normalized key for path-alias comparisons."""

    return os.path.normcase(str(path.expanduser().resolve(strict=False)))


def _paths_alias(left: Path, right: Path) -> bool:
    """Return whether paths name the same existing or prospective file."""

    try:
        return left.samefile(right)
    except (FileNotFoundError, OSError):
        return _path_key(left) == _path_key(right)


def _validate_artifact_paths(
    *,
    output_dir: Path,
    truth_path: Path,
    prediction_path: Path | None,
    metrics_path: Path | None,
    report_path: Path | None,
    matches_path: Path | None,
) -> None:
    """Reject output paths that alias inputs or another output artifact."""

    prediction_input = (
        default_prediction_path(output_dir)
        if prediction_path is None
        else prediction_path
    )
    inputs = (
        ("truth input", truth_path),
        ("prediction input", prediction_input),
    )
    outputs = (
        (
            "metrics output",
            output_dir / "tracklet_metrics.json"
            if metrics_path is None
            else metrics_path,
        ),
        (
            "report output",
            output_dir / "tracklet_report.md"
            if report_path is None
            else report_path,
        ),
        (
            "matches output",
            output_dir / "tracklet_matches.csv"
            if matches_path is None
            else matches_path,
        ),
    )

    prior_outputs: list[tuple[str, Path]] = []
    for output_label, output_path in outputs:
        for input_label, input_path in inputs:
            if _paths_alias(output_path, input_path):
                raise ValueError(
                    f"{output_label} must not overwrite {input_label}; "
                    f"both resolve to {output_path.resolve(strict=False)}"
                )
        for prior_label, prior_path in prior_outputs:
            if _paths_alias(output_path, prior_path):
                raise ValueError(
                    f"{prior_label} and {output_label} must use distinct paths; "
                    f"both resolve to {output_path.resolve(strict=False)}"
                )
        prior_outputs.append((output_label, output_path))


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
            "Track CSV to score. Default: OUTPUT_DIR/filtered_tracks.csv when "
            "present, otherwise OUTPUT_DIR/tracks.csv."
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
        _validate_artifact_paths(
            output_dir=args.output_dir,
            truth_path=args.truth_path,
            prediction_path=args.prediction_path,
            metrics_path=args.metrics_path,
            report_path=args.report_path,
            matches_path=args.matches_path,
        )
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
                    "prediction_path": (
                        None
                        if args.prediction_path is None
                        else str(args.prediction_path)
                    ),
                    "iou_threshold": args.iou_threshold,
                },
                indent=2,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
