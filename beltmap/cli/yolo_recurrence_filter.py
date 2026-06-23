from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# Import for side effect: patch beltmap.yolo_recurrence.row_key so same-frame,
# same-class YOLO detections do not overwrite each other's recurrence features.
import beltmap.yolo_recurrence_key_patch  # noqa: F401

# Import for side effect: score recurrence on pixelwise residual evidence, not
# raw intensity minus a background percentile.
import beltmap.yolo_recurrence_residual_excess_patch  # noqa: F401
from beltmap.yolo_recurrence import (
    YoloRecurrenceConfig,
    parse_belt_region,
    run_yolo_recurrence_filter,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beltmap-yolo-recurrence-filter",
        description=(
            "Post-filter raw YOLO detections with BeltMap belt-coordinate "
            "recurrence evidence from previous and next belt revolutions."
        ),
    )
    parser.add_argument("--yolo-run-dir", type=Path, required=True)
    parser.add_argument("--beltmap-reference-dir", type=Path, required=True)
    parser.add_argument("--source-image-dir", type=Path, required=True)
    parser.add_argument("--truth-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=500)
    parser.add_argument("--belt-region", default="0,220,1330,1800")
    parser.add_argument("--hard-ratio-threshold", type=float, default=0.40)
    parser.add_argument("--hard-min-revisits", type=int, default=2)
    parser.add_argument("--patch-margin-px", type=int, default=4)
    parser.add_argument("--min-patch-size-px", type=int, default=9)
    parser.add_argument("--excess-floor", type=float, default=1.0)
    parser.add_argument("--froc-max-thresholds", type=int, default=250)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--bootstrap-block-length-frames", type=int, default=5)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.frame_count < 1:
            raise ValueError("--frame-count must be positive")
        if args.hard_min_revisits < 1:
            raise ValueError("--hard-min-revisits must be positive")
        if args.hard_ratio_threshold < 0 or not math.isfinite(args.hard_ratio_threshold):
            raise ValueError("--hard-ratio-threshold must be finite and non-negative")
        if args.patch_margin_px < 0:
            raise ValueError("--patch-margin-px must be non-negative")
        if args.min_patch_size_px < 1:
            raise ValueError("--min-patch-size-px must be positive")
        if args.excess_floor <= 0 or not math.isfinite(args.excess_floor):
            raise ValueError("--excess-floor must be finite and positive")
        if args.froc_max_thresholds < 1:
            raise ValueError("--froc-max-thresholds must be positive")
        if args.bootstrap_samples < 0:
            raise ValueError("--bootstrap-samples must be non-negative")
        if args.bootstrap_block_length_frames < 1:
            raise ValueError("--bootstrap-block-length-frames must be positive")
        config = YoloRecurrenceConfig(
            frame_count=args.frame_count,
            belt_region=parse_belt_region(args.belt_region),
            hard_ratio_threshold=args.hard_ratio_threshold,
            hard_min_revisits=args.hard_min_revisits,
            patch_margin_px=args.patch_margin_px,
            min_patch_size_px=args.min_patch_size_px,
            excess_floor=args.excess_floor,
            froc_max_thresholds=args.froc_max_thresholds,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_block_length_frames=args.bootstrap_block_length_frames,
        )
        summary = run_yolo_recurrence_filter(
            yolo_run_dir=args.yolo_run_dir,
            beltmap_reference_dir=args.beltmap_reference_dir,
            source_image_dir=args.source_image_dir,
            truth_path=args.truth_path,
            output_dir=args.output_dir,
            config=config,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    if not args.quiet:
        print(
            json.dumps(
                {
                    "output_dir": str(summary.output_dir),
                    "features_csv": str(summary.features_csv),
                    "hard_run_dir": str(summary.hard_run_dir),
                    "rerank_run_dir": str(summary.rerank_run_dir),
                    "report_md": str(summary.report_md),
                    "contact_sheet_png": str(summary.contact_sheet_png),
                    "compare_summary_csv": (
                        None
                        if summary.compare_summary_csv is None
                        else str(summary.compare_summary_csv)
                    ),
                    "n_detections": summary.n_detections,
                    "n_hard_rejected": summary.n_hard_rejected,
                    "n_raw_false_positives_removed": summary.n_raw_false_positives_removed,
                    "n_raw_true_positives_removed": summary.n_raw_true_positives_removed,
                },
                indent=2,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
