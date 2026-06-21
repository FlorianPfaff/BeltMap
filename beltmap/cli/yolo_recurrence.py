from __future__ import annotations

import argparse
import json
from pathlib import Path

from beltmap.yolo_recurrence import CropRegion, score_yolo_recurrence


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beltmap-yolo-recurrence",
        description="Score exported detector boxes with belt-coordinate recurrence.",
    )
    parser.add_argument("--detections-csv", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase-estimates-csv", type=Path, default=None)
    parser.add_argument("--belt-map-path", type=Path, default=None)
    parser.add_argument("--map-height-px", type=float, default=None)
    parser.add_argument("--belt-velocity-px-per-frame", type=float, default=None)
    parser.add_argument("--phase-offset-px", type=float, default=0.0)
    parser.add_argument("--belt-region", default="0,0,0,0")
    parser.add_argument("--frame-index-pattern", default="([0-9]+)")
    parser.add_argument("--max-revolutions", type=_positive_int, default=1)
    parser.add_argument("--revisit-search-window-frames", type=_nonnegative_int, default=2)
    parser.add_argument("--recurrence-threshold", type=float, default=0.5)
    parser.add_argument("--min-recurrent-revisits", type=_positive_int, default=1)
    parser.add_argument("--min-original-excess", type=float, default=1.0)
    parser.add_argument("--signal-margin-px", type=_nonnegative_int, default=2)
    parser.add_argument("--background-margin-px", type=_nonnegative_int, default=12)
    parser.add_argument("--patch-correlation-margin-px", type=_nonnegative_int, default=4)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = score_yolo_recurrence(
            detections_csv=args.detections_csv,
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            phase_estimates_csv=args.phase_estimates_csv,
            belt_map_path=args.belt_map_path,
            map_height_px=args.map_height_px,
            belt_velocity_px_per_frame=args.belt_velocity_px_per_frame,
            phase_offset_px=args.phase_offset_px,
            crop_region=CropRegion.parse(args.belt_region),
            frame_index_pattern=args.frame_index_pattern,
            max_revolutions=args.max_revolutions,
            revisit_search_window_frames=args.revisit_search_window_frames,
            recurrence_threshold=args.recurrence_threshold,
            min_recurrent_revisits=args.min_recurrent_revisits,
            min_original_excess=args.min_original_excess,
            signal_margin_px=args.signal_margin_px,
            background_margin_px=args.background_margin_px,
            patch_correlation_margin_px=args.patch_correlation_margin_px,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if not args.quiet:
        payload = {
            "output_dir": str(summary.output_dir),
            "n_input_detections": summary.n_input_detections,
            "n_hard_kept": summary.n_hard_kept,
            "n_hard_rejected": summary.n_hard_rejected,
            "n_rerank_detections": summary.n_rerank_detections,
            "n_frames": summary.n_frames,
            "feature_csv": str(summary.feature_csv),
            "hard_filter_run": str(summary.output_dir / "hard_filter"),
            "rerank_run": str(summary.output_dir / "rerank"),
        }
        print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
