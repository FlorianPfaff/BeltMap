from __future__ import annotations

import argparse
import json
from pathlib import Path

from beltmap.yolo_export import (
    DEFAULT_FRAME_INDEX_PATTERN,
    export_yolo_predictions_to_beltmap_run,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beltmap-yolo-export",
        description=(
            "Convert Ultralytics YOLO prediction .txt files into a BeltMap-style "
            "output directory with detections.csv and detections_per_frame.csv. "
            "The exported directory can be passed to beltmap-compare."
        ),
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        required=True,
        help="Directory containing YOLO .txt prediction files, usually a predict/labels directory.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing the crop images used for prediction.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output BeltMap-compatible run directory.",
    )
    parser.add_argument(
        "--frame-index-pattern",
        default=DEFAULT_FRAME_INDEX_PATTERN,
        help=(
            "Regex used to infer the source frame index from image stems. The last match is used. "
            f"Default: {DEFAULT_FRAME_INDEX_PATTERN!r}"
        ),
    )
    parser.add_argument(
        "--default-confidence",
        type=float,
        default=1.0,
        help="Confidence assigned to 5-column YOLO label files without confidence values. Default: 1.0",
    )
    parser.add_argument(
        "--allow-label-without-image",
        action="store_true",
        help="Ignore YOLO label files whose stem has no matching image instead of raising an error.",
    )
    parser.add_argument(
        "--source",
        default="yolo",
        help="Source string written into detections.csv. Default: yolo",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress JSON summary output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = export_yolo_predictions_to_beltmap_run(
            labels_dir=args.labels_dir,
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            frame_index_pattern=args.frame_index_pattern,
            default_confidence=args.default_confidence,
            allow_label_without_image=args.allow_label_without_image,
            source=args.source,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    if not args.quiet:
        print(
            json.dumps(
                {
                    "output_dir": str(summary.output_dir),
                    "images_dir": str(summary.images_dir),
                    "labels_dir": str(summary.labels_dir),
                    "n_images": summary.n_images,
                    "n_label_files": summary.n_label_files,
                    "n_detections": summary.n_detections,
                    "n_frames_with_detections": summary.n_frames_with_detections,
                    "frame_index_min": summary.frame_index_min,
                    "frame_index_max": summary.frame_index_max,
                    "detections_csv": str(summary.output_dir / "detections.csv"),
                    "detections_per_frame_csv": str(summary.output_dir / "detections_per_frame.csv"),
                },
                indent=2,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
