from __future__ import annotations

import argparse
import json
from pathlib import Path

from beltmap.benchmark import generate_benchmark_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beltmap-benchmark",
        description=(
            "Compute synthetic ground-truth benchmark metrics for a BeltMap output "
            "directory using a synthetic_metadata.json truth file."
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
        default=Path("data/images/synthetic_metadata.json"),
        help="Synthetic ground-truth metadata JSON. Default: data/images/synthetic_metadata.json",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        help="Benchmark JSON output path. Default: OUTPUT_DIR/benchmark_metrics.json",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        help="Benchmark Markdown report path. Default: OUTPUT_DIR/benchmark_report.md",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.25,
        help="IoU threshold for matching detections to synthetic boxes. Default: 0.25",
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
        artifacts = generate_benchmark_report(
            output_dir=args.output_dir,
            truth_path=args.truth_path,
            metrics_path=args.metrics_path,
            report_path=args.report_path,
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
                },
                indent=2,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
