from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from beltmap.advanced_quality import evaluate_real_detections


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beltmap-evaluate-real",
        description="Evaluate detections against sparse manually annotated real-data boxes.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="BeltMap output directory containing detections.csv.")
    parser.add_argument("--labels", type=Path, required=True, help="JSON labels with frames[].boxes[].")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold used for greedy matching.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = evaluate_real_detections(args.output_dir, args.labels, iou_threshold=args.iou_threshold)
    output = args.output_dir / "real_label_metrics.json"
    output.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")
    print(json.dumps(asdict(metrics), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
