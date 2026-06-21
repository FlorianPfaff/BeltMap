from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from beltmap.compare_runs import DEFAULT_FROC_MAX_THRESHOLDS, parse_run_spec
from beltmap.ghost_objective import (
    GhostObjectiveWeights,
    labeled_evidence_from_runs,
    labeled_evidence_from_summary_csv,
    map_only_evidence_from_label_paths,
    map_only_evidence_from_summary_csv,
    parse_label_path,
    run_ghost_objective,
)


def parse_finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a finite number") from exc
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be a finite number")
    return parsed


def parse_iou_threshold(value: str) -> float:
    parsed = parse_finite_float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("IoU threshold must be in [0, 1]")
    return parsed


def parse_nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beltmap-ghost-objective",
        description=(
            "Select BeltMap configurations with a ghost-aware objective that "
            "combines labeled detection quality and map-only negative-control penalties."
        ),
    )
    parser.add_argument(
        "--truth-path",
        type=Path,
        help="Reviewed CSV/JSON truth labels. Required when computing labeled metrics from --run.",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run to score as LABEL=OUTPUT_DIR. May be repeated.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        help="Existing beltmap-compare summary.csv to use instead of or in addition to --run.",
    )
    parser.add_argument(
        "--map-only-metrics",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Map-only metrics JSON/CSV for one variant. May be repeated.",
    )
    parser.add_argument(
        "--map-only-summary-csv",
        type=Path,
        help="CSV table containing map-only false detection/track metrics for several variants.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Variant to include even if some evidence is missing. May be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for ghost_objective_table.csv, report, plot, and config_selection.json.",
    )
    parser.add_argument(
        "--truth-iou-threshold",
        type=parse_iou_threshold,
        default=0.25,
        help="IoU threshold used when --truth-path and --run compute labeled metrics. Default: 0.25",
    )
    parser.add_argument(
        "--froc-max-thresholds",
        type=parse_nonnegative_int,
        default=DEFAULT_FROC_MAX_THRESHOLDS,
        help=(
            "Maximum distinct score thresholds for direct-run labeled FROC. "
            f"Default: {DEFAULT_FROC_MAX_THRESHOLDS}; 0 requests exact."
        ),
    )
    parser.add_argument("--weight-f1", type=parse_finite_float, default=1.0)
    parser.add_argument("--weight-fp-frame", type=parse_finite_float, default=0.01)
    parser.add_argument("--weight-map-false-detections", type=parse_finite_float, default=0.05)
    parser.add_argument("--weight-map-false-long", type=parse_finite_float, default=1.0)
    parser.add_argument("--weight-map-false-accepted", type=parse_finite_float, default=1.0)
    parser.add_argument("--weight-small-accepted", type=parse_finite_float, default=0.1)
    parser.add_argument("--weight-mask-burden", type=parse_finite_float, default=0.1)
    parser.add_argument("--quiet", action="store_true", help="Do not print generated artifact JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        labeled = {}
        if args.summary_csv is not None:
            labeled.update(labeled_evidence_from_summary_csv(args.summary_csv))
        if args.run:
            if args.truth_path is None:
                parser.error("--truth-path is required when --run is supplied")
            specs = [parse_run_spec(value) for value in args.run]
            froc_max_thresholds = None if args.froc_max_thresholds == 0 else args.froc_max_thresholds
            labeled.update(
                labeled_evidence_from_runs(
                    specs,
                    truth_path=args.truth_path,
                    truth_iou_threshold=args.truth_iou_threshold,
                    froc_max_thresholds=froc_max_thresholds,
                )
            )
        if not labeled:
            parser.error("provide --summary-csv or --truth-path with at least one --run")

        map_only = {}
        if args.map_only_summary_csv is not None:
            map_only.update(map_only_evidence_from_summary_csv(args.map_only_summary_csv))
        if args.map_only_metrics:
            for value in args.map_only_metrics:
                parse_label_path(value)
            map_only.update(map_only_evidence_from_label_paths(args.map_only_metrics))
        if not map_only:
            parser.error("provide --map-only-summary-csv or at least one --map-only-metrics LABEL=PATH")

        weights = GhostObjectiveWeights(
            f1=args.weight_f1,
            fp_frame=args.weight_fp_frame,
            map_false_detections=args.weight_map_false_detections,
            map_false_long=args.weight_map_false_long,
            map_false_accepted=args.weight_map_false_accepted,
            small_accepted=args.weight_small_accepted,
            mask_burden=args.weight_mask_burden,
        )
        artifacts = run_ghost_objective(
            output_dir=args.output_dir,
            labeled=labeled,
            map_only=map_only,
            weights=weights,
            variants=args.variant,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    if not args.quiet:
        print(
            json.dumps(
                {
                    "table_csv": str(artifacts.table_csv),
                    "report_md": str(artifacts.report_md),
                    "plot_png": str(artifacts.plot_png),
                    "config_selection_json": str(artifacts.config_selection_json),
                },
                indent=2,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
