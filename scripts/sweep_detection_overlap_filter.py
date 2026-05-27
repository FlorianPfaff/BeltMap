from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.filter_run_by_detection_overlap import filter_run


DEFAULT_SPECS = (
    "c25f03:0.01:25:0.3",
    "c35f03:0.01:35:0.3",
    "c50f04:0.01:50:0.4",
    "i005c35f04:0.005:35:0.4",
)
SUMMARY_FIELDS = [
    "label",
    "min_iou",
    "max_center_distance_px",
    "track_rescue_min_confirmed_fraction",
    "n_detections",
    "detections_per_frame",
    "n_filtered_velocity_estimates",
    "detection_area_median_px",
    "kept_fraction",
    "track_rescued_detections",
    "duplicate_suppressed_detections",
    "rejected_detections",
    "output_dir",
]


@dataclass(frozen=True)
class SweepSpec:
    label: str
    min_iou: float
    max_center_distance_px: float
    track_rescue_min_confirmed_fraction: float


def parse_spec(value: str) -> SweepSpec:
    parts = [part.strip() for part in value.replace(",", ":").split(":")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "spec must be label:min_iou:max_center_distance:track_rescue_fraction"
        )
    label, min_iou, max_center, rescue_fraction = parts
    if not label:
        raise argparse.ArgumentTypeError("spec label must not be empty")
    return SweepSpec(
        label=label,
        min_iou=float(min_iou),
        max_center_distance_px=float(max_center),
        track_rescue_min_confirmed_fraction=float(rescue_fraction),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep raw/BeltMap detection-overlap filter gates and summarize the resulting tracks."
    )
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--confirming-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--spec", type=parse_spec, action="append", default=None)
    parser.add_argument("--label-prefix", default="")
    parser.add_argument("--candidate-margin-px", type=float, default=2.0)
    parser.add_argument("--confirming-margin-px", type=float, default=2.0)
    parser.add_argument("--one-to-one-confirmation", action="store_true")
    parser.add_argument("--disable-track-rescue", action="store_true")
    parser.add_argument("--track-rescue-min-detections", type=int, default=3)
    parser.add_argument("--track-rescue-min-confirmed", type=int, default=2)
    parser.add_argument("--dedupe-iou-threshold", type=float, default=0.0)
    parser.add_argument("--dedupe-containment-threshold", type=float, default=0.0)
    parser.add_argument("--dedupe-center-distance-px", type=float, default=0.0)
    parser.add_argument("--dedupe-margin-px", type=float, default=0.0)
    parser.add_argument("--belt-velocity-px-per-frame", type=float, default=None)
    parser.add_argument("--min-track-length", type=int, default=2)
    parser.add_argument("--tracking-assignment-method", choices=("global", "greedy", "pyrecest_gnn"), default="global")
    parser.add_argument("--tracking-max-frame-gap", type=float, default=2.0)
    parser.add_argument("--tracking-area-cost-weight-px", type=float, default=1.0)
    parser.add_argument("--tracking-signal-cost-weight-px", type=float, default=0.5)
    parser.add_argument("--tracking-lateral-cost-weight", type=float, default=0.25)
    parser.add_argument("--tracking-max-area-ratio", type=float, default=3.0)
    parser.add_argument("--velocity-fit-method", choices=("linear", "theil_sen"), default="theil_sen")
    parser.add_argument("--track-filter-min-length", type=int, default=5)
    parser.add_argument("--track-filter-min-velocity-ratio-y", type=float, default=0.0)
    parser.add_argument("--track-filter-max-velocity-ratio-y", type=float, default=1.1)
    parser.add_argument("--track-filter-max-abs-x-velocity-px-per-frame", type=float, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def namespace_for_spec(args: argparse.Namespace, spec: SweepSpec) -> argparse.Namespace:
    payload = vars(args).copy()
    payload.update(
        {
            "output_dir": args.output_root / spec.label,
            "label": f"{args.label_prefix}{spec.label}",
            "min_iou": spec.min_iou,
            "max_center_distance_px": spec.max_center_distance_px,
            "track_rescue_min_confirmed_fraction": spec.track_rescue_min_confirmed_fraction,
            "allow_many_to_one_confirmation": not args.one_to_one_confirmation,
        }
    )
    payload.pop("output_root", None)
    payload.pop("spec", None)
    payload.pop("label_prefix", None)
    return argparse.Namespace(**payload)


def summary_row(spec: SweepSpec, output_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    confirmation = metadata.get("confirmation", {})
    return {
        "label": spec.label,
        "min_iou": spec.min_iou,
        "max_center_distance_px": spec.max_center_distance_px,
        "track_rescue_min_confirmed_fraction": spec.track_rescue_min_confirmed_fraction,
        "n_detections": metadata.get("n_detections"),
        "detections_per_frame": metadata.get("detections_per_frame"),
        "n_filtered_velocity_estimates": metadata.get("n_filtered_velocity_estimates"),
        "detection_area_median_px": metadata.get("detection_area_median_px"),
        "kept_fraction": confirmation.get("kept_fraction"),
        "track_rescued_detections": confirmation.get("track_rescued_detections"),
        "duplicate_suppressed_detections": confirmation.get("duplicate_suppressed_detections"),
        "rejected_detections": confirmation.get("rejected_detections"),
        "output_dir": str(output_dir),
    }


def write_summary(output_root: Path, rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "sweep_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Detection-overlap filter sweep",
        "",
        "| label | detections | detections/frame | filtered velocities | median area px | kept fraction | duplicate suppressed | rejected |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {label} | {n_detections} | {dpf:.3g} | {velocities} | {area} | {kept:.3g} | {duplicates} | {rejected} |".format(
                label=row["label"],
                n_detections=row["n_detections"],
                dpf=float(row["detections_per_frame"] or 0.0),
                velocities=row["n_filtered_velocity_estimates"],
                area="" if row["detection_area_median_px"] is None else f"{row['detection_area_median_px']:.3g}",
                kept=float(row["kept_fraction"] or 0.0),
                duplicates=row["duplicate_suppressed_detections"],
                rejected=row["rejected_detections"],
            )
        )
    (output_root / "sweep_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sweep(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs = args.spec or [parse_spec(value) for value in DEFAULT_SPECS]
    rows: list[dict[str, Any]] = []
    for spec in specs:
        spec_args = namespace_for_spec(args, spec)
        metadata = filter_run(spec_args)
        rows.append(summary_row(spec, spec_args.output_dir, metadata))
    write_summary(args.output_root, rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        rows = run_sweep(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if not args.quiet:
        print(json.dumps(rows, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
