from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from decimal import Decimal
from decimal import InvalidOperation
from pathlib import Path
from typing import Any

from beltmap.phase import BeltMotionModel
from beltmap.recurrent_artifacts import belt_revolution_indices
from beltmap.revolution_recurrence import BeltRevolutionRecurrenceConfig
from beltmap.revolution_recurrence import score_belt_revolution_track_recurrence
from beltmap.tracking import ParticleDetection
from beltmap.tracking import ParticleTrack


TRACK_FIELDS = [
    "track_id",
    "track_detection_index",
    "frame_index",
    "image",
    "label",
    "y",
    "x",
    "area_px",
    "bbox_top",
    "bbox_left",
    "bbox_bottom",
    "bbox_right",
    "mean_signal",
    "peak_signal",
    "recurrent_artifact_overlap_fraction",
    "recurrent_artifact_probability",
    "recurrent_artifact_required_peak_signal",
]
VELOCITY_FIELDS = [
    "track_id",
    "n_detections",
    "frame_start",
    "frame_end",
    "velocity_y_px_per_frame",
    "velocity_x_px_per_frame",
    "speed_px_per_frame",
    "belt_velocity_y_px_per_frame",
    "velocity_ratio_y",
    "belt_minus_particle_velocity_y_px_per_frame",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beltmap-filter-revolution-recurrence",
        description=(
            "Filter accepted tracks whose belt-coordinate center recurs in other "
            "exposed belt revolutions. Belt-fixedness alone is not enough: the "
            "current track is left out when recurrence evidence is computed."
        ),
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Default: INPUT_DIR/runtime_recurrence_filter",
    )
    parser.add_argument("--radius-y-px", type=float, default=8.0)
    parser.add_argument("--radius-x-px", type=float, default=8.0)
    parser.add_argument("--min-track-detections", type=int, default=5)
    parser.add_argument("--min-other-revolutions", type=int, default=2)
    parser.add_argument("--min-other-detections", type=int, default=2)
    parser.add_argument("--min-recurrence-fraction", type=float, default=1.0)
    parser.add_argument("--quiet", action="store_true")
    return parser


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def exact_integer(value: Any, *, name: str) -> int:
    """Parse an exact finite integer without truncating fractional values."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite integer")
    try:
        parsed = Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be a finite integer") from exc
    if not parsed.is_finite() or parsed != parsed.to_integral_value():
        raise ValueError(f"{name} must be a finite integer")
    return int(parsed)


def parse_optional_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    return None if value is None or str(value).strip() == "" else float(value)


def parse_detection(row: dict[str, str]) -> ParticleDetection:
    return ParticleDetection(
        frame_index=float(row["frame_index"]),
        label=exact_integer(row["label"], name="label"),
        y=float(row["y"]),
        x=float(row["x"]),
        area_px=exact_integer(row["area_px"], name="area_px"),
        bbox_top=exact_integer(row["bbox_top"], name="bbox_top"),
        bbox_left=exact_integer(row["bbox_left"], name="bbox_left"),
        bbox_bottom=exact_integer(row["bbox_bottom"], name="bbox_bottom"),
        bbox_right=exact_integer(row["bbox_right"], name="bbox_right"),
        mean_signal=parse_optional_float(row, "mean_signal"),
        peak_signal=parse_optional_float(row, "peak_signal"),
        recurrent_artifact_overlap_fraction=parse_optional_float(
            row,
            "recurrent_artifact_overlap_fraction",
        ),
        recurrent_artifact_probability=parse_optional_float(
            row,
            "recurrent_artifact_probability",
        ),
        recurrent_artifact_required_peak_signal=parse_optional_float(
            row,
            "recurrent_artifact_required_peak_signal",
        ),
    )


def parse_tracks(rows: list[dict[str, str]]) -> list[ParticleTrack]:
    grouped: dict[int, list[tuple[int, ParticleDetection]]] = {}
    for row in rows:
        track_id = exact_integer(row["track_id"], name="track_id")
        raw_index = row.get("track_detection_index", "")
        detection_index = (
            len(grouped.get(track_id, []))
            if raw_index == ""
            else exact_integer(raw_index, name="track_detection_index")
        )
        grouped.setdefault(track_id, []).append((detection_index, parse_detection(row)))
    return [
        ParticleTrack(
            track_id=track_id,
            detections=tuple(
                detection for _index, detection in sorted(items, key=lambda item: item[0])
            ),
        )
        for track_id, items in sorted(grouped.items())
    ]


def infer_frame_count(metadata: dict[str, Any], track_rows: list[dict[str, str]]) -> int:
    if metadata.get("n_images") is not None:
        return int(metadata["n_images"])
    max_frame = max((int(float(row["frame_index"])) for row in track_rows), default=-1)
    return max_frame + 1


def infer_belt_region(metadata: dict[str, Any]) -> tuple[int, int, int, int]:
    region = metadata.get("belt_region")
    if not isinstance(region, dict):
        raise ValueError("metadata.json is missing belt_region")
    return (
        int(region["top"]),
        int(region["left"]),
        int(region["height"]),
        int(region["width"]),
    )


def finite_positive(value: Any, *, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


def load_phase_px_by_frame(path: Path, frame_count: int) -> list[float]:
    rows = read_csv(path)
    phase_by_frame: list[float | None] = [None] * frame_count
    for row in rows:
        frame_index = int(float(row["frame_index"]))
        if 0 <= frame_index < frame_count:
            phase_by_frame[frame_index] = float(row["phase_px"])
    missing = [index for index, value in enumerate(phase_by_frame) if value is None]
    if missing:
        preview = ", ".join(str(index) for index in missing[:8])
        raise ValueError(
            f"phase_estimates.csv is missing {len(missing)} frames; first: {preview}"
        )
    return [float(value) for value in phase_by_frame]


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def accepted_track_ids(input_dir: Path) -> set[int]:
    score_rows = read_csv(input_dir / "track_scores.csv")
    if score_rows:
        return {
            exact_integer(row["track_id"], name="track_id")
            for row in score_rows
            if bool_value(row.get("accepted", False))
        }
    filtered_rows = read_csv(input_dir / "filtered_tracks.csv")
    if filtered_rows:
        return {exact_integer(row["track_id"], name="track_id") for row in filtered_rows}
    velocity_rows = read_csv(input_dir / "filtered_velocities.csv")
    return {exact_integer(row["track_id"], name="track_id") for row in velocity_rows}


def filter_rows_by_track_id(rows: list[dict[str, str]], ids: set[int]) -> list[dict[str, str]]:
    return [
        row for row in rows if exact_integer(row["track_id"], name="track_id") in ids
    ]


def write_report(path: Path, summary: dict[str, Any], rejected_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Belt-Revolution Runtime Recurrence Filter",
        "",
        "This runtime track-level gate rejects only tracks whose belt-coordinate",
        "center is supported by other detections in other exposed belt revolutions.",
        "A belt-fixed track without leave-current-track-out recurrence is kept.",
        "",
        "## Summary",
        "",
        f"- Source accepted tracks: {summary['source_accepted_tracks']}",
        f"- Rejected accepted tracks: {summary['rejected_accepted_tracks']}",
        f"- Accepted tracks after recurrence filter: {summary['accepted_tracks_after_filter']}",
        f"- Radius: y={summary['config']['radius_y_px']} px, x={summary['config']['radius_x_px']} px",
        f"- Required other revolutions: {summary['config']['min_other_revolutions']}",
        "",
        "## Rejected Tracks",
        "",
        "| track | detections | other exposed revs | hit revs | hit detections | recurrence fraction | read |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    if not rejected_rows:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | no accepted tracks rejected |")
    for row in rejected_rows:
        lines.append(
            f"| {row['track_id']} | {row['n_detections']} | "
            f"{row['other_exposed_revolutions']} | {row['other_hit_revolutions']} | "
            f"{row['other_hit_detections']} | {float(row['recurrence_fraction']):.3f} | "
            f"{row['causal_read']} |"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- This is a runtime recurrence filter, not an empty-frame specificity claim.",
            "- It requires enough exposed belt revolutions to distinguish one-off particles from recurring belt-coordinate artifacts.",
            "- It does not use belt-fixedness alone as a rejection signal.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def filter_revolution_recurrence(
    *,
    input_dir: Path,
    output_dir: Path,
    config: BeltRevolutionRecurrenceConfig,
) -> dict[str, Any]:
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("--output-dir must be different from --input-dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = read_json(input_dir / "metadata.json")
    track_rows = read_csv(input_dir / "tracks.csv")
    if not track_rows:
        raise FileNotFoundError(f"missing or empty tracks.csv in {input_dir}")
    tracks = parse_tracks(track_rows)
    frame_count = infer_frame_count(metadata, track_rows)
    _top, _left, frame_height, _frame_width = infer_belt_region(metadata)
    map_height = finite_positive(
        metadata.get("belt_map_height_px", metadata.get("belt_period_px_input")),
        name="belt_map_height_px",
    )
    belt_velocity = finite_positive(
        abs(float(metadata.get("belt_velocity_px_per_frame"))),
        name="belt_velocity_px_per_frame",
    )
    period_px = finite_positive(
        metadata.get("belt_period_px_input", map_height),
        name="belt_period_px_input",
    )
    phase_px_by_frame = load_phase_px_by_frame(input_dir / "phase_estimates.csv", frame_count)
    revolution_by_frame = belt_revolution_indices(
        frame_count,
        BeltMotionModel(
            image_velocity_px_per_frame=belt_velocity,
            period_px=period_px,
            reference_phase_px=float(metadata.get("reference_phase_px", 0.0)),
        ),
    )
    scores = score_belt_revolution_track_recurrence(
        tracks,
        phase_px_by_frame=phase_px_by_frame,
        revolution_by_frame=revolution_by_frame,
        frame_height_px=float(frame_height),
        map_height_px=map_height,
        config=config,
    )

    accepted_before = accepted_track_ids(input_dir)
    rejected_ids = {
        score.track_id
        for score in scores
        if score.track_id in accepted_before and score.runtime_recurrence_rejected
    }
    accepted_after = accepted_before - rejected_ids
    score_rows: list[dict[str, Any]] = []
    for score in scores:
        row = score.to_row()
        row["accepted_before_runtime_recurrence"] = score.track_id in accepted_before
        row["accepted_after_runtime_recurrence"] = score.track_id in accepted_after
        score_rows.append(row)

    write_csv(output_dir / "runtime_recurrence_track_scores.csv", score_rows)
    write_csv(
        output_dir / "runtime_recurrence_rejected_tracks.csv",
        [row for row in score_rows if row["accepted_before_runtime_recurrence"] and row["runtime_recurrence_rejected"]],
    )
    source_velocity_rows = read_csv(input_dir / "velocities.csv")
    source_filtered_track_rows = read_csv(input_dir / "filtered_tracks.csv") or track_rows
    write_csv(
        output_dir / "filtered_velocities.csv",
        filter_rows_by_track_id(source_velocity_rows, accepted_after),
        VELOCITY_FIELDS,
    )
    write_csv(
        output_dir / "filtered_tracks.csv",
        filter_rows_by_track_id(source_filtered_track_rows, accepted_after),
        TRACK_FIELDS,
    )
    summary = {
        "method": "runtime_belt_revolution_recurrence_filter",
        "source_run": str(input_dir),
        "tracks_scored": len(scores),
        "source_accepted_tracks": len(accepted_before),
        "rejected_accepted_tracks": len(rejected_ids),
        "accepted_tracks_after_filter": len(accepted_after),
        "rejected_track_ids": sorted(rejected_ids),
        "observed_revolutions": sorted({int(value) for value in revolution_by_frame}),
        "config": asdict(config),
    }
    (output_dir / "runtime_recurrence_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(
        output_dir / "runtime_recurrence_report.md",
        summary,
        [row for row in score_rows if row["accepted_before_runtime_recurrence"] and row["runtime_recurrence_rejected"]],
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = args.output_dir or args.input_dir / "runtime_recurrence_filter"
    try:
        config = BeltRevolutionRecurrenceConfig(
            radius_y_px=args.radius_y_px,
            radius_x_px=args.radius_x_px,
            min_track_detections=args.min_track_detections,
            min_other_revolutions=args.min_other_revolutions,
            min_other_detections=args.min_other_detections,
            min_recurrence_fraction=args.min_recurrence_fraction,
        )
        summary = filter_revolution_recurrence(
            input_dir=args.input_dir,
            output_dir=output_dir,
            config=config,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if not args.quiet:
        print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
