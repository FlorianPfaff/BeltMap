from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from beltmap.tracking import (
    ParticleDetection,
    ParticleTrackingConfig,
    ParticleVelocity,
    TrackFilterConfig,
    score_particle_velocities,
    track_particle_detections,
)


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
TRACK_SCORE_FIELDS = [
    "track_id",
    "n_detections",
    "frame_start",
    "frame_end",
    "velocity_y_px_per_frame",
    "velocity_x_px_per_frame",
    "velocity_ratio_y",
    "abs_x_velocity_px_per_frame",
    "passes_min_track_length",
    "passes_velocity_ratio",
    "passes_lateral_velocity",
    "accepted",
    "plausibility_score",
]
TRACK_DETECTION_FIELDS = [
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
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beltmap-filter-tracks",
        description="Post-process BeltMap velocities.csv with track-level plausibility gates.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="BeltMap output directory containing velocities.csv. Default: outputs",
    )
    parser.add_argument(
        "--min-track-length",
        type=int,
        default=5,
        help="Minimum detections per accepted track. Default: 5",
    )
    parser.add_argument(
        "--min-velocity-ratio-y",
        type=float,
        default=0.0,
        help="Minimum accepted particle/belt vertical velocity ratio. Default: 0.0",
    )
    parser.add_argument(
        "--max-velocity-ratio-y",
        type=float,
        default=1.1,
        help="Maximum accepted particle/belt vertical velocity ratio. Default: 1.1",
    )
    parser.add_argument(
        "--max-abs-x-velocity-px-per-frame",
        type=float,
        default=None,
        help="Optional maximum accepted absolute lateral velocity.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print generated artifact paths and counts as JSON.",
    )
    return parser


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing velocity file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_optional_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_velocity(row: dict[str, str]) -> ParticleVelocity:
    return ParticleVelocity(
        track_id=int(row["track_id"]),
        n_detections=int(row["n_detections"]),
        frame_start=float(row["frame_start"]),
        frame_end=float(row["frame_end"]),
        velocity_y_px_per_frame=float(row["velocity_y_px_per_frame"]),
        velocity_x_px_per_frame=float(row["velocity_x_px_per_frame"]),
        speed_px_per_frame=float(row["speed_px_per_frame"]),
        belt_velocity_y_px_per_frame=float(row["belt_velocity_y_px_per_frame"]),
        velocity_ratio_y=float(row["velocity_ratio_y"]),
        belt_minus_particle_velocity_y_px_per_frame=float(
            row["belt_minus_particle_velocity_y_px_per_frame"]
        ),
    )


def parse_detection(row: dict[str, str]) -> ParticleDetection:
    return ParticleDetection(
        frame_index=float(row["frame_index"]),
        label=int(row["label"]),
        y=float(row["y"]),
        x=float(row["x"]),
        area_px=int(row["area_px"]),
        bbox_top=int(float(row["bbox_top"])),
        bbox_left=int(float(row["bbox_left"])),
        bbox_bottom=int(float(row["bbox_bottom"])),
        bbox_right=int(float(row["bbox_right"])),
        mean_signal=None if row.get("mean_signal", "") == "" else float(row["mean_signal"]),
        peak_signal=None if row.get("peak_signal", "") == "" else float(row["peak_signal"]),
    )


def option_value(config: dict, name: str) -> str | None:
    options = config.get("options", {})
    if not isinstance(options, dict):
        return None
    option = options.get(name)
    if not isinstance(option, dict):
        return None
    value = option.get("value")
    return None if value is None or str(value).strip() == "" else str(value)


def reconstruct_track_rows(output_dir: Path) -> list[dict[str, str]]:
    tracks_path = output_dir / "tracks.csv"
    if tracks_path.is_file():
        return read_optional_csv_rows(tracks_path)

    detection_rows = read_optional_csv_rows(output_dir / "detections.csv")
    if not detection_rows:
        return []

    metadata = read_json(output_dir / "metadata.json")
    config = read_json(output_dir / "config_resolved.json")
    velocity = float(metadata.get("belt_velocity_px_per_frame", 0.0))
    max_match_text = option_value(config, "max_match_distance_px")
    max_match = (
        float(max_match_text)
        if max_match_text is not None
        else max(5.0, 1.5 * abs(velocity))
    )
    detections_by_frame: list[list[ParticleDetection]] = []
    images_by_frame: dict[int, str] = {}
    for row in detection_rows:
        frame_index = int(float(row["frame_index"]))
        while len(detections_by_frame) <= frame_index:
            detections_by_frame.append([])
        detections_by_frame[frame_index].append(parse_detection(row))
        images_by_frame.setdefault(frame_index, row.get("image", ""))

    tracks = track_particle_detections(
        detections_by_frame,
        frame_indices=[float(index) for index in range(len(detections_by_frame))],
        config=ParticleTrackingConfig(
            max_match_distance_px=max_match,
            velocity_prior_y_px_per_frame=0.8 * velocity,
        ),
    )
    rows: list[dict] = []
    for track in tracks:
        for detection_index, detection in enumerate(track.detections):
            frame_index = int(detection.frame_index)
            rows.append(
                {
                    "track_id": track.track_id,
                    "track_detection_index": detection_index,
                    "frame_index": frame_index,
                    "image": images_by_frame.get(frame_index, ""),
                    "label": detection.label,
                    "y": detection.y,
                    "x": detection.x,
                    "area_px": detection.area_px,
                    "bbox_top": detection.bbox_top,
                    "bbox_left": detection.bbox_left,
                    "bbox_bottom": detection.bbox_bottom,
                    "bbox_right": detection.bbox_right,
                    "mean_signal": "" if detection.mean_signal is None else detection.mean_signal,
                    "peak_signal": "" if detection.peak_signal is None else detection.peak_signal,
                }
            )
    write_csv(tracks_path, rows, TRACK_DETECTION_FIELDS)
    return rows


def filter_tracks(
    output_dir: Path,
    *,
    config: TrackFilterConfig,
) -> dict[str, object]:
    velocities = [parse_velocity(row) for row in read_csv_rows(output_dir / "velocities.csv")]
    scores = score_particle_velocities(velocities, config=config)
    accepted_ids = {score.track_id for score in scores if score.accepted}
    filtered = [velocity for velocity in velocities if velocity.track_id in accepted_ids]
    track_scores_path = output_dir / "track_scores.csv"
    filtered_velocities_path = output_dir / "filtered_velocities.csv"
    filtered_tracks_path = output_dir / "filtered_tracks.csv"
    write_csv(track_scores_path, [asdict(score) for score in scores], TRACK_SCORE_FIELDS)
    write_csv(filtered_velocities_path, [asdict(velocity) for velocity in filtered], VELOCITY_FIELDS)
    track_rows = reconstruct_track_rows(output_dir)
    filtered_track_count: int | None = None
    filtered_tracks_result: str | None = None
    if track_rows:
        filtered_track_rows = [
            row for row in track_rows if int(row["track_id"]) in accepted_ids
        ]
        write_csv(filtered_tracks_path, filtered_track_rows, TRACK_DETECTION_FIELDS)
        filtered_track_count = len(filtered_track_rows)
        filtered_tracks_result = str(filtered_tracks_path)
    return {
        "track_scores": str(track_scores_path),
        "filtered_velocities": str(filtered_velocities_path),
        "filtered_tracks": filtered_tracks_result,
        "velocity_estimates": len(velocities),
        "accepted_velocity_estimates": len(filtered),
        "rejected_velocity_estimates": len(velocities) - len(filtered),
        "accepted_track_detection_rows": filtered_track_count,
        "track_filter": asdict(config),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = TrackFilterConfig(
        min_track_length=args.min_track_length,
        min_velocity_ratio_y=args.min_velocity_ratio_y,
        max_velocity_ratio_y=args.max_velocity_ratio_y,
        max_abs_x_velocity_px_per_frame=args.max_abs_x_velocity_px_per_frame,
    )
    payload = filter_tracks(args.output_dir, config=config)
    if not args.quiet:
        print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
