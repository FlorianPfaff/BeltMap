from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from beltmap.tracking import (
    ParticleVelocity,
    TrackFilterConfig,
    score_particle_velocities,
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
    write_csv(track_scores_path, [asdict(score) for score in scores], TRACK_SCORE_FIELDS)
    write_csv(filtered_velocities_path, [asdict(velocity) for velocity in filtered], VELOCITY_FIELDS)
    return {
        "track_scores": str(track_scores_path),
        "filtered_velocities": str(filtered_velocities_path),
        "velocity_estimates": len(velocities),
        "accepted_velocity_estimates": len(filtered),
        "rejected_velocity_estimates": len(velocities) - len(filtered),
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
