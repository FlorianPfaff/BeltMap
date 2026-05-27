from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from beltmap.tracking import (
    ParticleDetection,
    ParticleTrackingConfig,
    TrackFilterConfig,
    estimate_particle_velocities_vs_belt,
    score_particle_velocities,
    track_particle_detections,
)


DETECTION_FIELDS = [
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
CONFIRMATION_FIELDS = [
    "confirmation_iou",
    "confirmation_center_distance_px",
    "confirmation_label",
    "confirmation_track_id",
    "confirmation_rescued_by_track",
]
TRACK_DETECTION_FIELDS = [
    "track_id",
    "track_detection_index",
    *DETECTION_FIELDS,
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
PREVIEW_PATTERNS = (
    "residual_frame_*.png",
    "residual_fixed_frame_*.png",
    "raw_frame_*.png",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter one detection run to detections confirmed by overlapping detections from another run."
    )
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--confirming-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="raw_confirmed")
    parser.add_argument("--min-iou", type=float, default=0.01)
    parser.add_argument("--max-center-distance-px", type=float, default=35.0)
    parser.add_argument("--candidate-margin-px", type=float, default=2.0)
    parser.add_argument("--confirming-margin-px", type=float, default=2.0)
    parser.add_argument("--disable-track-rescue", action="store_true")
    parser.add_argument("--track-rescue-min-detections", type=int, default=3)
    parser.add_argument("--track-rescue-min-confirmed", type=int, default=2)
    parser.add_argument("--track-rescue-min-confirmed-fraction", type=float, default=0.3)
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


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.min_iou <= 1.0:
        raise ValueError("--min-iou must be in [0, 1]")
    for name in ("max_center_distance_px", "candidate_margin_px", "confirming_margin_px"):
        value = getattr(args, name)
        if value < 0 or not math.isfinite(value):
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative")
    if args.min_track_length < 1 or args.track_filter_min_length < 1:
        raise ValueError("track length thresholds must be positive")
    if args.track_rescue_min_detections < 1 or args.track_rescue_min_confirmed < 1:
        raise ValueError("track rescue count thresholds must be positive")
    if not 0.0 <= args.track_rescue_min_confirmed_fraction <= 1.0:
        raise ValueError("--track-rescue-min-confirmed-fraction must be in [0, 1]")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def optional_float(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "")
    return None if value == "" or value is None else float(value)


def parse_detection(row: dict[str, str]) -> ParticleDetection:
    return ParticleDetection(
        frame_index=float(row["frame_index"]),
        label=int(float(row["label"])),
        y=float(row["y"]),
        x=float(row["x"]),
        area_px=int(float(row["area_px"])),
        bbox_top=int(float(row["bbox_top"])),
        bbox_left=int(float(row["bbox_left"])),
        bbox_bottom=int(float(row["bbox_bottom"])),
        bbox_right=int(float(row["bbox_right"])),
        mean_signal=optional_float(row, "mean_signal"),
        peak_signal=optional_float(row, "peak_signal"),
        recurrent_artifact_overlap_fraction=optional_float(row, "recurrent_artifact_overlap_fraction"),
        recurrent_artifact_probability=optional_float(row, "recurrent_artifact_probability"),
        recurrent_artifact_required_peak_signal=optional_float(row, "recurrent_artifact_required_peak_signal"),
    )


def detection_row(detection: ParticleDetection, *, image: str, track_id: int | None = None, track_detection_index: int | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "frame_index": int(detection.frame_index),
        "image": image,
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
        "recurrent_artifact_overlap_fraction": (
            "" if detection.recurrent_artifact_overlap_fraction is None else detection.recurrent_artifact_overlap_fraction
        ),
        "recurrent_artifact_probability": (
            "" if detection.recurrent_artifact_probability is None else detection.recurrent_artifact_probability
        ),
        "recurrent_artifact_required_peak_signal": (
            ""
            if detection.recurrent_artifact_required_peak_signal is None
            else detection.recurrent_artifact_required_peak_signal
        ),
    }
    if track_id is not None:
        row["track_id"] = track_id
    if track_detection_index is not None:
        row["track_detection_index"] = track_detection_index
    return row


def group_by_frame(rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(int(float(row["frame_index"])), []).append(row)
    return grouped


def detection_key(row: dict[str, Any]) -> tuple[int, int]:
    """Return a stable per-frame detection key used by detections.csv and tracks.csv."""

    return int(float(row["frame_index"])), int(float(row["label"]))


def track_memberships(candidate_run: Path) -> tuple[dict[tuple[int, int], int], dict[int, set[tuple[int, int]]]]:
    path = candidate_run / "tracks.csv"
    if not path.is_file():
        return {}, {}
    rows = read_csv_rows(path)
    key_to_track: dict[tuple[int, int], int] = {}
    track_to_keys: dict[int, set[tuple[int, int]]] = {}
    for row in rows:
        track_id = int(float(row["track_id"]))
        key = detection_key(row)
        key_to_track[key] = track_id
        track_to_keys.setdefault(track_id, set()).add(key)
    return key_to_track, track_to_keys


def rescued_detection_keys(
    *,
    key_to_track: dict[tuple[int, int], int],
    track_to_keys: dict[int, set[tuple[int, int]]],
    confirmed_keys: set[tuple[int, int]],
    min_detections: int,
    min_confirmed: int,
    min_confirmed_fraction: float,
) -> set[tuple[int, int]]:
    rescued: set[tuple[int, int]] = set()
    for track_id, keys in track_to_keys.items():
        if len(keys) < min_detections:
            continue
        confirmed = len(keys & confirmed_keys)
        if confirmed < min_confirmed:
            continue
        if confirmed / len(keys) < min_confirmed_fraction:
            continue
        rescued.update(key for key in keys if key_to_track.get(key) == track_id)
    return rescued


def expanded_bbox(row: dict[str, str], *, margin: float) -> tuple[float, float, float, float]:
    return (
        float(row["bbox_top"]) - margin,
        float(row["bbox_left"]) - margin,
        float(row["bbox_bottom"]) + margin,
        float(row["bbox_right"]) + margin,
    )


def bbox_iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    top = max(first[0], second[0])
    left = max(first[1], second[1])
    bottom = min(first[2], second[2])
    right = min(first[3], second[3])
    intersection = max(0.0, bottom - top) * max(0.0, right - left)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return 0.0 if union <= 0.0 else intersection / union


def center_distance(first: dict[str, str], second: dict[str, str]) -> float:
    return float(np.hypot(float(first["y"]) - float(second["y"]), float(first["x"]) - float(second["x"])))


def best_confirmation(
    candidate: dict[str, str],
    confirmers: list[dict[str, str]],
    *,
    min_iou: float,
    max_center_distance_px: float,
    candidate_margin_px: float,
    confirming_margin_px: float,
) -> tuple[dict[str, str] | None, float, float]:
    candidate_bbox = expanded_bbox(candidate, margin=candidate_margin_px)
    best_row = None
    best_iou = 0.0
    best_distance = math.inf
    for confirmer in confirmers:
        iou = bbox_iou(candidate_bbox, expanded_bbox(confirmer, margin=confirming_margin_px))
        distance = center_distance(candidate, confirmer)
        if iou > best_iou or (iou == best_iou and distance < best_distance):
            best_row = confirmer
            best_iou = iou
            best_distance = distance
    if best_row is None:
        return None, 0.0, math.inf
    if best_iou >= min_iou or best_distance <= max_center_distance_px:
        return best_row, best_iou, best_distance
    return None, best_iou, best_distance


def infer_n_images(candidate_run: Path, candidate_rows: list[dict[str, str]]) -> int:
    metadata = read_json(candidate_run / "metadata.json")
    n_images = metadata.get("n_images")
    if isinstance(n_images, int) and n_images >= 0:
        return n_images
    per_frame = read_csv_rows(candidate_run / "detections_per_frame.csv")
    if per_frame:
        return max(int(float(row["frame_index"])) for row in per_frame) + 1
    if candidate_rows:
        return max(int(float(row["frame_index"])) for row in candidate_rows) + 1
    return 0


def infer_belt_velocity(args: argparse.Namespace, metadata: dict[str, Any]) -> float:
    if args.belt_velocity_px_per_frame is not None:
        return float(args.belt_velocity_px_per_frame)
    value = metadata.get("belt_velocity_px_per_frame")
    return 0.0 if value is None else float(value)


def copy_previews(candidate_run: Path, output_dir: Path) -> None:
    for pattern in PREVIEW_PATTERNS:
        for path in candidate_run.glob(pattern):
            shutil.copy2(path, output_dir / path.name)


def filter_run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    candidate_rows = read_csv_rows(args.candidate_run / "detections.csv")
    confirming_rows = read_csv_rows(args.confirming_run / "detections.csv")
    confirming_by_frame = group_by_frame(confirming_rows)
    key_to_track, track_to_keys = track_memberships(args.candidate_run)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated_rows: list[dict[str, Any]] = []
    confirmed_keys: set[tuple[int, int]] = set()
    for row in candidate_rows:
        frame_index = int(float(row["frame_index"]))
        match, iou, distance = best_confirmation(
            row,
            confirming_by_frame.get(frame_index, []),
            min_iou=args.min_iou,
            max_center_distance_px=args.max_center_distance_px,
            candidate_margin_px=args.candidate_margin_px,
            confirming_margin_px=args.confirming_margin_px,
        )
        annotated = dict(row)
        annotated["confirmation_iou"] = iou
        annotated["confirmation_center_distance_px"] = "" if not math.isfinite(distance) else distance
        annotated["confirmation_label"] = "" if match is None else match.get("label", "")
        annotated["confirmation_track_id"] = key_to_track.get(detection_key(row), "")
        annotated["confirmation_rescued_by_track"] = False
        if match is None:
            annotated_rows.append(annotated)
        else:
            confirmed_keys.add(detection_key(row))
            annotated_rows.append(annotated)

    rescued_keys: set[tuple[int, int]] = set()
    if not args.disable_track_rescue and key_to_track:
        rescued_keys = rescued_detection_keys(
            key_to_track=key_to_track,
            track_to_keys=track_to_keys,
            confirmed_keys=confirmed_keys,
            min_detections=args.track_rescue_min_detections,
            min_confirmed=args.track_rescue_min_confirmed,
            min_confirmed_fraction=args.track_rescue_min_confirmed_fraction,
        )

    kept_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    for row in annotated_rows:
        key = detection_key(row)
        if key in confirmed_keys or key in rescued_keys:
            row["confirmation_rescued_by_track"] = key not in confirmed_keys
            kept_rows.append(row)
        else:
            rejected_rows.append(row)

    n_images = infer_n_images(args.candidate_run, candidate_rows)
    image_by_frame = {
        int(float(row["frame_index"])): row.get("image", "")
        for row in candidate_rows
    }
    detections_by_frame: list[list[ParticleDetection]] = [[] for _ in range(n_images)]
    for row in kept_rows:
        frame_index = int(float(row["frame_index"]))
        if 0 <= frame_index < n_images:
            detections_by_frame[frame_index].append(parse_detection(row))

    candidate_metadata = read_json(args.candidate_run / "metadata.json")
    belt_velocity = infer_belt_velocity(args, candidate_metadata)
    tracking_config = ParticleTrackingConfig(
        max_match_distance_px=max(5.0, 1.5 * abs(belt_velocity)),
        max_frame_gap=args.tracking_max_frame_gap,
        velocity_prior_y_px_per_frame=0.8 * belt_velocity,
        assignment_method=args.tracking_assignment_method,
        area_cost_weight_px=args.tracking_area_cost_weight_px,
        signal_cost_weight_px=args.tracking_signal_cost_weight_px,
        lateral_cost_weight=args.tracking_lateral_cost_weight,
        max_area_ratio=None if args.tracking_max_area_ratio <= 0 else args.tracking_max_area_ratio,
    )
    tracks = track_particle_detections(
        detections_by_frame,
        config=tracking_config,
        frame_indices=[float(index) for index in range(n_images)],
    )
    velocity_objects = estimate_particle_velocities_vs_belt(
        tracks,
        belt_image_velocity_px_per_frame=belt_velocity,
        min_track_length=args.min_track_length,
        fit_method=args.velocity_fit_method,
    )
    track_filter = TrackFilterConfig(
        min_track_length=args.track_filter_min_length,
        min_velocity_ratio_y=args.track_filter_min_velocity_ratio_y,
        max_velocity_ratio_y=args.track_filter_max_velocity_ratio_y,
        max_abs_x_velocity_px_per_frame=(
            None
            if args.track_filter_max_abs_x_velocity_px_per_frame is None
            or args.track_filter_max_abs_x_velocity_px_per_frame <= 0
            else args.track_filter_max_abs_x_velocity_px_per_frame
        ),
    )
    track_scores = score_particle_velocities(velocity_objects, config=track_filter)
    accepted_track_ids = {score.track_id for score in track_scores if score.accepted}

    track_rows = [
        detection_row(
            detection,
            image=image_by_frame.get(int(detection.frame_index), ""),
            track_id=track.track_id,
            track_detection_index=detection_index,
        )
        for track in tracks
        for detection_index, detection in enumerate(track.detections)
    ]
    filtered_track_rows = [row for row in track_rows if row["track_id"] in accepted_track_ids]
    filtered_velocity_rows = [
        asdict(velocity)
        for velocity in velocity_objects
        if velocity.track_id in accepted_track_ids
    ]

    detection_fieldnames = list(dict.fromkeys([*DETECTION_FIELDS, *CONFIRMATION_FIELDS]))
    write_csv(output_dir / "detections.csv", kept_rows, detection_fieldnames)
    write_csv(output_dir / "rejected_detections.csv", rejected_rows, detection_fieldnames)
    write_csv(
        output_dir / "detections_per_frame.csv",
        [
            {"frame_index": index, "n_detections": len(detections)}
            for index, detections in enumerate(detections_by_frame)
        ],
        ["frame_index", "n_detections"],
    )
    write_csv(output_dir / "tracks.csv", track_rows, TRACK_DETECTION_FIELDS)
    write_csv(output_dir / "velocities.csv", [asdict(velocity) for velocity in velocity_objects], VELOCITY_FIELDS)
    write_csv(output_dir / "track_scores.csv", [asdict(score) for score in track_scores], TRACK_SCORE_FIELDS)
    write_csv(output_dir / "filtered_velocities.csv", filtered_velocity_rows, VELOCITY_FIELDS)
    write_csv(output_dir / "filtered_tracks.csv", filtered_track_rows, TRACK_DETECTION_FIELDS)
    copy_previews(args.candidate_run, output_dir)

    areas = np.asarray([float(row["area_px"]) for row in kept_rows], dtype=np.float64)
    metadata = {
        **candidate_metadata,
        "method": args.label,
        "candidate_run": str(args.candidate_run),
        "confirming_run": str(args.confirming_run),
        "confirmation": {
            "min_iou": args.min_iou,
            "max_center_distance_px": args.max_center_distance_px,
            "candidate_margin_px": args.candidate_margin_px,
            "confirming_margin_px": args.confirming_margin_px,
            "track_rescue_enabled": not args.disable_track_rescue,
            "track_rescue_min_detections": args.track_rescue_min_detections,
            "track_rescue_min_confirmed": args.track_rescue_min_confirmed,
            "track_rescue_min_confirmed_fraction": args.track_rescue_min_confirmed_fraction,
            "track_rescued_detections": sum(1 for row in kept_rows if row["confirmation_rescued_by_track"]),
            "rejected_detections": len(rejected_rows),
            "kept_fraction": None if not candidate_rows else len(kept_rows) / len(candidate_rows),
        },
        "n_images": n_images,
        "n_detections": len(kept_rows),
        "n_tracks": len(tracks),
        "n_velocity_estimates": len(velocity_objects),
        "n_filtered_velocity_estimates": len(filtered_velocity_rows),
        "belt_velocity_px_per_frame": belt_velocity,
        "detection_area_median_px": None if areas.size == 0 else float(np.median(areas)),
        "detections_per_frame": None if n_images == 0 else len(kept_rows) / n_images,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = filter_run(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if not args.quiet:
        print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
