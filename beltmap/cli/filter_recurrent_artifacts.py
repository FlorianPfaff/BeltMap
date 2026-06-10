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
from PIL import Image

from beltmap.phase import BeltMotionModel
from beltmap.recurrent_artifacts import (
    RECURRENT_ARTIFACT_MODES,
    RecurrentArtifactConfig,
    belt_revolution_indices,
    build_recurrent_artifact_map,
    score_recurrent_artifact_detections_excluding_current_revolution,
)
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
RECURRENT_ARTIFACT_DETECTION_FIELDS = [
    *DETECTION_FIELDS,
    "recurrent_artifact_rejected",
    "recurrent_artifact_protected_by_source_track",
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
    "raw_frame_*.png",
    "residual_frame_*.png",
    "residual_fixed_frame_*.png",
    "belt_map.png",
    "static_noise.png",
    "static_background.png",
    "preview_scales.json",
    "config_resolved.json",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beltmap-filter-recurrent-artifacts",
        description=(
            "Post-filter an existing BeltMap run by suppressing detections that "
            "recur at the same belt-coordinate locations across revolutions."
        ),
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-revolutions", type=int, default=2)
    parser.add_argument("--margin-px", type=int, default=2)
    parser.add_argument("--max-overlap-fraction", type=float, default=0.3)
    parser.add_argument("--min-recurrence-probability", type=float, default=0.0)
    parser.add_argument(
        "--mode",
        choices=sorted(RECURRENT_ARTIFACT_MODES),
        default="hard",
        help="hard rejects by overlap; soft keeps strong peaks; probabilistic gates by recurrence probability.",
    )
    parser.add_argument("--soft-penalty-weight", type=float, default=1.0)
    parser.add_argument(
        "--candidate-max-area-px",
        type=int,
        default=None,
        help="Only use detections up to this area when building the recurrent artifact prior.",
    )
    parser.add_argument(
        "--candidate-max-peak-signal",
        type=float,
        default=None,
        help="Only use detections up to this peak signal when building the recurrent artifact prior.",
    )
    parser.add_argument(
        "--reject-max-area-px",
        type=int,
        default=None,
        help="Only reject detections up to this area. Larger detections are scored but kept.",
    )
    parser.add_argument(
        "--reject-max-peak-signal",
        type=float,
        default=None,
        help="Only reject detections up to this peak signal. Stronger detections are scored but kept.",
    )
    parser.add_argument(
        "--velocity-fit-method",
        choices=["linear", "theil_sen"],
        default=None,
        help="Velocity fit method. Default: source run config or linear.",
    )
    parser.add_argument(
        "--min-track-length",
        type=int,
        default=None,
        help="Minimum detections for velocity estimation. Default: source run config or 2.",
    )
    parser.add_argument("--track-filter-min-length", type=int, default=None)
    parser.add_argument("--track-filter-min-velocity-ratio-y", type=float, default=None)
    parser.add_argument("--track-filter-max-velocity-ratio-y", type=float, default=None)
    parser.add_argument("--track-filter-max-abs-x-velocity-px-per-frame", type=float, default=None)
    parser.add_argument(
        "--protect-source-filtered-tracks",
        action="store_true",
        help=(
            "Keep recurrent-artifact hits that were part of an accepted source "
            "filtered track. This is useful for stricter post-run cleanup without "
            "breaking already plausible particle tracks."
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing required CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def option_value(config: dict[str, Any], name: str) -> str | None:
    options = config.get("options", {})
    if not isinstance(options, dict):
        return None
    option = options.get(name)
    if not isinstance(option, dict):
        return None
    value = option.get("value")
    return None if value is None or str(value).strip() == "" else str(value)


def parse_optional_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    return None if value is None or str(value).strip() == "" else float(value)


def finite_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or not parsed.is_integer():
        return None
    return int(parsed)


def parse_required_int(value: Any, *, name: str) -> int:
    parsed = finite_int(value)
    if parsed is None:
        raise ValueError(f"{name} must be an integer-valued field")
    return parsed


def parse_detection(row: dict[str, str]) -> ParticleDetection:
    frame_index = parse_required_int(row.get("frame_index"), name="frame_index")
    return ParticleDetection(
        frame_index=frame_index,
        label=parse_required_int(row.get("label"), name="label"),
        y=float(row["y"]),
        x=float(row["x"]),
        area_px=parse_required_int(row.get("area_px"), name="area_px"),
        bbox_top=parse_required_int(row.get("bbox_top"), name="bbox_top"),
        bbox_left=parse_required_int(row.get("bbox_left"), name="bbox_left"),
        bbox_bottom=parse_required_int(row.get("bbox_bottom"), name="bbox_bottom"),
        bbox_right=parse_required_int(row.get("bbox_right"), name="bbox_right"),
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


def detection_key(detection: ParticleDetection) -> tuple[int, int]:
    return (int(detection.frame_index), int(detection.label))


def row_detection_key(row: dict[str, Any]) -> tuple[int, int]:
    return (
        parse_required_int(row.get("frame_index"), name="frame_index"),
        parse_required_int(row.get("label"), name="label"),
    )


def read_source_filtered_track_keys(input_dir: Path) -> set[tuple[int, int]]:
    return {
        row_detection_key(row)
        for row in read_csv_rows(input_dir / "filtered_tracks.csv")
    }


def detection_to_row(
    detection: ParticleDetection,
    *,
    image_by_frame: dict[int, str],
    rejected: bool | None = None,
    protected_by_source_track: bool | None = None,
) -> dict[str, Any]:
    frame_index = int(detection.frame_index)
    row: dict[str, Any] = {
        "frame_index": frame_index,
        "image": image_by_frame.get(frame_index, ""),
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
            ""
            if detection.recurrent_artifact_overlap_fraction is None
            else detection.recurrent_artifact_overlap_fraction
        ),
        "recurrent_artifact_probability": (
            ""
            if detection.recurrent_artifact_probability is None
            else detection.recurrent_artifact_probability
        ),
        "recurrent_artifact_required_peak_signal": (
            ""
            if detection.recurrent_artifact_required_peak_signal is None
            else detection.recurrent_artifact_required_peak_signal
        ),
    }
    if rejected is not None:
        row["recurrent_artifact_rejected"] = rejected
    if protected_by_source_track is not None:
        row["recurrent_artifact_protected_by_source_track"] = protected_by_source_track
    return row


def group_detections(
    rows: list[dict[str, str]],
    *,
    frame_count: int,
) -> tuple[list[list[ParticleDetection]], dict[int, str]]:
    detections_by_frame: list[list[ParticleDetection]] = [[] for _ in range(frame_count)]
    image_by_frame: dict[int, str] = {}
    for row in rows:
        frame_index = parse_required_int(row.get("frame_index"), name="frame_index")
        if frame_index < 0 or frame_index >= frame_count:
            raise ValueError(
                f"detection frame_index {frame_index} is outside frame_count {frame_count}"
            )
        detections_by_frame[frame_index].append(parse_detection(row))
        image_by_frame.setdefault(frame_index, row.get("image", ""))
    return detections_by_frame, image_by_frame


def load_phase_px_by_frame(path: Path, *, frame_count: int) -> list[float]:
    phase_by_frame: list[float | None] = [None for _ in range(frame_count)]
    for row in read_csv_rows(path):
        frame_index = finite_int(row.get("frame_index"))
        if frame_index is not None and 0 <= frame_index < frame_count:
            phase_by_frame[frame_index] = float(row["phase_px"])
    missing = [index for index, value in enumerate(phase_by_frame) if value is None]
    if missing:
        preview = ", ".join(str(index) for index in missing[:8])
        raise ValueError(f"phase_estimates.csv is missing {len(missing)} frames; first: {preview}")
    return [float(value) for value in phase_by_frame]


def infer_frame_count(metadata: dict[str, Any], detection_rows: list[dict[str, str]]) -> int:
    metadata_count = metadata.get("n_images")
    if metadata_count is not None:
        return parse_required_int(metadata_count, name="n_images")
    max_detection_frame = max(
        (
            frame_index
            for row in detection_rows
            if (frame_index := finite_int(row.get("frame_index"))) is not None
        ),
        default=-1,
    )
    return max_detection_frame + 1


def infer_region(metadata: dict[str, Any], config: dict[str, Any]) -> tuple[int, int, int, int]:
    region = metadata.get("belt_region")
    if isinstance(region, dict):
        return (
            parse_required_int(region["top"], name="belt_region.top"),
            parse_required_int(region["left"], name="belt_region.left"),
            parse_required_int(region["height"], name="belt_region.height"),
            parse_required_int(region["width"], name="belt_region.width"),
        )
    value = option_value(config, "belt_region")
    if value:
        parts = [
            parse_required_int(part.strip(), name="belt_region")
            for part in value.split(",")
        ]
        if len(parts) == 4:
            return tuple(parts)  # type: ignore[return-value]
    raise ValueError("cannot infer belt region from metadata/config_resolved.json")


def finite_nonzero_float(value: Any, *, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or abs(parsed) <= 1e-12:
        raise ValueError(f"{name} must be finite and non-zero")
    return parsed


def optional_nonnegative_float(value: float | None, *, name: str) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return None if value == 0 else value


def int_option(config: dict[str, Any], name: str, default: int) -> int:
    value = option_value(config, name)
    return default if value is None else parse_required_int(value, name=name)


def float_option(config: dict[str, Any], name: str, default: float) -> float:
    value = option_value(config, name)
    return default if value is None else float(value)


def optional_float_option(config: dict[str, Any], name: str) -> float | None:
    value = option_value(config, name)
    if value is None:
        return None
    parsed = float(value)
    return None if parsed <= 0 else parsed


def save_scaled_png(array: np.ndarray, path: Path) -> None:
    arr = np.asarray(array, dtype=np.float64)
    finite = np.isfinite(arr)
    low, high = np.percentile(arr[finite], [1, 99]) if finite.any() else (0.0, 1.0)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = 0.0, 1.0
    image = np.clip((arr - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(path)


def copy_preview_files(input_dir: Path, output_dir: Path) -> None:
    for pattern in PREVIEW_PATTERNS:
        for source in input_dir.glob(pattern):
            target = output_dir / source.name
            if source.is_file() and source.resolve() != target.resolve():
                shutil.copy2(source, target)


def write_detection_outputs(
    output_dir: Path,
    detections_by_frame: list[list[ParticleDetection]],
    *,
    image_by_frame: dict[int, str],
) -> list[dict[str, Any]]:
    rows = [
        detection_to_row(detection, image_by_frame=image_by_frame)
        for detections in detections_by_frame
        for detection in detections
    ]
    write_csv(output_dir / "detections.csv", rows, DETECTION_FIELDS)
    write_csv(
        output_dir / "detections_per_frame.csv",
        [
            {"frame_index": index, "n_detections": len(detections)}
            for index, detections in enumerate(detections_by_frame)
        ],
        ["frame_index", "n_detections"],
    )
    return rows


def write_tracking_outputs(
    output_dir: Path,
    detections_by_frame: list[list[ParticleDetection]],
    *,
    image_by_frame: dict[int, str],
    metadata: dict[str, Any],
    config: dict[str, Any],
    min_track_length: int,
    velocity_fit_method: str,
    track_filter_config: TrackFilterConfig,
) -> tuple[int, int, int, int]:
    velocity = finite_nonzero_float(
        metadata.get("belt_velocity_px_per_frame", option_value(config, "belt_velocity_px_per_frame")),
        name="belt_velocity_px_per_frame",
    )
    max_match_text = option_value(config, "max_match_distance_px")
    max_match = (
        float(max_match_text)
        if max_match_text is not None
        else max(5.0, 1.5 * abs(velocity))
    )
    tracking_config = ParticleTrackingConfig(
        max_match_distance_px=max_match,
        max_frame_gap=float(metadata.get("tracking_max_frame_gap", 2.0)),
        velocity_prior_y_px_per_frame=0.8 * velocity,
    )
    tracks = track_particle_detections(
        detections_by_frame,
        config=tracking_config,
        frame_indices=[float(index) for index in range(len(detections_by_frame))],
    )
    track_rows = []
    for track in tracks:
        for detection_index, detection in enumerate(track.detections):
            row = detection_to_row(detection, image_by_frame=image_by_frame)
            row["track_id"] = track.track_id
            row["track_detection_index"] = detection_index
            track_rows.append(row)
    write_csv(output_dir / "tracks.csv", track_rows, TRACK_DETECTION_FIELDS)

    velocities = list(
        estimate_particle_velocities_vs_belt(
            tracks,
            belt_image_velocity_px_per_frame=velocity,
            min_track_length=min_track_length,
            fit_method=velocity_fit_method,
        )
    )
    write_csv(output_dir / "velocities.csv", [asdict(item) for item in velocities], VELOCITY_FIELDS)
    scores = score_particle_velocities(velocities, config=track_filter_config)
    accepted_track_ids = {score.track_id for score in scores if score.accepted}
    filtered_velocities = [
        velocity_row for velocity_row in velocities if velocity_row.track_id in accepted_track_ids
    ]
    filtered_track_rows = [
        row for row in track_rows if row["track_id"] in accepted_track_ids
    ]
    write_csv(output_dir / "track_scores.csv", [asdict(item) for item in scores], TRACK_SCORE_FIELDS)
    write_csv(
        output_dir / "filtered_velocities.csv",
        [asdict(item) for item in filtered_velocities],
        VELOCITY_FIELDS,
    )
    write_csv(output_dir / "filtered_tracks.csv", filtered_track_rows, TRACK_DETECTION_FIELDS)
    return len(tracks), len(velocities), len(filtered_velocities), len(filtered_track_rows)


def summarize_detection_areas(rows: list[dict[str, Any]]) -> float | None:
    values = np.asarray([float(row["area_px"]) for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    return None if values.size == 0 else float(np.median(values))


def filter_recurrent_artifacts(
    *,
    input_dir: Path,
    output_dir: Path,
    recurrent_config: RecurrentArtifactConfig,
    min_track_length: int | None,
    velocity_fit_method: str | None,
    track_filter_config: TrackFilterConfig | None,
    protect_source_filtered_tracks: bool = False,
) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if input_dir == output_dir:
        raise ValueError("--output-dir must be different from --input-dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = read_json(input_dir / "metadata.json")
    config = read_json(input_dir / "config_resolved.json")
    detection_rows = read_csv_rows(input_dir / "detections.csv")
    frame_count = infer_frame_count(metadata, detection_rows)
    region = infer_region(metadata, config)
    map_height = parse_required_int(
        metadata.get("belt_map_height_px") or metadata.get("belt_period_px_input") or region[2],
        name="belt_map_height_px",
    )
    detection_threshold = float(
        metadata.get("detection_threshold", option_value(config, "detection_threshold") or 0.0)
    )
    if detection_threshold <= 0:
        raise ValueError("cannot infer positive detection threshold from source run")
    belt_velocity = finite_nonzero_float(
        metadata.get("belt_velocity_px_per_frame", option_value(config, "belt_velocity_px_per_frame")),
        name="belt_velocity_px_per_frame",
    )
    period_px = float(metadata.get("belt_period_px_input") or map_height)
    reference_phase = float(metadata.get("reference_phase_px", 0.0))
    phase_px_by_frame = load_phase_px_by_frame(input_dir / "phase_estimates.csv", frame_count=frame_count)
    detections_by_frame, image_by_frame = group_detections(detection_rows, frame_count=frame_count)
    protected_track_keys = (
        read_source_filtered_track_keys(input_dir)
        if protect_source_filtered_tracks
        else set()
    )

    recurrent_result = build_recurrent_artifact_map(
        detections_by_frame,
        phase_px_by_frame,
        belt_revolution_indices(
            frame_count,
            BeltMotionModel(
                image_velocity_px_per_frame=belt_velocity,
                period_px=period_px,
                reference_phase_px=reference_phase,
            ),
        ),
        map_shape=(map_height, region[3]),
        config=recurrent_config,
        frame_shape=(region[2], region[3]),
    )
    revolution_by_frame = belt_revolution_indices(
        frame_count,
        BeltMotionModel(
            image_velocity_px_per_frame=belt_velocity,
            period_px=period_px,
            reference_phase_px=reference_phase,
        ),
    )
    recurrent_scores = score_recurrent_artifact_detections_excluding_current_revolution(
        detections_by_frame,
        phase_px_by_frame,
        revolution_by_frame,
        recurrent_result,
        config=recurrent_config,
        detection_threshold=detection_threshold,
        frame_shape=(region[2], region[3]),
    )
    recurrent_rows = []
    for frame_scores in recurrent_scores:
        for score in frame_scores:
            protected = score.rejected and detection_key(score.detection) in protected_track_keys
            recurrent_rows.append(
                detection_to_row(
                    score.detection,
                    image_by_frame=image_by_frame,
                    rejected=score.rejected and not protected,
                    protected_by_source_track=protected,
                )
            )
    write_csv(
        output_dir / "recurrent_artifact_detections.csv",
        recurrent_rows,
        RECURRENT_ARTIFACT_DETECTION_FIELDS,
    )
    np.save(output_dir / "recurrent_artifact_map.npy", recurrent_result.mask)
    np.save(output_dir / "recurrent_artifact_counts.npy", recurrent_result.counts)
    np.save(output_dir / "recurrent_artifact_exposure_counts.npy", recurrent_result.exposure_counts)
    np.save(output_dir / "recurrent_artifact_probability.npy", recurrent_result.probability)
    save_scaled_png(recurrent_result.mask.astype(np.float32), output_dir / "recurrent_artifact_map.png")
    save_scaled_png(recurrent_result.counts.astype(np.float32), output_dir / "recurrent_artifact_counts.png")
    save_scaled_png(recurrent_result.probability.astype(np.float32), output_dir / "recurrent_artifact_probability.png")

    filtered_by_frame = []
    for frame_scores in recurrent_scores:
        kept = []
        for score in frame_scores:
            protected = score.rejected and detection_key(score.detection) in protected_track_keys
            if not score.rejected or protected:
                kept.append(score.detection)
        filtered_by_frame.append(kept)
    filtered_rows = write_detection_outputs(
        output_dir,
        filtered_by_frame,
        image_by_frame=image_by_frame,
    )
    shutil.copy2(input_dir / "phase_estimates.csv", output_dir / "phase_estimates.csv")
    copy_preview_files(input_dir, output_dir)

    min_track_length = (
        min_track_length
        if min_track_length is not None
        else int_option(config, "min_track_length", 2)
    )
    velocity_fit_method = (
        velocity_fit_method
        or str(metadata.get("tracking_velocity_fit_method") or option_value(config, "tracking_velocity_fit_method") or "linear")
    )
    track_filter_config = track_filter_config or TrackFilterConfig(
        min_track_length=parse_required_int(
            metadata.get("track_filter_min_length", max(5, min_track_length)),
            name="track_filter_min_length",
        ),
        min_velocity_ratio_y=float(metadata.get("track_filter_min_velocity_ratio_y", 0.0)),
        max_velocity_ratio_y=float(metadata.get("track_filter_max_velocity_ratio_y", 1.1)),
        max_abs_x_velocity_px_per_frame=metadata.get("track_filter_max_abs_x_velocity_px_per_frame"),
    )
    n_tracks, n_velocities, n_filtered_velocities, n_filtered_track_rows = write_tracking_outputs(
        output_dir,
        filtered_by_frame,
        image_by_frame=image_by_frame,
        metadata=metadata,
        config=config,
        min_track_length=min_track_length,
        velocity_fit_method=velocity_fit_method,
        track_filter_config=track_filter_config,
    )

    rejected = sum(
        1
        for frame_scores in recurrent_scores
        for score in frame_scores
        if score.rejected and detection_key(score.detection) not in protected_track_keys
    )
    protected = sum(
        1
        for frame_scores in recurrent_scores
        for score in frame_scores
        if score.rejected and detection_key(score.detection) in protected_track_keys
    )
    payload = {
        **metadata,
        "method": "postrun_recurrent_artifact_filter",
        "source_run": str(input_dir),
        "n_images": frame_count,
        "n_source_detections": len(detection_rows),
        "n_detections": len(filtered_rows),
        "n_recurrent_artifact_rejected": rejected,
        "n_recurrent_artifact_protected_by_source_track": protected,
        "n_tracks": n_tracks,
        "n_velocity_estimates": n_velocities,
        "n_filtered_velocity_estimates": n_filtered_velocities,
        "n_filtered_track_detection_rows": n_filtered_track_rows,
        "detection_area_median_px": summarize_detection_areas(filtered_rows),
        "detections_per_frame": len(filtered_rows) / frame_count if frame_count else 0.0,
        "recurrent_artifact_filter_used": True,
        "recurrent_artifact_source": "postrun_built",
        "recurrent_artifact_min_revolutions": recurrent_config.min_revolutions,
        "recurrent_artifact_margin_px": recurrent_config.margin_px,
        "recurrent_artifact_max_overlap_fraction": recurrent_config.max_overlap_fraction,
        "recurrent_artifact_min_recurrence_probability": (
            recurrent_config.min_recurrence_probability
        ),
        "recurrent_artifact_mode": recurrent_config.mode,
        "recurrent_artifact_soft_penalty_weight": recurrent_config.soft_penalty_weight,
        "recurrent_artifact_candidate_max_area_px": recurrent_config.candidate_max_area_px,
        "recurrent_artifact_candidate_max_peak_signal": recurrent_config.candidate_max_peak_signal,
        "recurrent_artifact_reject_max_area_px": recurrent_config.reject_max_area_px,
        "recurrent_artifact_reject_max_peak_signal": recurrent_config.reject_max_peak_signal,
        "recurrent_artifact_protect_source_filtered_tracks": protect_source_filtered_tracks,
        "recurrent_artifact_revolutions": recurrent_result.revolution_count,
        "recurrent_artifact_pixels": recurrent_result.artifact_pixels,
        "track_filter_min_length": track_filter_config.min_track_length,
        "track_filter_min_velocity_ratio_y": track_filter_config.min_velocity_ratio_y,
        "track_filter_max_velocity_ratio_y": track_filter_config.max_velocity_ratio_y,
        "track_filter_max_abs_x_velocity_px_per_frame": (
            track_filter_config.max_abs_x_velocity_px_per_frame
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.min_revolutions < 1:
            raise ValueError("--min-revolutions must be at least 1")
        recurrent_config = RecurrentArtifactConfig(
            min_revolutions=args.min_revolutions,
            margin_px=args.margin_px,
            max_overlap_fraction=args.max_overlap_fraction,
            mode=args.mode,
            soft_penalty_weight=args.soft_penalty_weight,
            min_recurrence_probability=args.min_recurrence_probability,
            candidate_max_area_px=args.candidate_max_area_px,
            candidate_max_peak_signal=args.candidate_max_peak_signal,
            reject_max_area_px=args.reject_max_area_px,
            reject_max_peak_signal=args.reject_max_peak_signal,
        )
        max_abs_x_velocity = optional_nonnegative_float(
            args.track_filter_max_abs_x_velocity_px_per_frame,
            name="--track-filter-max-abs-x-velocity-px-per-frame",
        )
        track_filter_config = None
        if (
            args.track_filter_min_length is not None
            or args.track_filter_min_velocity_ratio_y is not None
            or args.track_filter_max_velocity_ratio_y is not None
            or max_abs_x_velocity is not None
        ):
            track_filter_config = TrackFilterConfig(
                min_track_length=args.track_filter_min_length or max(5, args.min_track_length or 2),
                min_velocity_ratio_y=(
                    0.0
                    if args.track_filter_min_velocity_ratio_y is None
                    else args.track_filter_min_velocity_ratio_y
                ),
                max_velocity_ratio_y=(
                    1.1
                    if args.track_filter_max_velocity_ratio_y is None
                    else args.track_filter_max_velocity_ratio_y
                ),
                max_abs_x_velocity_px_per_frame=max_abs_x_velocity,
            )
        payload = filter_recurrent_artifacts(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            recurrent_config=recurrent_config,
            min_track_length=args.min_track_length,
            velocity_fit_method=args.velocity_fit_method,
            track_filter_config=track_filter_config,
            protect_source_filtered_tracks=args.protect_source_filtered_tracks,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if not args.quiet:
        print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
