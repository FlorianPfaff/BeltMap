from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from beltmap import (
    ParticleComponentConfig,
    ParticleDetection,
    ParticleTrackingConfig,
    ResidualConfig,
    ResidualImage,
    TrackFilterConfig,
    detect_particles_from_residual,
    estimate_particle_velocities_vs_belt,
    extract_particle_detections,
    generate_residual_image,
    score_particle_velocities,
    track_particle_detections,
)
from beltmap.compare_runs import RunSpec, generate_comparison_report


EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

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
    "phase_estimates.csv",
    "preview_scales.json",
    "belt_map.png",
    "raw_frame_*.png",
    "residual_frame_*.png",
    "residual_fixed_frame_*.png",
)


def natural_key(path: Path) -> list[Any]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(path))
    ]


def list_images(image_dir: Path) -> list[Path]:
    paths = sorted(
        [
            path
            for path in image_dir.rglob("*")
            if path.suffix.lower() in EXTENSIONS and not path.name.startswith("._")
        ],
        key=natural_key,
    )
    if not paths:
        raise SystemExit(f"No image files found below {image_dir}")
    return paths


def parse_region(value: str) -> tuple[int, int, int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("region must be top,left,height,width")
    top, left, height, width = parts
    if top < 0 or left < 0 or height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("region values must be non-negative with positive height/width")
    return top, left, height, width


def crop(frame: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
    top, left, height, width = region
    return frame[top : top + height, left : left + width]


def read_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32)


def sample_indices(count: int, sample_count: int) -> list[int]:
    sample_count = max(1, min(count, sample_count))
    return sorted(set(int(index) for index in np.linspace(0, count - 1, sample_count)))


def learn_average_background(
    paths: list[Path],
    *,
    region: tuple[int, int, int, int],
    sample_frames: int,
) -> np.ndarray:
    samples = sample_indices(len(paths), sample_frames)
    total: np.ndarray | None = None
    for number, index in enumerate(samples, start=1):
        frame = crop(read_gray(paths[index]), region).astype(np.float64, copy=False)
        total = frame.copy() if total is None else total + frame
        if number == 1 or number == len(samples) or number % 25 == 0:
            print(f"average_background: sampled {number}/{len(samples)} frame={index}", flush=True)
    if total is None:
        raise RuntimeError("No frames were sampled for the average background")
    return (total / len(samples)).astype(np.float32)


def display_values(array: np.ndarray | ResidualImage) -> np.ndarray:
    values = array.normalized if isinstance(array, ResidualImage) else array
    return np.asarray(values, dtype=np.float64)


def robust_display_scale(
    arrays: list[np.ndarray | ResidualImage],
    *,
    percentiles: tuple[float, float] = (1.0, 99.0),
) -> tuple[float, float]:
    values = []
    for array in arrays:
        arr = display_values(array)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            values.append(finite.ravel())
    if not values:
        return 0.0, 1.0
    joined = np.concatenate(values)
    low, high = np.percentile(joined, percentiles)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return 0.0, 1.0
    return float(low), float(high)


def save_scaled_png(
    array: np.ndarray | ResidualImage,
    path: Path,
    *,
    scale: tuple[float, float],
) -> None:
    arr = display_values(array)
    low, high = scale
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = 0.0, 1.0
    image = np.clip((arr - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def save_png(array: np.ndarray | ResidualImage, path: Path) -> None:
    arr = display_values(array)
    finite = np.isfinite(arr)
    low, high = np.percentile(arr[finite], [1, 99]) if finite.any() else (0.0, 1.0)
    save_scaled_png(array, path, scale=(float(low), float(high)))


def save_preview_set(
    arrays_by_frame: dict[int, np.ndarray | ResidualImage],
    output_dir: Path,
    *,
    prefix: str,
    scale: tuple[float, float] | None = None,
) -> tuple[float, float]:
    display_scale = scale or robust_display_scale(list(arrays_by_frame.values()))
    for frame_index, array in sorted(arrays_by_frame.items()):
        save_scaled_png(array, output_dir / f"{prefix}_frame_{frame_index:06d}.png", scale=display_scale)
    return display_scale


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def optional_csv_value(value: Any) -> Any:
    return "" if value is None else value


def detection_row(detection: Any, *, path: Path, image_dir: Path, frame_index: int) -> dict[str, Any]:
    row = {
        "frame_index": frame_index,
        "image": str(path.relative_to(image_dir)),
        "label": detection.label,
        "y": detection.y,
        "x": detection.x,
        "area_px": detection.area_px,
        "bbox_top": detection.bbox_top,
        "bbox_left": detection.bbox_left,
        "bbox_bottom": detection.bbox_bottom,
        "bbox_right": detection.bbox_right,
        "mean_signal": optional_csv_value(getattr(detection, "mean_signal", None)),
        "peak_signal": optional_csv_value(getattr(detection, "peak_signal", None)),
        "recurrent_artifact_overlap_fraction": optional_csv_value(
            getattr(detection, "recurrent_artifact_overlap_fraction", None)
        ),
        "recurrent_artifact_probability": optional_csv_value(
            getattr(detection, "recurrent_artifact_probability", None)
        ),
        "recurrent_artifact_required_peak_signal": optional_csv_value(
            getattr(detection, "recurrent_artifact_required_peak_signal", None)
        ),
    }
    return row


def track_detection_rows(tracks: list[Any], paths: list[Path], *, image_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for track in tracks:
        for detection_index, detection in enumerate(track.detections):
            frame_index = int(detection.frame_index)
            row = detection_row(
                detection,
                path=paths[frame_index],
                image_dir=image_dir,
                frame_index=frame_index,
            )
            row["track_id"] = track.track_id
            row["track_detection_index"] = detection_index
            rows.append(row)
    return rows


def detections_to_rows(
    detections_by_frame: list[list[Any]],
    paths: list[Path],
    *,
    image_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame_index, detections in enumerate(detections_by_frame):
        rows.extend(
            detection_row(
                detection,
                path=paths[frame_index],
                image_dir=image_dir,
                frame_index=frame_index,
            )
            for detection in detections
        )
    return rows


def raw_zscore_residual(frame: np.ndarray, _: np.ndarray | None, config: ResidualConfig) -> ResidualImage:
    center = float(np.median(frame[np.isfinite(frame)]))
    expected = np.full(frame.shape, center, dtype=np.float32)
    return generate_residual_image(frame, expected, config=config)


def average_subtracted_residual(
    frame: np.ndarray,
    average_background: np.ndarray | None,
    config: ResidualConfig,
) -> ResidualImage:
    if average_background is None:
        raise ValueError("average_background is required")
    return generate_residual_image(frame, average_background, config=config)


def read_csv_rows(path: Path, *, required: bool = True) -> list[dict[str, str]]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def optional_float(row: dict[str, Any], field: str) -> float | None:
    value = row.get(field, "")
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "nan"}:
        return None
    parsed = float(text)
    return parsed if np.isfinite(parsed) else None


def required_float(row: dict[str, Any], field: str) -> float:
    parsed = optional_float(row, field)
    if parsed is None:
        raise ValueError(f"Missing finite {field!r} in detection row: {row}")
    return parsed


def required_int(row: dict[str, Any], field: str) -> int:
    parsed = required_float(row, field)
    rounded = round(parsed)
    if abs(parsed - rounded) > 1e-6:
        raise ValueError(f"Expected integer-like {field!r}, got {parsed!r}")
    return int(rounded)


def parse_detection(row: dict[str, str]) -> ParticleDetection:
    return ParticleDetection(
        frame_index=required_float(row, "frame_index"),
        label=required_int(row, "label"),
        y=required_float(row, "y"),
        x=required_float(row, "x"),
        area_px=required_int(row, "area_px"),
        bbox_top=required_int(row, "bbox_top"),
        bbox_left=required_int(row, "bbox_left"),
        bbox_bottom=required_int(row, "bbox_bottom"),
        bbox_right=required_int(row, "bbox_right"),
        mean_signal=optional_float(row, "mean_signal"),
        peak_signal=optional_float(row, "peak_signal"),
        recurrent_artifact_overlap_fraction=optional_float(row, "recurrent_artifact_overlap_fraction"),
        recurrent_artifact_probability=optional_float(row, "recurrent_artifact_probability"),
        recurrent_artifact_required_peak_signal=optional_float(row, "recurrent_artifact_required_peak_signal"),
    )


def safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    return label or "beltmap"


def unique_label(label: str, existing: set[str]) -> str:
    if label not in existing:
        return label
    suffix = 2
    while f"{label}_{suffix}" in existing:
        suffix += 1
    return f"{label}_{suffix}"


def selected_image_name(path: Path, *, image_dir: Path) -> str:
    try:
        return str(path.relative_to(image_dir))
    except ValueError:
        return str(path)


def infer_run_frame_count(run_dir: Path, detection_rows: list[dict[str, str]], metadata: dict[str, Any]) -> int | None:
    value = metadata.get("n_images")
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, float) and value >= 0 and value.is_integer():
        return int(value)
    per_frame = read_csv_rows(run_dir / "detections_per_frame.csv", required=False)
    if per_frame:
        return max(required_int(row, "frame_index") for row in per_frame) + 1
    if detection_rows:
        return max(required_int(row, "frame_index") for row in detection_rows) + 1
    return None


def load_existing_beltmap_detections(
    run_dir: Path,
    *,
    paths: list[Path],
    image_dir: Path,
    current_frame_stride: int,
    strict_frame_match: bool,
) -> tuple[list[list[ParticleDetection]], dict[str, Any]]:
    detection_path = run_dir / "detections.csv"
    detection_rows = read_csv_rows(detection_path)
    metadata = read_json(run_dir / "metadata.json")
    source_n_images = infer_run_frame_count(run_dir, detection_rows, metadata)

    source_stride = metadata.get("frame_stride")
    if strict_frame_match and source_stride is not None:
        source_stride_int = required_int({"frame_stride": source_stride}, "frame_stride")
        if source_stride_int != current_frame_stride:
            raise ValueError(
                f"{run_dir} was produced with frame_stride={source_stride_int}, "
                f"but this comparison selected frame_stride={current_frame_stride}. "
                "Re-run the BeltMap output with the same frame selection or pass "
                "--allow-beltmap-frame-mismatch for an explicitly non-strict diagnostic run."
            )
    if strict_frame_match and source_n_images is not None and source_n_images < len(paths):
        raise ValueError(
            f"{run_dir} contains {source_n_images} processed frames, "
            f"but this comparison selected {len(paths)} frames."
        )

    detections_by_frame: list[list[ParticleDetection]] = [[] for _ in paths]
    ignored_outside_selected_frames = 0
    for row in detection_rows:
        frame_index = required_int(row, "frame_index")
        if frame_index < 0:
            raise ValueError(f"Negative frame_index in {detection_path}: {frame_index}")
        if frame_index >= len(paths):
            ignored_outside_selected_frames += 1
            continue
        observed_image = str(row.get("image", "")).strip()
        if strict_frame_match and observed_image:
            expected_image = selected_image_name(paths[frame_index], image_dir=image_dir)
            if observed_image != expected_image:
                raise ValueError(
                    f"{run_dir} frame {frame_index} image mismatch: "
                    f"detections.csv has {observed_image!r}, selected frame is {expected_image!r}. "
                    "Use the same image-dir/max-frames/frame-stride for a fair comparison."
                )
        detections_by_frame[frame_index].append(parse_detection(row))

    return detections_by_frame, {
        "source_run_n_images": source_n_images,
        "ignored_source_detections_outside_selected_frames": ignored_outside_selected_frames,
        "strict_beltmap_frame_match": strict_frame_match,
    }


def copy_preview_files(source_run: Path, method_dir: Path) -> None:
    for pattern in PREVIEW_PATTERNS:
        for path in source_run.glob(pattern):
            shutil.copy2(path, method_dir / path.name)


def run_existing_beltmap_same_tracker(
    *,
    label: str,
    source_run: Path,
    paths: list[Path],
    image_dir: Path,
    output_dir: Path,
    current_frame_stride: int,
    strict_frame_match: bool,
    tracking_config: ParticleTrackingConfig,
    velocity_fit_method: str,
    belt_velocity_px_per_frame: float,
    min_track_length: int,
    track_filter_config: TrackFilterConfig,
) -> dict[str, Any]:
    start = time.perf_counter()
    method_dir = output_dir / label
    method_dir.mkdir(parents=True, exist_ok=True)
    copy_preview_files(source_run, method_dir)

    source_metadata = read_json(source_run / "metadata.json")
    detections_by_frame, load_metadata = load_existing_beltmap_detections(
        source_run,
        paths=paths,
        image_dir=image_dir,
        current_frame_stride=current_frame_stride,
        strict_frame_match=strict_frame_match,
    )
    detection_rows = detections_to_rows(detections_by_frame, paths, image_dir=image_dir)
    print(
        f"{label}: loaded {len(detection_rows)} BeltMap detections from {source_run}; "
        "re-tracking with shared PyRecEst settings",
        flush=True,
    )
    write_csv(method_dir / "detections.csv", detection_rows, DETECTION_FIELDS)
    write_csv(
        method_dir / "detections_per_frame.csv",
        [
            {"frame_index": index, "n_detections": len(detections)}
            for index, detections in enumerate(detections_by_frame)
        ],
        ["frame_index", "n_detections"],
    )

    tracks = track_particle_detections(
        detections_by_frame,
        config=tracking_config,
        frame_indices=[float(index) for index in range(len(paths))],
    )
    tracks_rows = track_detection_rows(tracks, paths, image_dir=image_dir)
    write_csv(method_dir / "tracks.csv", tracks_rows, TRACK_DETECTION_FIELDS)

    velocity_objects = estimate_particle_velocities_vs_belt(
        tracks,
        belt_image_velocity_px_per_frame=belt_velocity_px_per_frame,
        min_track_length=min_track_length,
        fit_method=velocity_fit_method,
    )
    velocity_rows = [asdict(velocity) for velocity in velocity_objects]
    write_csv(method_dir / "velocities.csv", velocity_rows, VELOCITY_FIELDS)

    track_scores = score_particle_velocities(velocity_objects, config=track_filter_config)
    accepted_track_ids = {score.track_id for score in track_scores if score.accepted}
    filtered_velocity_rows = [
        asdict(velocity)
        for velocity in velocity_objects
        if velocity.track_id in accepted_track_ids
    ]
    filtered_track_rows = [row for row in tracks_rows if row["track_id"] in accepted_track_ids]
    write_csv(method_dir / "track_scores.csv", [asdict(score) for score in track_scores], TRACK_SCORE_FIELDS)
    write_csv(method_dir / "filtered_velocities.csv", filtered_velocity_rows, VELOCITY_FIELDS)
    write_csv(method_dir / "filtered_tracks.csv", filtered_track_rows, TRACK_DETECTION_FIELDS)

    areas = np.asarray([row["area_px"] for row in detection_rows], dtype=np.float64)
    elapsed_s = time.perf_counter() - start
    metadata = {
        "method": label,
        "source_run": str(source_run),
        "same_tracker_recomputed": True,
        "n_images": len(paths),
        "n_detections": len(detection_rows),
        "n_tracks": len(tracks),
        "n_velocity_estimates": len(velocity_rows),
        "n_filtered_velocity_estimates": len(filtered_velocity_rows),
        "belt_velocity_px_per_frame": belt_velocity_px_per_frame,
        "detection_threshold": source_metadata.get("detection_threshold"),
        "detection_low_threshold": source_metadata.get("detection_low_threshold"),
        "detection_mode": source_metadata.get("detection_mode"),
        "min_area_px": source_metadata.get("min_area_px"),
        "detection_area_median_px": None if areas.size == 0 else float(np.median(areas)),
        "detections_per_frame": None if not paths else len(detection_rows) / len(paths),
        "elapsed_s": elapsed_s,
        **load_metadata,
    }
    (method_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"label": label, "output_dir": str(method_dir), **metadata}


def run_method(
    *,
    label: str,
    residual_factory: Callable[[np.ndarray, np.ndarray | None, ResidualConfig], ResidualImage],
    average_background: np.ndarray | None,
    paths: list[Path],
    image_dir: Path,
    output_dir: Path,
    region: tuple[int, int, int, int],
    threshold: float,
    low_threshold: float | None,
    mode: str,
    component_config: ParticleComponentConfig,
    residual_config: ResidualConfig,
    tracking_config: ParticleTrackingConfig,
    velocity_fit_method: str,
    belt_velocity_px_per_frame: float,
    min_track_length: int,
    track_filter_config: TrackFilterConfig,
    preview_frames: set[int],
    residual_preview_range: tuple[float, float],
    progress_interval: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    method_dir = output_dir / label
    method_dir.mkdir(parents=True, exist_ok=True)
    detections_by_frame: list[list[Any]] = []
    raw_previews: dict[int, np.ndarray] = {}
    residual_previews: dict[int, np.ndarray] = {}
    for frame_index, path in enumerate(paths):
        frame = crop(read_gray(path), region)
        residual = residual_factory(frame, average_background, residual_config)
        if frame_index in preview_frames:
            raw_previews[frame_index] = frame.copy()
            residual_previews[frame_index] = np.asarray(residual.normalized, dtype=np.float32)
            save_png(residual, method_dir / f"residual_frame_{frame_index:06d}.png")
        mask = detect_particles_from_residual(
            residual,
            threshold=threshold,
            mode=mode,
            low_threshold=low_threshold,
        )
        detections = extract_particle_detections(
            mask,
            residual=residual,
            frame_index=float(frame_index),
            config=component_config,
            signal_mode=mode,
        )
        detections_by_frame.append(detections)
        processed = frame_index + 1
        if processed == 1 or processed == len(paths) or processed % progress_interval == 0:
            print(
                f"{label}: processed {processed}/{len(paths)} "
                f"detections={sum(len(items) for items in detections_by_frame)}",
                flush=True,
            )

    preview_scales = {}
    if raw_previews:
        raw_scale = save_preview_set(raw_previews, method_dir, prefix="raw")
        fixed_residual_scale = save_preview_set(
            residual_previews,
            method_dir,
            prefix="residual_fixed",
            scale=residual_preview_range,
        )
        preview_scales = {
            "raw": {"low": raw_scale[0], "high": raw_scale[1], "mode": "shared_preview_percentile_1_99"},
            "residual_fixed": {
                "low": fixed_residual_scale[0],
                "high": fixed_residual_scale[1],
                "mode": "fixed_normalized_residual",
            },
        }
        (method_dir / "preview_scales.json").write_text(
            json.dumps(preview_scales, indent=2),
            encoding="utf-8",
        )

    detection_rows = detections_to_rows(detections_by_frame, paths, image_dir=image_dir)
    write_csv(method_dir / "detections.csv", detection_rows, DETECTION_FIELDS)
    write_csv(
        method_dir / "detections_per_frame.csv",
        [
            {"frame_index": index, "n_detections": len(detections)}
            for index, detections in enumerate(detections_by_frame)
        ],
        ["frame_index", "n_detections"],
    )

    tracks = track_particle_detections(
        detections_by_frame,
        config=tracking_config,
        frame_indices=[float(index) for index in range(len(paths))],
    )
    tracks_rows = track_detection_rows(tracks, paths, image_dir=image_dir)
    write_csv(method_dir / "tracks.csv", tracks_rows, TRACK_DETECTION_FIELDS)

    velocity_objects = estimate_particle_velocities_vs_belt(
        tracks,
        belt_image_velocity_px_per_frame=belt_velocity_px_per_frame,
        min_track_length=min_track_length,
        fit_method=velocity_fit_method,
    )
    velocity_rows = [asdict(velocity) for velocity in velocity_objects]
    write_csv(method_dir / "velocities.csv", velocity_rows, VELOCITY_FIELDS)

    track_scores = score_particle_velocities(velocity_objects, config=track_filter_config)
    accepted_track_ids = {score.track_id for score in track_scores if score.accepted}
    filtered_velocity_rows = [
        asdict(velocity)
        for velocity in velocity_objects
        if velocity.track_id in accepted_track_ids
    ]
    filtered_track_rows = [
        row for row in tracks_rows if row["track_id"] in accepted_track_ids
    ]
    write_csv(method_dir / "track_scores.csv", [asdict(score) for score in track_scores], TRACK_SCORE_FIELDS)
    write_csv(method_dir / "filtered_velocities.csv", filtered_velocity_rows, VELOCITY_FIELDS)
    write_csv(method_dir / "filtered_tracks.csv", filtered_track_rows, TRACK_DETECTION_FIELDS)

    areas = np.asarray([row["area_px"] for row in detection_rows], dtype=np.float64)
    elapsed_s = time.perf_counter() - start
    metadata = {
        "method": label,
        "source_run": "",
        "same_tracker_recomputed": True,
        "n_images": len(paths),
        "n_detections": len(detection_rows),
        "n_tracks": len(tracks),
        "n_velocity_estimates": len(velocity_rows),
        "n_filtered_velocity_estimates": len(filtered_velocity_rows),
        "belt_velocity_px_per_frame": belt_velocity_px_per_frame,
        "detection_threshold": threshold,
        "detection_low_threshold": low_threshold,
        "detection_mode": mode,
        "min_area_px": component_config.min_area_px,
        "detection_area_median_px": None if areas.size == 0 else float(np.median(areas)),
        "detections_per_frame": None if not paths else len(detection_rows) / len(paths),
        "preview_scales": preview_scales,
        "elapsed_s": elapsed_s,
    }
    (method_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"label": label, "output_dir": str(method_dir), **metadata}


def parse_optional_float(value: str) -> float | None:
    parsed = float(value)
    if parsed <= 0:
        return None
    return parsed


def parse_range(value: str) -> tuple[float, float]:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("range must be low,high")
    low, high = parts
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise argparse.ArgumentTypeError("range must contain finite values with high > low")
    return low, high


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare raw-image and raw-minus-average baselines with the BeltMap detector/tracker stack."
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/raw_baseline_comparison"))
    parser.add_argument(
        "--beltmap-run",
        action="append",
        type=Path,
        default=[],
        help=(
            "Existing BeltMap output dir whose detections.csv should be included in the "
            "fair comparison. Detections are re-tracked and re-filtered with the same "
            "PyRecEst settings used for the raw baselines."
        ),
    )
    parser.add_argument(
        "--include-original-beltmap-runs",
        action="store_true",
        help="Also include the unmodified BeltMap run dirs in comparison_report for debugging only.",
    )
    parser.add_argument(
        "--allow-beltmap-frame-mismatch",
        action="store_true",
        help=(
            "Allow BeltMap detections from a different frame selection. By default the script "
            "requires matching processed image names/stride and truncates only extra trailing frames."
        ),
    )
    parser.add_argument("--max-frames", type=int, default=1000)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--belt-region", type=parse_region, default=parse_region("0,220,1330,1800"))
    parser.add_argument("--belt-velocity-px-per-frame", type=float, default=59.16)
    parser.add_argument("--average-sample-frames", type=int, default=400)
    parser.add_argument("--average-source", choices=("all", "selected"), default="all")
    parser.add_argument("--method", action="append", choices=("raw_zscore", "raw_minus_average"), default=None)
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--low-threshold", type=parse_optional_float, default=None)
    parser.add_argument("--mode", choices=("positive", "negative", "absolute"), default="positive")
    parser.add_argument("--min-area-px", type=int, default=4)
    parser.add_argument("--max-area-px", type=parse_optional_float, default=None)
    parser.add_argument("--min-bbox-width-px", type=int, default=3)
    parser.add_argument("--min-bbox-height-px", type=int, default=3)
    parser.add_argument("--max-bbox-aspect-ratio", type=float, default=4.0)
    parser.add_argument("--min-bbox-extent", type=float, default=0.15)
    parser.add_argument("--split-merged-components", action="store_true")
    parser.add_argument("--split-min-projection-gap-px", type=int, default=1)
    parser.add_argument("--split-min-component-area-px", type=int, default=4)
    parser.add_argument("--min-track-length", type=int, default=2)
    parser.add_argument("--tracking-max-frame-gap", type=float, default=2.0)
    parser.add_argument("--velocity-fit-method", choices=("linear", "theil_sen"), default="theil_sen")
    parser.add_argument("--track-filter-min-length", type=int, default=5)
    parser.add_argument("--track-filter-min-velocity-ratio-y", type=float, default=0.0)
    parser.add_argument("--track-filter-max-velocity-ratio-y", type=float, default=1.1)
    parser.add_argument("--track-filter-max-abs-x-velocity-px-per-frame", type=parse_optional_float, default=None)
    parser.add_argument("--preview-frames", default="0,248,496,744,992")
    parser.add_argument(
        "--residual-preview-range",
        type=parse_range,
        default=parse_range("-3,8"),
        help="Fixed low,high display range for residual_fixed_frame previews.",
    )
    parser.add_argument("--progress-interval", type=int, default=25)
    return parser


def parse_preview_frames(value: str, *, frame_count: int) -> set[int]:
    frames = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        frame = int(part)
        if 0 <= frame < frame_count:
            frames.add(frame)
    return frames


def write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    fields = [
        "label",
        "source_run",
        "same_tracker_recomputed",
        "n_images",
        "n_detections",
        "detections_per_frame",
        "n_tracks",
        "n_velocity_estimates",
        "n_filtered_velocity_estimates",
        "detection_area_median_px",
        "elapsed_s",
        "output_dir",
    ]
    write_csv(output_dir / "raw_baseline_summary.csv", rows, fields)
    lines = [
        "# Same-Tracker Baseline Comparison",
        "",
        "Raw-image, raw-minus-average, and external BeltMap detections in this table use "
        "the same PyRecEst tracking and track-filter configuration.",
        "",
        "| method | detections | detections/frame | tracks | velocities | filtered velocities | median area px | elapsed s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {label} | {n_detections} | {dpf:.3g} | {n_tracks} | {n_velocity_estimates} | "
            "{n_filtered_velocity_estimates} | {area} | {elapsed:.1f} |".format(
                label=row["label"],
                n_detections=row["n_detections"],
                dpf=row["detections_per_frame"] or math.nan,
                n_tracks=row["n_tracks"],
                n_velocity_estimates=row["n_velocity_estimates"],
                n_filtered_velocity_estimates=row["n_filtered_velocity_estimates"],
                area="" if row["detection_area_median_px"] is None else f"{row['detection_area_median_px']:.3g}",
                elapsed=row["elapsed_s"],
            )
        )
    (output_dir / "raw_baseline_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_frames < 0:
        raise SystemExit("--max-frames must be non-negative")
    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be positive")
    if args.average_sample_frames < 1:
        raise SystemExit("--average-sample-frames must be positive")
    if args.progress_interval < 1:
        raise SystemExit("--progress-interval must be positive")

    all_paths = list_images(args.image_dir)
    selected_paths = all_paths[:: args.frame_stride]
    if args.max_frames > 0:
        selected_paths = selected_paths[: args.max_frames]
    if not selected_paths:
        raise SystemExit("No frames selected")
    average_paths = all_paths if args.average_source == "all" else selected_paths

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "image_dir": str(args.image_dir),
        "output_dir": str(args.output_dir),
        "discovered_images": len(all_paths),
        "selected_images": len(selected_paths),
        "average_source_images": len(average_paths),
        "arguments": vars(args) | {
            "image_dir": str(args.image_dir),
            "output_dir": str(args.output_dir),
            "belt_region": list(args.belt_region),
            "beltmap_run": [str(path) for path in args.beltmap_run],
        },
    }
    (args.output_dir / "raw_baseline_config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    methods = args.method or ["raw_zscore", "raw_minus_average"]
    average_background = None
    if "raw_minus_average" in methods:
        average_background = learn_average_background(
            average_paths,
            region=args.belt_region,
            sample_frames=args.average_sample_frames,
        )
        np.save(args.output_dir / "sampled_average_background.npy", average_background)
        save_png(average_background, args.output_dir / "sampled_average_background.png")

    component_config = ParticleComponentConfig(
        min_area_px=args.min_area_px,
        max_area_px=None if args.max_area_px is None else int(args.max_area_px),
        min_bbox_width_px=args.min_bbox_width_px,
        min_bbox_height_px=args.min_bbox_height_px,
        max_bbox_aspect_ratio=args.max_bbox_aspect_ratio,
        min_bbox_extent=args.min_bbox_extent,
        split_merged_components=args.split_merged_components,
        split_min_projection_gap_px=args.split_min_projection_gap_px,
        split_min_component_area_px=args.split_min_component_area_px,
    )
    residual_config = ResidualConfig()
    tracking_config = ParticleTrackingConfig(
        max_match_distance_px=max(5.0, 1.5 * abs(args.belt_velocity_px_per_frame)),
        max_frame_gap=args.tracking_max_frame_gap,
        velocity_prior_y_px_per_frame=0.8 * args.belt_velocity_px_per_frame,
    )
    track_filter_config = TrackFilterConfig(
        min_track_length=args.track_filter_min_length,
        min_velocity_ratio_y=args.track_filter_min_velocity_ratio_y,
        max_velocity_ratio_y=args.track_filter_max_velocity_ratio_y,
        max_abs_x_velocity_px_per_frame=args.track_filter_max_abs_x_velocity_px_per_frame,
    )
    preview_frames = parse_preview_frames(args.preview_frames, frame_count=len(selected_paths))

    factories = {
        "raw_zscore": raw_zscore_residual,
        "raw_minus_average": average_subtracted_residual,
    }
    summary_rows = []
    for method in methods:
        summary_rows.append(
            run_method(
                label=method,
                residual_factory=factories[method],
                average_background=average_background,
                paths=selected_paths,
                image_dir=args.image_dir,
                output_dir=args.output_dir,
                region=args.belt_region,
                threshold=args.threshold,
                low_threshold=args.low_threshold,
                mode=args.mode,
                component_config=component_config,
                residual_config=residual_config,
                tracking_config=tracking_config,
                velocity_fit_method=args.velocity_fit_method,
                belt_velocity_px_per_frame=args.belt_velocity_px_per_frame,
                min_track_length=args.min_track_length,
                track_filter_config=track_filter_config,
                preview_frames=preview_frames,
                residual_preview_range=args.residual_preview_range,
                progress_interval=args.progress_interval,
            )
        )

    for path in args.beltmap_run:
        base_label = (
            "beltmap_same_tracker"
            if len(args.beltmap_run) == 1
            else f"beltmap_{safe_label(path.name)}_same_tracker"
        )
        label = unique_label(base_label, {str(row["label"]) for row in summary_rows})
        summary_rows.append(
            run_existing_beltmap_same_tracker(
                label=label,
                source_run=path,
                paths=selected_paths,
                image_dir=args.image_dir,
                output_dir=args.output_dir,
                current_frame_stride=args.frame_stride,
                strict_frame_match=not args.allow_beltmap_frame_mismatch,
                tracking_config=tracking_config,
                velocity_fit_method=args.velocity_fit_method,
                belt_velocity_px_per_frame=args.belt_velocity_px_per_frame,
                min_track_length=args.min_track_length,
                track_filter_config=track_filter_config,
            )
        )

    write_summary(summary_rows, args.output_dir)
    compare_runs = [
        RunSpec(row["label"], Path(row["output_dir"]))
        for row in summary_rows
    ]
    if args.include_original_beltmap_runs:
        for path in args.beltmap_run:
            compare_runs.append(RunSpec(f"{path.name}_original", path))
    if len(compare_runs) >= 2:
        generate_comparison_report(
            compare_runs,
            report_dir=args.output_dir / "comparison_report",
            frames=sorted(preview_frames),
        )
    print((args.output_dir / "raw_baseline_summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
