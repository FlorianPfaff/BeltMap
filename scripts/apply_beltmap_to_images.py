from __future__ import annotations

import csv
import json
import math
import os
import re
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from beltmap import (
    BeltMotionModel,
    ParticleComponentConfig,
    ParticleTrackingConfig,
    PhaseRegistrationConfig,
    ResidualConfig,
    ResidualImage,
    detect_particles_from_residual,
    estimate_particle_velocities_vs_belt,
    extract_particle_detections,
    render_clean_belt_residual,
    track_particle_detections,
)

DATA = Path(os.getenv("BELTMAP_IMAGE_DIR", "data/images"))
OUT = Path(os.getenv("BELTMAP_OUTPUT_DIR", "outputs"))
EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
START_TIME = time.perf_counter()

DETECTION_FIELDS = [
    "frame_index", "image", "label", "y", "x", "area_px",
    "bbox_top", "bbox_left", "bbox_bottom", "bbox_right",
    "mean_signal", "peak_signal",
]
VELOCITY_FIELDS = [
    "track_id", "n_detections", "frame_start", "frame_end",
    "velocity_y_px_per_frame", "velocity_x_px_per_frame", "speed_px_per_frame",
    "belt_velocity_y_px_per_frame", "velocity_ratio_y",
    "belt_minus_particle_velocity_y_px_per_frame",
]


def env_int(name: str, default: int, minimum: int | None = None) -> int:
    value = os.getenv(name, "").strip()
    parsed = default if value == "" else int(value)
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name}={parsed} is below minimum {minimum}")
    return parsed


def env_float(name: str, default: float, minimum: float | None = None) -> float:
    value = os.getenv(name, "").strip()
    parsed = default if value == "" else float(value)
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name}={parsed} is below minimum {minimum}")
    return parsed


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "").strip().lower()
    if value == "":
        return default
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def elapsed_s() -> float:
    return time.perf_counter() - START_TIME


def rss_mb() -> float | None:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return None


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    return value


def emit(stage: str, message: str, **data: Any) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(elapsed_s(), 3),
        "stage": stage,
        "message": message,
    }
    mem = rss_mb()
    if mem is not None:
        payload["rss_mb"] = round(mem, 1)
    payload.update({k: jsonable(v) for k, v in data.items()})
    compact = {k: v for k, v in payload.items() if k not in {"timestamp", "stage", "message"}}
    print(f"[{payload['elapsed_s']:9.1f}s] {stage}: {message} {json.dumps(compact, sort_keys=True)}", flush=True)
    with (OUT / "progress.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    (OUT / "progress_latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def natural_key(path: Path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(path))]


def image_paths() -> tuple[list[Path], int, int]:
    all_paths = sorted(
        [p for p in DATA.rglob("*") if p.suffix.lower() in EXTS and not p.name.startswith("._")],
        key=natural_key,
    )
    if not all_paths:
        raise SystemExit(f"No image files found below {DATA}")
    frame_stride = env_int("FRAME_STRIDE", 1, minimum=1)
    paths = all_paths[::frame_stride]
    max_frames = env_int("MAX_FRAMES", 0, minimum=0)
    if max_frames > 0:
        paths = paths[:max_frames]
    return paths, len(all_paths), frame_stride


def read_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.float32)


def parse_region(first_frame: np.ndarray) -> tuple[int, int, int, int]:
    value = os.getenv("BELT_REGION", "").strip()
    height, width = first_frame.shape
    if not value:
        return 0, 0, height, width
    top, left, crop_height, crop_width = [int(x.strip()) for x in value.split(",")]
    if (
        top < 0 or left < 0 or crop_height <= 0 or crop_width <= 0
        or top + crop_height > height or left + crop_width > width
    ):
        raise ValueError(f"Invalid BELT_REGION={value!r} for image shape {(height, width)}")
    return top, left, crop_height, crop_width


def crop(frame: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
    top, left, height, width = region
    return frame[top: top + height, left: left + width]


def is_full_frame_region(
    region: tuple[int, int, int, int],
    frame_shape: tuple[int, int],
) -> bool:
    top, left, height, width = region
    return top == 0 and left == 0 and (height, width) == frame_shape


def validate_auto_velocity_region(
    region: tuple[int, int, int, int],
    frame_shape: tuple[int, int],
) -> None:
    if is_full_frame_region(region, frame_shape) and not env_bool("ALLOW_FULL_FRAME_AUTO_VELOCITY"):
        raise ValueError(
            "BELT_VELOCITY_PX_PER_FRAME=auto is unsafe with a full-frame BELT_REGION. "
            "Set BELT_REGION to the belt crop, supply BELT_VELOCITY_PX_PER_FRAME explicitly, "
            "or set ALLOW_FULL_FRAME_AUTO_VELOCITY=1 if the full frame truly contains only belt texture."
        )


def validate_auto_velocity_estimate(
    velocity: float,
    shifts: list[float],
    *,
    max_shift: int,
) -> None:
    min_abs_velocity = env_float("AUTO_VELOCITY_MIN_ABS_PX_PER_FRAME", 0.25, minimum=0.0)
    if abs(velocity) < min_abs_velocity:
        raise ValueError(
            f"Auto-estimated belt velocity {velocity:.6g} px/frame is below "
            f"AUTO_VELOCITY_MIN_ABS_PX_PER_FRAME={min_abs_velocity}. "
            "This usually means static background dominated the crop. Supply BELT_REGION and/or "
            "BELT_VELOCITY_PX_PER_FRAME explicitly."
        )

    max_edge_fraction = env_float("AUTO_VELOCITY_MAX_EDGE_FRACTION", 0.2, minimum=0.0)
    if max_edge_fraction > 1.0:
        raise ValueError("AUTO_VELOCITY_MAX_EDGE_FRACTION must be in [0, 1]")
    if shifts:
        edge_fraction = float(np.mean(np.abs(np.asarray(shifts)) >= 0.9 * max_shift))
        if edge_fraction > max_edge_fraction:
            raise ValueError(
                f"Auto velocity search often hit the search edge: edge_fraction={edge_fraction:.3f}, "
                f"max_shift={max_shift}. Increase VELOCITY_SEARCH_RADIUS_PX or supply "
                "BELT_VELOCITY_PX_PER_FRAME explicitly."
            )


def correlation_shift(previous: np.ndarray, current: np.ndarray, max_shift: int) -> float:
    def score(shift: int) -> float:
        if shift > 0:
            a, b = previous[:-shift], current[shift:]
        elif shift < 0:
            a, b = previous[-shift:], current[:shift]
        else:
            a, b = previous, current
        a = a.astype(np.float64, copy=False) - float(np.mean(a))
        b = b.astype(np.float64, copy=False) - float(np.mean(b))
        denominator = math.sqrt(float(np.sum(a * a)) * float(np.sum(b * b)))
        return -np.inf if denominator <= 0 else float(np.sum(a * b) / denominator)

    shifts = np.arange(-max_shift, max_shift + 1)
    scores = np.array([score(int(s)) for s in shifts])
    best_index = int(np.argmax(scores))
    best_shift = float(shifts[best_index])
    if 0 < best_index < len(scores) - 1:
        y0, y1, y2 = scores[best_index - 1], scores[best_index], scores[best_index + 1]
        denominator = y0 - 2 * y1 + y2
        if abs(float(denominator)) > 1e-12:
            delta = 0.5 * (y0 - y2) / denominator
            if np.isfinite(delta) and -1 <= delta <= 1:
                best_shift += float(delta)
    return best_shift


def estimate_velocity(paths: list[Path], region: tuple[int, int, int, int]) -> tuple[float, list[float]]:
    max_shift = env_int("VELOCITY_SEARCH_RADIUS_PX", 50, minimum=1)
    pair_count = min(len(paths) - 1, env_int("VELOCITY_ESTIMATION_PAIRS", 100, minimum=1))
    progress_interval = env_int("PROGRESS_INTERVAL_FRAMES", 25, minimum=1)
    if pair_count < 1:
        raise ValueError("Automatic velocity estimation requires at least two frames")
    emit("velocity", "estimating belt velocity", pair_count=pair_count, max_shift_px=max_shift)
    shifts: list[float] = []
    previous = crop(read_gray(paths[0]), region)
    for index in range(1, pair_count + 1):
        current = crop(read_gray(paths[index]), region)
        shifts.append(correlation_shift(previous, current, max_shift))
        previous = current
        if index == 1 or index == pair_count or index % progress_interval == 0:
            emit("velocity", f"estimated {index}/{pair_count} shifts", current_shift_px=shifts[-1], median_shift_px=float(np.median(shifts)))
    velocity = float(np.median(shifts))
    validate_auto_velocity_estimate(velocity, shifts, max_shift=max_shift)
    return velocity, shifts


def optional_positive_int(name: str) -> int | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def belt_phase(frame_index: int, velocity: float, reference_phase: float, period: float | None) -> float:
    phase = reference_phase - velocity * frame_index
    return phase % period if period else phase


def map_geometry(frame_count: int, crop_height: int, velocity: float, supplied_period: int | None) -> tuple[int, float, float | None]:
    if supplied_period:
        return supplied_period, 0.0, float(supplied_period)
    phases = -velocity * np.arange(frame_count, dtype=np.float64)
    reference_phase = -float(np.min(phases))
    map_height = int(math.ceil(float(np.max(phases) - np.min(phases)) + crop_height + 2))
    return max(map_height, crop_height), reference_phase, None


def sample_indices(frame_count: int, sample_count: int) -> list[int]:
    sample_count = max(1, min(frame_count, sample_count))
    return sorted(set(int(i) for i in np.linspace(0, frame_count - 1, sample_count)))


def build_belt_map(paths: list[Path], region: tuple[int, int, int, int], velocity: float, supplied_period: int | None) -> tuple[np.ndarray, float, int]:
    _, _, crop_height, crop_width = region
    max_samples = env_int("MAP_SAMPLE_FRAMES", 120, minimum=1)
    progress_interval = env_int("PROGRESS_INTERVAL_FRAMES", 25, minimum=1)
    map_height, reference_phase, model_period = map_geometry(len(paths), crop_height, velocity, supplied_period)
    samples = sample_indices(len(paths), max_samples)
    emit("belt_map", "building clean belt map", sampled_frames=len(samples), selected_frames=len(paths), crop_height=crop_height, crop_width=crop_width, map_height=map_height)

    sums = np.zeros((map_height, crop_width), dtype=np.float64)
    counts = np.zeros(map_height, dtype=np.float64)
    for sample_number, index in enumerate(samples, start=1):
        frame = crop(read_gray(paths[index]), region).astype(np.float64, copy=False)
        coordinates = np.rint(np.arange(crop_height) + belt_phase(index, velocity, reference_phase, model_period)).astype(np.int64)
        coordinates = coordinates % map_height if model_period else np.clip(coordinates, 0, map_height - 1)
        for y, row in enumerate(coordinates):
            sums[row] += frame[y]
            counts[row] += 1
        if sample_number == 1 or sample_number == len(samples) or sample_number % progress_interval == 0:
            emit("belt_map", f"accumulated {sample_number}/{len(samples)} sampled frames", source_frame_index=index, covered_rows=int(np.count_nonzero(counts)))

    known_rows = counts > 0
    if not np.any(known_rows):
        raise RuntimeError("No rows contributed to the belt map")
    emit("belt_map", "interpolating unobserved rows", known_rows=int(np.count_nonzero(known_rows)), total_rows=int(map_height))

    belt_map = np.empty_like(sums, dtype=np.float32)
    belt_map[known_rows] = (sums[known_rows] / counts[known_rows, None]).astype(np.float32)
    x = np.arange(map_height, dtype=np.float64)
    known = np.flatnonzero(known_rows)
    for col in range(crop_width):
        values = belt_map[known, col].astype(np.float64)
        if supplied_period and known.size > 1:
            xp = np.r_[known - map_height, known, known + map_height].astype(np.float64)
            fp = np.r_[values, values, values]
            belt_map[:, col] = np.interp(x, xp, fp).astype(np.float32)
        else:
            belt_map[:, col] = np.interp(x, known.astype(np.float64), values).astype(np.float32)
    return belt_map, reference_phase, map_height


def as_display_array(array: Any) -> np.ndarray:
    if isinstance(array, ResidualImage):
        return np.asarray(array.normalized, dtype=np.float64)
    if hasattr(array, "normalized"):
        return np.asarray(array.normalized, dtype=np.float64)
    return np.asarray(array, dtype=np.float64)


def save_png(array: Any, path: Path) -> None:
    arr = as_display_array(array)
    finite = np.isfinite(arr)
    low, high = np.percentile(arr[finite], [1, 99]) if finite.any() else (0, 1)
    if high <= low:
        high = low + 1
    Image.fromarray(np.clip((arr - low) / (high - low) * 255, 0, 255).astype(np.uint8)).save(path)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_detection_outputs(detections_by_frame: list, detection_rows: list[dict]) -> None:
    write_csv(OUT / "detections.csv", detection_rows, DETECTION_FIELDS)
    write_csv(OUT / "detections_per_frame.csv", [{"frame_index": i, "n_detections": len(dets)} for i, dets in enumerate(detections_by_frame)], ["frame_index", "n_detections"])


def should_save_residual_preview(frame_index: int, preview_frames: int, preview_interval: int) -> bool:
    return frame_index < preview_frames or (preview_interval > 0 and frame_index % preview_interval == 0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    emit("startup", "starting BeltMap image driver", data_dir=DATA, output_dir=OUT)
    paths, discovered_frame_count, frame_stride = image_paths()
    emit("images", "selected image sequence", discovered_frames=discovered_frame_count, selected_frames=len(paths), frame_stride=frame_stride, first_image=paths[0], last_image=paths[-1])

    first = read_gray(paths[0])
    region = parse_region(first)
    emit("images", "loaded first frame and parsed crop region", first_image_shape=list(first.shape), belt_region={"top": region[0], "left": region[1], "height": region[2], "width": region[3]})

    velocity_spec = os.getenv("BELT_VELOCITY_PX_PER_FRAME", "auto").strip().lower()
    if velocity_spec == "auto":
        validate_auto_velocity_region(region, first.shape)
        belt_velocity, pair_shifts = estimate_velocity(paths, region)
    else:
        belt_velocity, pair_shifts = float(velocity_spec), []
        emit("velocity", "using supplied belt velocity", belt_velocity_px_per_frame=belt_velocity)

    period_px = optional_positive_int("BELT_PERIOD_PX")
    detection_threshold = env_float("DETECTION_THRESHOLD", 5.0)
    min_area_px = env_int("MIN_AREA_PX", 4, minimum=1)
    min_track_length = env_int("MIN_TRACK_LENGTH", 2, minimum=1)
    emit("config", "runtime parameters", belt_velocity_px_per_frame=belt_velocity, belt_period_px=period_px, detection_threshold=detection_threshold, min_area_px=min_area_px, min_track_length=min_track_length)

    belt_map, reference_phase, map_height = build_belt_map(paths, region, belt_velocity, period_px)
    np.save(OUT / "belt_map.npy", belt_map)
    save_png(belt_map, OUT / "belt_map.png")
    emit("belt_map", "saved belt-map outputs", belt_map_shape=list(belt_map.shape), belt_map_npy=OUT / "belt_map.npy", belt_map_png=OUT / "belt_map.png")

    motion_model = BeltMotionModel(image_velocity_px_per_frame=belt_velocity, period_px=float(map_height), reference_frame=0.0, reference_phase_px=reference_phase)
    registration_config = PhaseRegistrationConfig(search_radius_px=env_float("REGISTRATION_SEARCH_RADIUS_PX", 8.0, minimum=0.0), search_step_px=env_float("REGISTRATION_SEARCH_STEP_PX", 0.5, minimum=1e-9))
    component_config = ParticleComponentConfig(min_area_px=min_area_px)
    residual_config = ResidualConfig()

    progress_interval = env_int("PROGRESS_INTERVAL_FRAMES", 25, minimum=1)
    partial_output_interval = env_int("PARTIAL_OUTPUT_INTERVAL_FRAMES", 250, minimum=0)
    residual_preview_frames = env_int("DEBUG_RESIDUAL_PREVIEW_FRAMES", 3, minimum=0)
    residual_preview_interval = env_int("DEBUG_RESIDUAL_PREVIEW_INTERVAL_FRAMES", 0, minimum=0)
    emit("detect", "starting residual rendering and particle detection", selected_frames=len(paths), progress_interval_frames=progress_interval, partial_output_interval_frames=partial_output_interval, residual_preview_frames=residual_preview_frames, residual_preview_interval_frames=residual_preview_interval)

    detections_by_frame = []
    detection_rows: list[dict] = []
    detection_start = time.perf_counter()
    for frame_index, path in enumerate(paths):
        frame = crop(read_gray(path), region)
        residual = render_clean_belt_residual(image=frame, belt_map=belt_map, frame_index=float(frame_index), motion_model=motion_model, belt_region=None, registration_config=registration_config, residual_config=residual_config)
        if should_save_residual_preview(frame_index, residual_preview_frames, residual_preview_interval):
            save_png(residual, OUT / f"residual_frame_{frame_index:06d}.png")
        mask = detect_particles_from_residual(residual, threshold=detection_threshold)
        detections = extract_particle_detections(mask, residual=residual, frame_index=float(frame_index), config=component_config)
        detections_by_frame.append(detections)
        for detection in detections:
            detection_rows.append({
                "frame_index": frame_index,
                "image": str(path.relative_to(DATA)),
                "label": detection.label,
                "y": detection.y,
                "x": detection.x,
                "area_px": detection.area_px,
                "bbox_top": detection.bbox_top,
                "bbox_left": detection.bbox_left,
                "bbox_bottom": detection.bbox_bottom,
                "bbox_right": detection.bbox_right,
                "mean_signal": detection.mean_signal,
                "peak_signal": detection.peak_signal,
            })
        processed = frame_index + 1
        if partial_output_interval > 0 and (processed == 1 or processed % partial_output_interval == 0):
            write_detection_outputs(detections_by_frame, detection_rows)
            emit("detect", "wrote partial detection outputs", processed_frames=processed, total_detections=len(detection_rows))
        if processed == 1 or processed == len(paths) or processed % progress_interval == 0:
            dt = time.perf_counter() - detection_start
            fps = processed / dt if dt > 0 else float("inf")
            remaining = len(paths) - processed
            eta = remaining / fps if fps > 0 else float("inf")
            emit("detect", f"processed {processed}/{len(paths)} frames", processed_frames=processed, remaining_frames=remaining, detections_this_frame=len(detections), total_detections=len(detection_rows), frames_per_second=round(fps, 4), eta_s=round(eta, 1) if math.isfinite(eta) else None, current_image=path)

    write_detection_outputs(detections_by_frame, detection_rows)
    emit("detect", "finished residual rendering and detection", processed_frames=len(paths), total_detections=len(detection_rows))

    max_match = os.getenv("MAX_MATCH_DISTANCE_PX", "").strip()
    tracking_config = ParticleTrackingConfig(max_match_distance_px=float(max_match) if max_match else max(5.0, 1.5 * abs(belt_velocity)), velocity_prior_y_px_per_frame=0.8 * belt_velocity)
    emit("track", "starting particle tracking", frames=len(detections_by_frame), max_match_distance_px=tracking_config.max_match_distance_px, velocity_prior_y_px_per_frame=tracking_config.velocity_prior_y_px_per_frame)
    tracks = track_particle_detections(detections_by_frame, config=tracking_config, frame_indices=[float(i) for i in range(len(paths))])
    emit("track", "finished particle tracking", tracks=len(tracks))

    velocity_rows = []
    if abs(belt_velocity) > 1e-9:
        emit("velocity", "estimating particle velocities relative to belt", min_track_length=min_track_length)
        for velocity in estimate_particle_velocities_vs_belt(tracks, belt_image_velocity_px_per_frame=belt_velocity, min_track_length=min_track_length):
            velocity_rows.append(asdict(velocity))
    else:
        emit("velocity", "skipped particle velocity estimation because belt velocity is near zero")
    write_csv(OUT / "velocities.csv", velocity_rows, VELOCITY_FIELDS)
    emit("velocity", "wrote velocity estimates", velocity_estimates=len(velocity_rows))

    metadata = {
        "n_images": len(paths),
        "discovered_frame_count": discovered_frame_count,
        "frame_stride": frame_stride,
        "first_image_shape": list(first.shape),
        "belt_region": {"top": region[0], "left": region[1], "height": region[2], "width": region[3]},
        "belt_velocity_px_per_frame": belt_velocity,
        "belt_period_px_input": period_px,
        "belt_map_height_px": map_height,
        "reference_phase_px": reference_phase,
        "detection_threshold": detection_threshold,
        "min_area_px": min_area_px,
        "n_detections": len(detection_rows),
        "n_tracks": len(tracks),
        "n_velocity_estimates": len(velocity_rows),
        "auto_velocity_pair_shifts": pair_shifts,
        "elapsed_s": elapsed_s(),
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (OUT / "summary.md").write_text(
        "# BeltMap run summary\n\n"
        f"- Images discovered: {discovered_frame_count}\n"
        f"- Images processed: {len(paths)}\n"
        f"- Frame stride: {frame_stride}\n"
        f"- Belt velocity: {belt_velocity:.6g} px/frame\n"
        f"- Belt map height: {map_height}\n"
        f"- Detections: {len(detection_rows)}\n"
        f"- Tracks: {len(tracks)}\n"
        f"- Velocity estimates: {len(velocity_rows)}\n"
        f"- Elapsed seconds: {elapsed_s():.1f}\n",
        encoding="utf-8",
    )
    emit("done", "BeltMap image driver completed", **metadata)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
