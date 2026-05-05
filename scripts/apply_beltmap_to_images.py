from __future__ import annotations

import csv
import json
import math
import os
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from beltmap import (
    BeltMotionModel,
    ParticleComponentConfig,
    ParticleTrackingConfig,
    PhaseRegistrationConfig,
    ResidualConfig,
    detect_particles_from_residual,
    estimate_particle_velocities_vs_belt,
    extract_particle_detections,
    render_clean_belt_residual,
    track_particle_detections,
)

DATA = Path(os.getenv("BELTMAP_IMAGE_DIR", "data/images"))
OUT = Path(os.getenv("BELTMAP_OUTPUT_DIR", "outputs"))
EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def natural_key(path: Path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(path))]


def image_paths() -> list[Path]:
    paths = sorted(
        [p for p in DATA.rglob("*") if p.suffix.lower() in EXTS and not p.name.startswith("._")],
        key=natural_key,
    )
    if not paths:
        raise SystemExit(f"No image files found below {DATA}")
    max_frames = int(os.getenv("MAX_FRAMES", "0") or "0")
    return paths[:max_frames] if max_frames > 0 else paths


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
        top < 0
        or left < 0
        or crop_height <= 0
        or crop_width <= 0
        or top + crop_height > height
        or left + crop_width > width
    ):
        raise ValueError(f"Invalid BELT_REGION={value!r} for image shape {(height, width)}")
    return top, left, crop_height, crop_width


def crop(frame: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
    top, left, height, width = region
    return frame[top : top + height, left : left + width]


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
        if denominator <= 0:
            return -np.inf
        return float(np.sum(a * b) / denominator)

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
    max_shift = int(os.getenv("VELOCITY_SEARCH_RADIUS_PX", "50"))
    pair_count = min(len(paths) - 1, int(os.getenv("VELOCITY_ESTIMATION_PAIRS", "100")))
    if pair_count < 1:
        raise ValueError("Automatic velocity estimation requires at least two frames")
    shifts: list[float] = []
    previous = crop(read_gray(paths[0]), region)
    for index in range(1, pair_count + 1):
        current = crop(read_gray(paths[index]), region)
        shifts.append(correlation_shift(previous, current, max_shift))
        previous = current
    return float(np.median(shifts)), shifts


def optional_positive_int(name: str) -> int | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def belt_phase(frame_index: int, velocity: float, reference_phase: float, period: float | None) -> float:
    phase = reference_phase - velocity * frame_index
    return phase % period if period else phase


def map_geometry(
    frame_count: int, crop_height: int, velocity: float, supplied_period: int | None
) -> tuple[int, float, float | None]:
    if supplied_period:
        return supplied_period, 0.0, float(supplied_period)
    phases = -velocity * np.arange(frame_count, dtype=np.float64)
    reference_phase = -float(np.min(phases))
    map_height = int(math.ceil(float(np.max(phases) - np.min(phases)) + crop_height + 2))
    return max(map_height, crop_height), reference_phase, None


def sample_indices(frame_count: int, sample_count: int) -> list[int]:
    sample_count = max(1, min(frame_count, sample_count))
    return sorted(set(int(i) for i in np.linspace(0, frame_count - 1, sample_count)))


def build_belt_map(
    paths: list[Path],
    region: tuple[int, int, int, int],
    velocity: float,
    supplied_period: int | None,
) -> tuple[np.ndarray, float, int]:
    _, _, crop_height, crop_width = region
    max_samples = int(os.getenv("MAP_SAMPLE_FRAMES", "120"))
    map_height, reference_phase, model_period = map_geometry(
        len(paths), crop_height, velocity, supplied_period
    )
    sums = np.zeros((map_height, crop_width), dtype=np.float64)
    counts = np.zeros(map_height, dtype=np.float64)

    for index in sample_indices(len(paths), max_samples):
        frame = crop(read_gray(paths[index]), region).astype(np.float64, copy=False)
        coordinates = np.rint(
            np.arange(crop_height) + belt_phase(index, velocity, reference_phase, model_period)
        ).astype(np.int64)
        coordinates = coordinates % map_height if model_period else np.clip(coordinates, 0, map_height - 1)
        for y, row in enumerate(coordinates):
            sums[row] += frame[y]
            counts[row] += 1

    known_rows = counts > 0
    if not np.any(known_rows):
        raise RuntimeError("No rows contributed to the belt map")

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


def save_png(array: np.ndarray, path: Path) -> None:
    finite = np.isfinite(array)
    low, high = np.percentile(array[finite], [1, 99]) if finite.any() else (0, 1)
    if high <= low:
        high = low + 1
    Image.fromarray(np.clip((array - low) / (high - low) * 255, 0, 255).astype(np.uint8)).save(path)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = image_paths()
    first = read_gray(paths[0])
    region = parse_region(first)

    velocity_spec = os.getenv("BELT_VELOCITY_PX_PER_FRAME", "auto").strip().lower()
    if velocity_spec == "auto":
        belt_velocity, pair_shifts = estimate_velocity(paths, region)
    else:
        belt_velocity, pair_shifts = float(velocity_spec), []

    period_px = optional_positive_int("BELT_PERIOD_PX")
    detection_threshold = float(os.getenv("DETECTION_THRESHOLD", "5.0"))
    min_area_px = int(os.getenv("MIN_AREA_PX", "4"))
    min_track_length = int(os.getenv("MIN_TRACK_LENGTH", "2"))

    belt_map, reference_phase, map_height = build_belt_map(paths, region, belt_velocity, period_px)
    np.save(OUT / "belt_map.npy", belt_map)
    save_png(belt_map, OUT / "belt_map.png")

    motion_model = BeltMotionModel(
        image_velocity_px_per_frame=belt_velocity,
        period_px=float(map_height),
        reference_frame=0.0,
        reference_phase_px=reference_phase,
    )
    registration_config = PhaseRegistrationConfig(
        search_radius_px=float(os.getenv("REGISTRATION_SEARCH_RADIUS_PX", "8.0")),
        search_step_px=float(os.getenv("REGISTRATION_SEARCH_STEP_PX", "0.5")),
    )
    component_config = ParticleComponentConfig(min_area_px=min_area_px)
    residual_config = ResidualConfig()

    detections_by_frame = []
    detection_rows: list[dict] = []
    for frame_index, path in enumerate(paths):
        frame = crop(read_gray(path), region)
        residual = render_clean_belt_residual(
            image=frame,
            belt_map=belt_map,
            frame_index=float(frame_index),
            motion_model=motion_model,
            belt_region=None,
            registration_config=registration_config,
            residual_config=residual_config,
        )
        mask = detect_particles_from_residual(residual, threshold=detection_threshold)
        detections = extract_particle_detections(
            mask,
            residual=residual,
            frame_index=float(frame_index),
            config=component_config,
        )
        detections_by_frame.append(detections)
        for detection in detections:
            detection_rows.append(
                {
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
                }
            )

    detection_fields = [
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
    write_csv(OUT / "detections.csv", detection_rows, detection_fields)
    write_csv(
        OUT / "detections_per_frame.csv",
        [{"frame_index": i, "n_detections": len(dets)} for i, dets in enumerate(detections_by_frame)],
        ["frame_index", "n_detections"],
    )

    max_match = os.getenv("MAX_MATCH_DISTANCE_PX", "").strip()
    tracking_config = ParticleTrackingConfig(
        max_match_distance_px=float(max_match) if max_match else max(5.0, 1.5 * abs(belt_velocity)),
        velocity_prior_y_px_per_frame=0.8 * belt_velocity,
    )
    tracks = track_particle_detections(
        detections_by_frame,
        config=tracking_config,
        frame_indices=[float(i) for i in range(len(paths))],
    )

    velocity_rows = []
    if abs(belt_velocity) > 1e-9:
        for velocity in estimate_particle_velocities_vs_belt(
            tracks,
            belt_image_velocity_px_per_frame=belt_velocity,
            min_track_length=min_track_length,
        ):
            velocity_rows.append(asdict(velocity))

    velocity_fields = [
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
    write_csv(OUT / "velocities.csv", velocity_rows, velocity_fields)

    metadata = {
        "n_images": len(paths),
        "first_image_shape": list(first.shape),
        "belt_region": {
            "top": region[0],
            "left": region[1],
            "height": region[2],
            "width": region[3],
        },
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
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (OUT / "summary.md").write_text(
        "# BeltMap run summary\n\n"
        f"- Images processed: {len(paths)}\n"
        f"- Belt velocity: {belt_velocity:.6g} px/frame\n"
        f"- Belt map height: {map_height} px\n"
        f"- Detections: {len(detection_rows)}\n"
        f"- Tracks: {len(tracks)}\n"
        f"- Velocity estimates: {len(velocity_rows)}\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
