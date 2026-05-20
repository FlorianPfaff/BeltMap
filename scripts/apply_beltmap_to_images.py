from __future__ import annotations

from beltmap.cli.apply import main as _beltmap_cli_main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_beltmap_cli_main())


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
    PhaseTrajectorySmoothingConfig,
    ResidualConfig,
    ResidualImage,
    detect_particles_from_residual,
    estimate_phase,
    estimate_particle_velocities_vs_belt,
    extract_particle_detections,
    generate_residual_image,
    render_clean_belt_residual,
    render_belt_view,
    smooth_phase_estimates,
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
PHASE_FIELDS = [
    "frame_index", "image", "phase_px", "phase_fraction", "phase_rad",
    "predicted_phase_px", "correction_px", "loss", "score", "method",
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


def env_optional_float(
    name: str,
    default: float | None = None,
    minimum: float | None = None,
) -> float | None:
    value = os.getenv(name, "").strip()
    if value == "":
        return default
    parsed = float(value)
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


SELECTED_FRAME_VELOCITY_UNIT = "selected_frame"
SOURCE_FRAME_VELOCITY_UNIT = "source_frame"
VELOCITY_FRAME_UNITS = {
    SELECTED_FRAME_VELOCITY_UNIT,
    SOURCE_FRAME_VELOCITY_UNIT,
}


def normalize_velocity_frame_unit(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        SELECTED_FRAME_VELOCITY_UNIT: SELECTED_FRAME_VELOCITY_UNIT,
        "selected": SELECTED_FRAME_VELOCITY_UNIT,
        "processed": SELECTED_FRAME_VELOCITY_UNIT,
        "processed_frame": SELECTED_FRAME_VELOCITY_UNIT,
        "strided_frame": SELECTED_FRAME_VELOCITY_UNIT,
        "output_frame": SELECTED_FRAME_VELOCITY_UNIT,
        SOURCE_FRAME_VELOCITY_UNIT: SOURCE_FRAME_VELOCITY_UNIT,
        "source": SOURCE_FRAME_VELOCITY_UNIT,
        "original": SOURCE_FRAME_VELOCITY_UNIT,
        "original_frame": SOURCE_FRAME_VELOCITY_UNIT,
        "input_frame": SOURCE_FRAME_VELOCITY_UNIT,
        "raw_frame": SOURCE_FRAME_VELOCITY_UNIT,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        choices = ", ".join(sorted(VELOCITY_FRAME_UNITS))
        raise ValueError(
            f"BELT_VELOCITY_FRAME_UNIT must be one of {choices}; got {value!r}"
        ) from exc


def resolve_velocity_frame_unit(frame_stride: int) -> str:
    if frame_stride < 1:
        raise ValueError("FRAME_STRIDE must be at least 1")
    value = os.getenv("BELT_VELOCITY_FRAME_UNIT", "").strip()
    if value:
        return normalize_velocity_frame_unit(value)
    if frame_stride == 1:
        return SELECTED_FRAME_VELOCITY_UNIT
    raise ValueError(
        f"BELT_VELOCITY_PX_PER_FRAME was supplied with FRAME_STRIDE={frame_stride}. "
        "Set BELT_VELOCITY_FRAME_UNIT=selected_frame if the supplied velocity is "
        "already in pixels per processed/selected frame, or set "
        "BELT_VELOCITY_FRAME_UNIT=source_frame if it is in pixels per adjacent "
        "original input frame. Source-frame velocities are multiplied by FRAME_STRIDE."
    )


def resolve_supplied_velocity(velocity_spec: str, frame_stride: int) -> tuple[float, str, float]:
    raw_velocity = float(velocity_spec)
    frame_unit = resolve_velocity_frame_unit(frame_stride)
    if frame_unit == SOURCE_FRAME_VELOCITY_UNIT:
        return raw_velocity * frame_stride, frame_unit, raw_velocity
    return raw_velocity, frame_unit, raw_velocity


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


def build_belt_map(
    paths: list[Path],
    region: tuple[int, int, int, int],
    velocity: float,
    supplied_period: int | None,
    *,
    mask_iterations: int = 0,
    mask_threshold: float = 5.0,
    mask_margin_px: int = 8,
    mask_min_area_px: int = 4,
) -> tuple[np.ndarray, float, int]:
    _, _, crop_height, crop_width = region
    max_samples = env_int("MAP_SAMPLE_FRAMES", 120, minimum=1)
    map_height, reference_phase, model_period = map_geometry(len(paths), crop_height, velocity, supplied_period)
    samples = sample_indices(len(paths), max_samples)
    emit(
        "belt_map",
        "building clean belt map",
        sampled_frames=len(samples),
        selected_frames=len(paths),
        crop_height=crop_height,
        crop_width=crop_width,
        map_height=map_height,
        mask_iterations=mask_iterations,
        mask_threshold=mask_threshold,
        mask_margin_px=mask_margin_px,
        mask_min_area_px=mask_min_area_px,
    )

    belt_map, _coverage = accumulate_belt_map(
        paths=paths,
        samples=samples,
        region=region,
        velocity=velocity,
        reference_phase=reference_phase,
        model_period=model_period,
        map_height=map_height,
        previous_belt_map=None,
        mask_threshold=mask_threshold,
        mask_margin_px=mask_margin_px,
        mask_min_area_px=mask_min_area_px,
        pass_label="initial",
    )

    for iteration in range(1, mask_iterations + 1):
        belt_map, coverage = accumulate_belt_map(
            paths=paths,
            samples=samples,
            region=region,
            velocity=velocity,
            reference_phase=reference_phase,
            model_period=model_period,
            map_height=map_height,
            previous_belt_map=belt_map,
            mask_threshold=mask_threshold,
            mask_margin_px=mask_margin_px,
            mask_min_area_px=mask_min_area_px,
            pass_label=f"masked-{iteration}",
        )
        emit(
            "belt_map",
            f"completed particle-masked map iteration {iteration}/{mask_iterations}",
            masked_pixels=coverage["masked_pixels"],
            contributed_pixels=coverage["contributed_pixels"],
            observed_pixels=coverage["observed_pixels"],
            total_pixels=coverage["total_pixels"],
        )

    return belt_map, reference_phase, map_height


def accumulate_belt_map(
    *,
    paths: list[Path],
    samples: list[int],
    region: tuple[int, int, int, int],
    velocity: float,
    reference_phase: float,
    model_period: float | None,
    map_height: int,
    previous_belt_map: np.ndarray | None,
    mask_threshold: float,
    mask_margin_px: int,
    mask_min_area_px: int,
    pass_label: str,
) -> tuple[np.ndarray, dict[str, int]]:
    _, _, crop_height, crop_width = region
    progress_interval = env_int("PROGRESS_INTERVAL_FRAMES", 25, minimum=1)
    use_particle_mask = previous_belt_map is not None
    component_config = ParticleComponentConfig(min_area_px=mask_min_area_px)
    residual_config = ResidualConfig()

    sums = np.zeros((map_height, crop_width), dtype=np.float64)
    weights = np.zeros((map_height, crop_width), dtype=np.float64)
    masked_pixels = 0
    contributed_pixels = 0
    for sample_number, index in enumerate(samples, start=1):
        frame = crop(read_gray(paths[index]), region).astype(np.float64, copy=False)
        phase = belt_phase(index, velocity, reference_phase, model_period)
        valid = np.ones(frame.shape, dtype=bool)
        if use_particle_mask:
            expected = render_belt_view(previous_belt_map, phase, crop_height)
            residual = generate_residual_image(frame, expected, config=residual_config)
            raw_mask = detect_particles_from_residual(residual, threshold=mask_threshold)
            detections = extract_particle_detections(
                raw_mask,
                residual=residual,
                frame_index=float(index),
                config=component_config,
            )
            particle_mask = expanded_detection_mask(
                detections,
                frame.shape,
                margin_px=mask_margin_px,
            )
            valid &= ~particle_mask
            masked_pixels += int(np.count_nonzero(particle_mask))

        contributed_pixels += _accumulate_frame_linear(
            sums=sums,
            weights=weights,
            frame=frame,
            valid=valid,
            phase=phase,
            map_height=map_height,
            model_period=model_period,
        )
        if sample_number == 1 or sample_number == len(samples) or sample_number % progress_interval == 0:
            emit(
                "belt_map",
                f"accumulated {sample_number}/{len(samples)} sampled frames",
                pass_label=pass_label,
                source_frame_index=index,
                observed_pixels=int(np.count_nonzero(weights)),
                masked_pixels=masked_pixels,
            )

    known_pixels = weights > 0
    if not np.any(known_pixels):
        raise RuntimeError("No pixels contributed to the belt map")
    emit(
        "belt_map",
        "interpolating unobserved belt-map pixels",
        pass_label=pass_label,
        observed_pixels=int(np.count_nonzero(known_pixels)),
        total_pixels=int(weights.size),
        masked_pixels=masked_pixels,
    )

    belt_map = np.empty_like(sums, dtype=np.float32)
    x = np.arange(map_height, dtype=np.float64)
    total_weight = float(np.sum(weights))
    if total_weight <= 0:
        raise RuntimeError("No pixels contributed to the belt map")
    global_mean = float(np.sum(sums) / total_weight)
    for col in range(crop_width):
        known = np.flatnonzero(known_pixels[:, col])
        if known.size == 0:
            belt_map[:, col] = global_mean
            continue
        values = sums[known, col] / weights[known, col]
        if model_period and known.size > 1:
            xp = np.r_[known - map_height, known, known + map_height].astype(np.float64)
            fp = np.r_[values, values, values]
            belt_map[:, col] = np.interp(x, xp, fp).astype(np.float32)
        elif known.size == 1:
            belt_map[:, col] = float(values[0])
        else:
            belt_map[:, col] = np.interp(x, known.astype(np.float64), values).astype(np.float32)

    return belt_map, {
        "masked_pixels": masked_pixels,
        "contributed_pixels": contributed_pixels,
        "observed_pixels": int(np.count_nonzero(known_pixels)),
        "total_pixels": int(weights.size),
    }


def _accumulate_frame_linear(
    *,
    sums: np.ndarray,
    weights: np.ndarray,
    frame: np.ndarray,
    valid: np.ndarray,
    phase: float,
    map_height: int,
    model_period: float | None,
) -> int:
    """Accumulate one frame with the same linear row model used for rendering."""

    if sums.shape != weights.shape:
        raise ValueError("sums and weights must have the same shape")
    if sums.shape[0] != map_height:
        raise ValueError("map_height must match the accumulator height")
    if frame.ndim != 2:
        raise ValueError("frame must be a 2-D array")
    if valid.shape != frame.shape:
        raise ValueError("valid must have the same shape as frame")
    if frame.shape[1] != sums.shape[1]:
        raise ValueError("frame width must match the accumulator width")

    rows = np.arange(frame.shape[0], dtype=np.float64) + float(phase)
    if model_period:
        rows = np.mod(rows, map_height)
    else:
        rows = np.clip(rows, 0.0, float(map_height - 1))
    row0 = np.floor(rows).astype(np.int64)
    if model_period:
        row1 = (row0 + 1) % map_height
    else:
        row1 = np.minimum(row0 + 1, map_height - 1)
    row1_weight = rows - row0
    row0_weight = 1.0 - row1_weight

    contributed_pixels = 0
    for y in range(frame.shape[0]):
        valid_cols = valid[y]
        pixel_count = int(np.count_nonzero(valid_cols))
        if pixel_count == 0:
            continue
        values = frame[y, valid_cols]
        weight0 = float(row0_weight[y])
        weight1 = float(row1_weight[y])
        if weight0 > 0.0:
            sums[row0[y], valid_cols] += weight0 * values
            weights[row0[y], valid_cols] += weight0
        if weight1 > 0.0:
            sums[row1[y], valid_cols] += weight1 * values
            weights[row1[y], valid_cols] += weight1
        contributed_pixels += pixel_count
    return contributed_pixels


def expanded_detection_mask(
    detections: list,
    shape: tuple[int, int],
    *,
    margin_px: int,
) -> np.ndarray:
    if margin_px < 0:
        raise ValueError("margin_px must be non-negative")
    mask = np.zeros(shape, dtype=bool)
    height, width = shape
    for detection in detections:
        top = max(0, detection.bbox_top - margin_px)
        left = max(0, detection.bbox_left - margin_px)
        bottom = min(height, detection.bbox_bottom + margin_px)
        right = min(width, detection.bbox_right + margin_px)
        mask[top:bottom, left:right] = True
    return mask


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


def phase_estimate_row(frame_index: int, path: Path, residual: ResidualImage, period_px: float) -> dict:
    if residual.clean_render is None:
        raise ValueError("phase estimates require residuals with a clean belt render")
    estimate = residual.clean_render.phase_estimate
    phase_fraction = estimate.phase_px / period_px
    return {
        "frame_index": frame_index,
        "image": str(path.relative_to(DATA)),
        "phase_px": estimate.phase_px,
        "phase_fraction": phase_fraction,
        "phase_rad": phase_fraction * 2.0 * math.pi,
        "predicted_phase_px": estimate.predicted_phase_px,
        "correction_px": estimate.correction_px,
        "loss": "" if estimate.loss is None else estimate.loss,
        "score": "" if estimate.score is None else estimate.score,
        "method": estimate.method,
    }


def write_phase_outputs(phase_rows: list[dict]) -> None:
    write_csv(OUT / "phase_estimates.csv", phase_rows, PHASE_FIELDS)


def should_save_residual_preview(frame_index: int, preview_frames: int, preview_interval: int) -> bool:
    return frame_index < preview_frames or (preview_interval > 0 and frame_index % preview_interval == 0)


def maybe_smooth_phase_trajectory(
    *,
    paths: list[Path],
    region: tuple[int, int, int, int],
    belt_map: np.ndarray,
    motion_model: BeltMotionModel,
    registration_config: PhaseRegistrationConfig,
    smoothing_config: PhaseTrajectorySmoothingConfig,
    progress_interval: int,
) -> list | None:
    if smoothing_config.window_radius_frames <= 0:
        return None

    emit(
        "phase",
        "estimating per-frame registration trajectory",
        selected_frames=len(paths),
        window_radius_frames=smoothing_config.window_radius_frames,
        min_score=smoothing_config.min_score,
        max_abs_correction_px=smoothing_config.max_abs_correction_px,
        robust_sigma=smoothing_config.robust_sigma,
        min_support=smoothing_config.min_support,
    )
    start = time.perf_counter()
    raw_estimates = []
    for frame_index, path in enumerate(paths):
        frame = crop(read_gray(path), region)
        raw_estimates.append(
            estimate_phase(
                float(frame_index),
                motion_model,
                frame=frame,
                belt_map=belt_map,
                config=registration_config,
            )
        )
        processed = frame_index + 1
        if (
            processed == 1
            or processed == len(paths)
            or processed % progress_interval == 0
        ):
            dt = time.perf_counter() - start
            fps = processed / dt if dt > 0 else float("inf")
            emit(
                "phase",
                f"registered phase {processed}/{len(paths)} frames",
                processed_frames=processed,
                remaining_frames=len(paths) - processed,
                frames_per_second=round(fps, 4),
                current_image=path,
            )

    smoothed = smooth_phase_estimates(
        raw_estimates,
        period_px=motion_model.period_px,
        config=smoothing_config,
    )
    raw_corrections = np.asarray(
        [estimate.correction_px for estimate in raw_estimates],
        dtype=np.float64,
    )
    smoothed_corrections = np.asarray(
        [estimate.correction_px for estimate in smoothed],
        dtype=np.float64,
    )
    finite_delta = np.isfinite(raw_corrections) & np.isfinite(smoothed_corrections)
    max_delta = None
    median_delta = None
    if np.any(finite_delta):
        abs_delta = np.abs(
            smoothed_corrections[finite_delta] - raw_corrections[finite_delta]
        )
        max_delta = float(np.max(abs_delta))
        median_delta = float(np.median(abs_delta))
    emit(
        "phase",
        "finished phase trajectory smoothing",
        phase_estimates=len(smoothed),
        max_abs_smoothing_delta_px=max_delta,
        median_abs_smoothing_delta_px=median_delta,
    )
    return smoothed


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    emit("startup", "starting BeltMap image driver", data_dir=DATA, output_dir=OUT)
    paths, discovered_frame_count, frame_stride = image_paths()
    emit("images", "selected image sequence", discovered_frames=discovered_frame_count, selected_frames=len(paths), frame_stride=frame_stride, first_image=paths[0], last_image=paths[-1])

    first = read_gray(paths[0])
    region = parse_region(first)
    emit("images", "loaded first frame and parsed crop region", first_image_shape=list(first.shape), belt_region={"top": region[0], "left": region[1], "height": region[2], "width": region[3]})

    velocity_spec = os.getenv("BELT_VELOCITY_PX_PER_FRAME", "auto").strip().lower()
    belt_velocity_source = "auto"
    belt_velocity_frame_unit = "selected_frame"
    supplied_belt_velocity_px_per_frame: float | None = None
    if velocity_spec == "auto":
        validate_auto_velocity_region(region, first.shape)
        belt_velocity, pair_shifts = estimate_velocity(paths, region)
    else:
        (
            belt_velocity,
            belt_velocity_frame_unit,
            supplied_belt_velocity_px_per_frame,
        ) = resolve_supplied_velocity(velocity_spec, frame_stride)
        belt_velocity_source = "supplied"
        pair_shifts = []
        emit(
            "velocity",
            "using supplied belt velocity",
            supplied_belt_velocity_px_per_frame=supplied_belt_velocity_px_per_frame,
            belt_velocity_frame_unit=belt_velocity_frame_unit,
            belt_velocity_px_per_selected_frame=belt_velocity,
            frame_stride=frame_stride,
        )

    period_px = optional_positive_int("BELT_PERIOD_PX")
    detection_threshold = env_float("DETECTION_THRESHOLD", 5.0)
    min_area_px = env_int("MIN_AREA_PX", 4, minimum=1)
    min_track_length = env_int("MIN_TRACK_LENGTH", 2, minimum=1)
    map_mask_iterations = env_int("MAP_MASK_ITERATIONS", 1, minimum=0)
    map_particle_mask_threshold = env_float("MAP_PARTICLE_MASK_THRESHOLD", detection_threshold, minimum=0.0)
    map_particle_mask_margin_px = env_int("MAP_PARTICLE_MASK_MARGIN_PX", 8, minimum=0)
    map_particle_mask_min_area_px = env_int("MAP_PARTICLE_MASK_MIN_AREA_PX", min_area_px, minimum=1)
    emit(
        "config",
        "runtime parameters",
        belt_velocity_px_per_frame=belt_velocity,
        belt_velocity_source=belt_velocity_source,
        belt_velocity_frame_unit=belt_velocity_frame_unit,
        supplied_belt_velocity_px_per_frame=supplied_belt_velocity_px_per_frame,
        belt_period_px=period_px,
        detection_threshold=detection_threshold,
        min_area_px=min_area_px,
        min_track_length=min_track_length,
        map_mask_iterations=map_mask_iterations,
        map_particle_mask_threshold=map_particle_mask_threshold,
        map_particle_mask_margin_px=map_particle_mask_margin_px,
        map_particle_mask_min_area_px=map_particle_mask_min_area_px,
    )

    belt_map, reference_phase, map_height = build_belt_map(
        paths,
        region,
        belt_velocity,
        period_px,
        mask_iterations=map_mask_iterations,
        mask_threshold=map_particle_mask_threshold,
        mask_margin_px=map_particle_mask_margin_px,
        mask_min_area_px=map_particle_mask_min_area_px,
    )
    np.save(OUT / "belt_map.npy", belt_map)
    save_png(belt_map, OUT / "belt_map.png")
    emit("belt_map", "saved belt-map outputs", belt_map_shape=list(belt_map.shape), belt_map_npy=OUT / "belt_map.npy", belt_map_png=OUT / "belt_map.png")

    motion_model = BeltMotionModel(image_velocity_px_per_frame=belt_velocity, period_px=float(map_height), reference_frame=0.0, reference_phase_px=reference_phase)
    registration_config = PhaseRegistrationConfig(search_radius_px=env_float("REGISTRATION_SEARCH_RADIUS_PX", 8.0, minimum=0.0), search_step_px=env_float("REGISTRATION_SEARCH_STEP_PX", 0.5, minimum=1e-9))
    phase_smoothing_config = PhaseTrajectorySmoothingConfig(
        window_radius_frames=env_int(
            "PHASE_SMOOTHING_WINDOW_RADIUS_FRAMES",
            0,
            minimum=0,
        ),
        min_score=env_optional_float(
            "PHASE_SMOOTHING_MIN_SCORE",
            None,
            minimum=0.0,
        ),
        max_abs_correction_px=env_optional_float(
            "PHASE_SMOOTHING_MAX_ABS_CORRECTION_PX",
            registration_config.search_radius_px,
            minimum=0.0,
        ),
        robust_sigma=env_float(
            "PHASE_SMOOTHING_ROBUST_SIGMA",
            3.0,
            minimum=1e-9,
        ),
        min_support=env_int("PHASE_SMOOTHING_MIN_SUPPORT", 3, minimum=1),
    )
    component_config = ParticleComponentConfig(min_area_px=min_area_px)
    residual_config = ResidualConfig()

    progress_interval = env_int("PROGRESS_INTERVAL_FRAMES", 25, minimum=1)
    partial_output_interval = env_int("PARTIAL_OUTPUT_INTERVAL_FRAMES", 250, minimum=0)
    residual_preview_frames = env_int("DEBUG_RESIDUAL_PREVIEW_FRAMES", 3, minimum=0)
    residual_preview_interval = env_int("DEBUG_RESIDUAL_PREVIEW_INTERVAL_FRAMES", 0, minimum=0)
    phase_estimates_by_frame = maybe_smooth_phase_trajectory(
        paths=paths,
        region=region,
        belt_map=belt_map,
        motion_model=motion_model,
        registration_config=registration_config,
        smoothing_config=phase_smoothing_config,
        progress_interval=progress_interval,
    )
    emit("detect", "starting residual rendering and particle detection", selected_frames=len(paths), progress_interval_frames=progress_interval, partial_output_interval_frames=partial_output_interval, residual_preview_frames=residual_preview_frames, residual_preview_interval_frames=residual_preview_interval)

    detections_by_frame = []
    detection_rows: list[dict] = []
    phase_rows: list[dict] = []
    detection_start = time.perf_counter()
    for frame_index, path in enumerate(paths):
        frame = crop(read_gray(path), region)
        phase_estimate = (
            None
            if phase_estimates_by_frame is None
            else phase_estimates_by_frame[frame_index]
        )
        residual = render_clean_belt_residual(
            image=frame,
            belt_map=belt_map,
            frame_index=float(frame_index),
            motion_model=motion_model,
            belt_region=None,
            phase_estimate=phase_estimate,
            registration_config=(
                None if phase_estimate is not None else registration_config
            ),
            residual_config=residual_config,
        )
        phase_rows.append(phase_estimate_row(frame_index, path, residual, float(map_height)))
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
            write_phase_outputs(phase_rows)
            emit("detect", "wrote partial detection and phase outputs", processed_frames=processed, total_detections=len(detection_rows), phase_estimates=len(phase_rows))
        if processed == 1 or processed == len(paths) or processed % progress_interval == 0:
            dt = time.perf_counter() - detection_start
            fps = processed / dt if dt > 0 else float("inf")
            remaining = len(paths) - processed
            eta = remaining / fps if fps > 0 else float("inf")
            emit("detect", f"processed {processed}/{len(paths)} frames", processed_frames=processed, remaining_frames=remaining, detections_this_frame=len(detections), total_detections=len(detection_rows), frames_per_second=round(fps, 4), eta_s=round(eta, 1) if math.isfinite(eta) else None, current_image=path)

    write_detection_outputs(detections_by_frame, detection_rows)
    write_phase_outputs(phase_rows)
    emit("detect", "finished residual rendering, phase estimation, and detection", processed_frames=len(paths), total_detections=len(detection_rows), phase_estimates=len(phase_rows))

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
        "belt_velocity_source": belt_velocity_source,
        "belt_velocity_frame_unit": belt_velocity_frame_unit,
        "supplied_belt_velocity_px_per_frame": supplied_belt_velocity_px_per_frame,
        "belt_velocity_px_per_frame": belt_velocity,
        "belt_period_px_input": period_px,
        "belt_map_height_px": map_height,
        "reference_phase_px": reference_phase,
        "detection_threshold": detection_threshold,
        "min_area_px": min_area_px,
        "map_mask_iterations": map_mask_iterations,
        "map_particle_mask_threshold": map_particle_mask_threshold,
        "map_particle_mask_margin_px": map_particle_mask_margin_px,
        "map_particle_mask_min_area_px": map_particle_mask_min_area_px,
        "phase_smoothing_window_radius_frames": (
            phase_smoothing_config.window_radius_frames
        ),
        "phase_smoothing_min_score": phase_smoothing_config.min_score,
        "phase_smoothing_max_abs_correction_px": (
            phase_smoothing_config.max_abs_correction_px
        ),
        "phase_smoothing_robust_sigma": phase_smoothing_config.robust_sigma,
        "phase_smoothing_min_support": phase_smoothing_config.min_support,
        "n_phase_estimates": len(phase_rows),
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
        f"- Map particle-mask iterations: {map_mask_iterations}\n"
        "- Phase smoothing window radius: "
        f"{phase_smoothing_config.window_radius_frames}\n"
        f"- Phase estimates: {len(phase_rows)}\n"
        f"- Detections: {len(detection_rows)}\n"
        f"- Tracks: {len(tracks)}\n"
        f"- Velocity estimates: {len(velocity_rows)}\n"
        f"- Elapsed seconds: {elapsed_s():.1f}\n",
        encoding="utf-8",
    )
    emit("done", "BeltMap image driver completed", **metadata)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    # Keep this legacy script path aligned with the packaged console entry point.
    from beltmap.cli.apply import main as cli_main

    raise SystemExit(cli_main())
