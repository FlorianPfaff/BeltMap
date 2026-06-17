"""Trust, calibration, and evidence helpers for BeltMap runs.

The functions in this module are intentionally lightweight and depend only on
NumPy and Pillow. They complement the normal BeltMap outputs with preflight
checks, run-level quality diagnostics, physical-measurement sanity checks, and
compact evidence reports for experimental use.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


PROFILE_CONFIGS: dict[str, dict[str, dict[str, object]]] = {
    "high_precision": {
        "detection": {
            "threshold": 5.5,
            "min_area_px": 6,
            "min_bbox_width_px": 3,
            "min_bbox_height_px": 3,
            "max_bbox_aspect_ratio": 4.0,
            "min_bbox_extent": 0.15,
        },
        "track_filter": {
            "min_length": 8,
            "min_velocity_ratio_y": 0.0,
            "max_velocity_ratio_y": 1.05,
        },
        "recurrent_artifact": {
            "min_revolutions": 3,
            "mode": "soft",
            "max_overlap_fraction": 0.25,
        },
    },
    "high_recall": {
        "detection": {
            "threshold": 3.5,
            "min_area_px": 3,
            "min_bbox_width_px": 2,
            "min_bbox_height_px": 2,
            "max_bbox_aspect_ratio": 6.0,
            "min_bbox_extent": 0.08,
        },
        "track_filter": {
            "min_length": 4,
            "min_velocity_ratio_y": -0.1,
            "max_velocity_ratio_y": 1.2,
        },
        "recurrent_artifact": {
            "min_revolutions": 3,
            "mode": "soft",
            "max_overlap_fraction": 0.4,
        },
    },
    "velocity_quality": {
        "tracking": {
            "min_track_length": 3,
        },
        "track_filter": {
            "min_length": 10,
            "min_velocity_ratio_y": 0.0,
            "max_velocity_ratio_y": 1.1,
            "max_abs_x_velocity_px_per_frame": 0.0,
        },
        "debug": {
            "residual_preview_interval_frames": 500,
        },
    },
    "map_quality": {
        "map": {
            "sample_frames": 500,
            "mask_iterations": 2,
            "particle_mask_mode": "hysteresis_abs",
            "particle_mask_threshold": 4.0,
            "particle_mask_grow_threshold": 1.5,
            "particle_mask_dilation_px": 8,
            "particle_mask_margin_px": 16,
            "particle_mask_min_area_px": 8,
        },
        "static_background": {
            "sample_frames": 500,
            "mask_threshold": 4.0,
            "mask_margin_px": 16,
            "mask_min_area_px": 4,
        },
    },
    "fast_screening": {
        "frames": {
            "max_frames": 500,
            "stride": 2,
        },
        "map": {
            "sample_frames": 80,
            "mask_iterations": 1,
        },
        "debug": {
            "residual_preview_frames": 3,
            "residual_preview_interval_frames": 0,
        },
    },
}


@dataclass(frozen=True)
class ScaleCalibration:
    """Pixel-to-physical scale derived from a two-point calibration target."""

    px_per_mm: float
    point_a: tuple[float, float]
    point_b: tuple[float, float]
    known_distance_mm: float

    @property
    def mm_per_px(self) -> float:
        return 1.0 / self.px_per_mm


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object, returning an empty object when the file is absent."""

    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON object with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV rows, returning an empty list when the file is absent."""

    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    """Write CSV rows with explicit field order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def finite_float(value: Any) -> float | None:
    """Parse a finite float, returning ``None`` for blanks and invalid values."""

    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def finite_float_or(value: Any, default: float) -> float:
    parsed = finite_float(value)
    return default if parsed is None else parsed


def finite_int(value: Any) -> int | None:
    """Parse an integer, returning ``None`` when parsing fails."""

    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_region(value: str | Sequence[int] | None) -> tuple[int, int, int, int] | None:
    """Parse ``top,left,height,width`` from text or sequence form."""

    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")] if isinstance(value, str) else list(value)
    if len(parts) != 4:
        raise ValueError("region must contain four values: top,left,height,width")
    top, left, height, width = (int(part) for part in parts)
    if height <= 0 or width <= 0:
        raise ValueError("region height and width must be positive")
    return top, left, height, width


def natural_key(path: Path) -> list[int | str]:
    """Natural sort key for image filenames."""

    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", str(path))]


def image_paths(image_dir: Path) -> list[Path]:
    """Return naturally sorted image paths below ``image_dir``."""

    return sorted(
        [
            path
            for path in image_dir.rglob("*")
            if path.suffix.lower() in IMAGE_EXTENSIONS and not path.name.startswith("._")
        ],
        key=natural_key,
    )


def frame_number_from_path(path: Path) -> int | None:
    """Extract the last integer from an image stem as a likely frame number."""

    matches = re.findall(r"\d+", path.stem)
    return int(matches[-1]) if matches else None


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_report(
    image_dir: Path,
    *,
    hash_duplicates: bool = True,
    max_hash_files: int = 5000,
) -> dict[str, Any]:
    """Detect missing frame numbers, duplicate frame numbers, and duplicate images."""

    paths = image_paths(image_dir)
    frame_numbers = [frame_number_from_path(path) for path in paths]
    numeric = [number for number in frame_numbers if number is not None]
    missing_numbers: list[int] = []
    if numeric:
        observed = set(numeric)
        missing_numbers = [
            number
            for number in range(min(observed), max(observed) + 1)
            if number not in observed
        ]

    duplicate_frame_numbers: dict[int, list[str]] = {}
    by_number: dict[int, list[str]] = {}
    for path, number in zip(paths, frame_numbers):
        if number is None:
            continue
        by_number.setdefault(number, []).append(str(path))
    duplicate_frame_numbers = {
        number: names
        for number, names in by_number.items()
        if len(names) > 1
    }

    dimensions: dict[str, int] = {}
    for path in paths[: min(len(paths), 100)]:
        try:
            with Image.open(path) as image:
                key = f"{image.height}x{image.width}"
                dimensions[key] = dimensions.get(key, 0) + 1
        except OSError:
            dimensions["unreadable"] = dimensions.get("unreadable", 0) + 1

    duplicate_hashes: dict[str, list[str]] = {}
    if hash_duplicates and len(paths) <= max_hash_files:
        by_hash: dict[str, list[str]] = {}
        for path in paths:
            try:
                by_hash.setdefault(sha256_file(path), []).append(str(path))
            except OSError:
                continue
        duplicate_hashes = {
            digest: names
            for digest, names in by_hash.items()
            if len(names) > 1
        }

    warnings: list[str] = []
    if missing_numbers:
        warnings.append(f"missing {len(missing_numbers)} frame numbers")
    if duplicate_frame_numbers:
        warnings.append(f"duplicate frame numbers: {len(duplicate_frame_numbers)}")
    if duplicate_hashes:
        warnings.append(f"duplicate image files: {len(duplicate_hashes)}")
    if len(dimensions) > 1:
        warnings.append("multiple image dimensions observed in the sampled files")

    return {
        "image_dir": str(image_dir),
        "n_images": len(paths),
        "first_image": str(paths[0]) if paths else "",
        "last_image": str(paths[-1]) if paths else "",
        "numbered_frames": len(numeric),
        "missing_frame_numbers": missing_numbers,
        "duplicate_frame_numbers": duplicate_frame_numbers,
        "duplicate_hashes": duplicate_hashes,
        "sampled_dimensions": dimensions,
        "warnings": warnings,
    }


def _load_gray(path: Path, *, region: tuple[int, int, int, int] | None = None) -> np.ndarray:
    with Image.open(path) as image:
        arr = np.asarray(image.convert("L"), dtype=np.float64)
    if region is None:
        return arr
    top, left, height, width = region
    return arr[top : top + height, left : left + width]


def _laplacian_variance(gray: np.ndarray) -> float:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    center = gray[1:-1, 1:-1]
    lap = (
        -4.0 * center
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(np.var(lap))


def frame_quality_metrics(
    path: Path,
    *,
    region: tuple[int, int, int, int] | None = None,
    saturation_threshold: float = 250.0,
    dark_threshold: float = 5.0,
) -> dict[str, Any]:
    """Compute blur, saturation, and dynamic-range metrics for one frame."""

    gray = _load_gray(path, region=region)
    finite = np.isfinite(gray)
    values = gray[finite]
    if values.size == 0:
        return {
            "image": str(path),
            "valid": False,
            "reason": "no finite pixels",
        }

    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    gx_var = float(np.var(gx)) if gx.size else 0.0
    gy_var = float(np.var(gy)) if gy.size else 0.0
    p01, p50, p99 = np.percentile(values, [1, 50, 99])
    saturated_fraction = float(np.mean(values >= saturation_threshold))
    dark_fraction = float(np.mean(values <= dark_threshold))
    lap_var = _laplacian_variance(gray)

    warnings: list[str] = []
    if saturated_fraction > 0.01:
        warnings.append("more than 1% saturated pixels")
    if dark_fraction > 0.01:
        warnings.append("more than 1% near-black pixels")
    if lap_var < 1.0:
        warnings.append("very low Laplacian variance; possible blur or weak texture")

    return {
        "image": str(path),
        "valid": True,
        "height": int(gray.shape[0]),
        "width": int(gray.shape[1]),
        "intensity_p01": float(p01),
        "intensity_median": float(p50),
        "intensity_p99": float(p99),
        "dynamic_range_p99_p01": float(p99 - p01),
        "saturated_pixel_fraction": saturated_fraction,
        "dark_pixel_fraction": dark_fraction,
        "laplacian_variance": lap_var,
        "horizontal_gradient_variance": gx_var,
        "vertical_gradient_variance": gy_var,
        "vertical_to_horizontal_gradient_ratio": gy_var / max(gx_var, 1e-12),
        "warnings": warnings,
    }


def quality_report(
    image_dir: Path,
    *,
    region: tuple[int, int, int, int] | None = None,
    sample_limit: int = 100,
) -> dict[str, Any]:
    """Summarize image quality over a sampled subset of frames."""

    paths = image_paths(image_dir)
    if not paths:
        return {"image_dir": str(image_dir), "n_images": 0, "sampled_frames": 0, "warnings": ["no images found"]}
    sample_count = max(1, min(sample_limit, len(paths)))
    indices = sorted({int(i) for i in np.linspace(0, len(paths) - 1, sample_count)})
    rows = [
        frame_quality_metrics(paths[index], region=region)
        for index in indices
    ]
    valid_rows = [row for row in rows if row.get("valid")]
    summary = {
        "laplacian_variance": summarize_numeric(row.get("laplacian_variance") for row in valid_rows),
        "saturated_pixel_fraction": summarize_numeric(row.get("saturated_pixel_fraction") for row in valid_rows),
        "dark_pixel_fraction": summarize_numeric(row.get("dark_pixel_fraction") for row in valid_rows),
        "dynamic_range_p99_p01": summarize_numeric(row.get("dynamic_range_p99_p01") for row in valid_rows),
    }

    warnings: list[str] = []
    median_blur = summary["laplacian_variance"]["median"]
    if median_blur is not None and median_blur < 1.0:
        warnings.append("median Laplacian variance is very low; check motion blur or weak focus")
    sat_p95 = summary["saturated_pixel_fraction"]["p95"]
    if sat_p95 is not None and sat_p95 > 0.01:
        warnings.append("sampled frames contain substantial saturation")
    dark_p95 = summary["dark_pixel_fraction"]["p95"]
    if dark_p95 is not None and dark_p95 > 0.01:
        warnings.append("sampled frames contain substantial near-black clipping")

    return {
        "image_dir": str(image_dir),
        "n_images": len(paths),
        "sampled_frames": len(rows),
        "sample_indices": indices,
        "summary": summary,
        "warnings": warnings,
        "frames": rows,
    }


def summarize_numeric(values: Iterable[Any]) -> dict[str, float | int | None]:
    """Compact statistics for finite numeric values."""

    arr = np.asarray([value for value in (finite_float(v) for v in values) if value is not None], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None, "p05": None, "p95": None}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
    }


def _linear_slope(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2 or np.unique(x).size < 2:
        return None
    centered = x - float(np.mean(x))
    denom = float(np.sum(centered * centered))
    if denom <= 0:
        return None
    return float(np.sum(centered * (y - float(np.mean(y)))) / denom)


def speed_consistency_report(output_dir: Path) -> dict[str, Any]:
    """Compare configured belt speed with registration correction trends."""

    metadata = read_json(output_dir / "metadata.json")
    phase_rows = read_csv_rows(output_dir / "phase_estimates.csv")
    configured_velocity = finite_float(metadata.get("belt_velocity_px_per_frame"))
    frames: list[float] = []
    corrections: list[float] = []
    scores: list[float] = []
    boundary_hits = 0
    search_radius = finite_float(metadata.get("registration_search_radius_px"))
    for row in phase_rows:
        frame = finite_float(row.get("frame_index"))
        correction = finite_float(row.get("correction_px"))
        if frame is None or correction is None:
            continue
        frames.append(frame)
        corrections.append(correction)
        score = finite_float(row.get("score"))
        if score is not None:
            scores.append(score)
        if search_radius is not None and abs(abs(correction) - search_radius) <= 1e-9:
            boundary_hits += 1

    correction_slope = _linear_slope(frames, corrections)
    inferred_velocity = None
    if configured_velocity is not None and correction_slope is not None:
        # phase = reference - velocity * t. A positive correction trend means
        # the supplied velocity is too high in magnitude for the observed phase.
        inferred_velocity = configured_velocity - correction_slope

    pair_shifts = metadata.get("auto_velocity_pair_shifts", [])
    auto_shift_values: list[float] = []
    if isinstance(pair_shifts, list):
        for item in pair_shifts:
            value = item.get("shift_px") if isinstance(item, dict) else item
            parsed = finite_float(value)
            if parsed is not None:
                auto_shift_values.append(parsed)

    warnings: list[str] = []
    if correction_slope is not None and abs(correction_slope) > 0.05:
        warnings.append("phase corrections show a non-trivial linear trend; supplied belt velocity may be biased")
    if phase_rows and boundary_hits / len(phase_rows) > 0.05:
        warnings.append("more than 5% of registration corrections hit the search boundary")
    if scores and np.median(scores) < 0.1:
        warnings.append("median registration score is low")

    return {
        "output_dir": str(output_dir),
        "configured_belt_velocity_px_per_frame": configured_velocity,
        "phase_rows": len(phase_rows),
        "correction_slope_px_per_frame": correction_slope,
        "inferred_belt_velocity_px_per_frame": inferred_velocity,
        "registration_score_summary": summarize_numeric(scores),
        "boundary_hit_count": boundary_hits,
        "auto_velocity_shift_summary": summarize_numeric(auto_shift_values),
        "warnings": warnings,
    }


def run_drift_report(output_dir: Path) -> dict[str, Any]:
    """Track time trends that indicate map staleness, belt contamination, or unstable acquisition."""

    detection_counts = [
        finite_float(row.get("n_detections"))
        for row in read_csv_rows(output_dir / "detections_per_frame.csv")
    ]
    detection_counts = [value for value in detection_counts if value is not None]
    phase_rows = read_csv_rows(output_dir / "phase_estimates.csv")
    frames: list[float] = []
    corrections: list[float] = []
    losses: list[float] = []
    for row in phase_rows:
        frame = finite_float(row.get("frame_index"))
        if frame is None:
            continue
        frames.append(frame)
        correction = finite_float(row.get("correction_px"))
        loss = finite_float(row.get("loss"))
        if correction is not None:
            corrections.append(correction)
        if loss is not None:
            losses.append(loss)

    detection_slope = _linear_slope(list(range(len(detection_counts))), detection_counts)
    correction_slope = _linear_slope(frames[: len(corrections)], corrections) if corrections else None
    loss_slope = _linear_slope(frames[: len(losses)], losses) if losses else None

    warnings: list[str] = []
    if detection_slope is not None and abs(detection_slope) > 0.01:
        warnings.append("detection counts drift over time; check contamination, illumination drift, or threshold stability")
    if loss_slope is not None and loss_slope > 0:
        warnings.append("registration loss increases over time; consider multi-epoch maps or illumination correction")
    if correction_slope is not None and abs(correction_slope) > 0.05:
        warnings.append("phase correction drift suggests speed or timing mismatch")

    return {
        "output_dir": str(output_dir),
        "detections_per_frame_summary": summarize_numeric(detection_counts),
        "detection_count_slope_per_frame": detection_slope,
        "phase_correction_summary": summarize_numeric(corrections),
        "phase_correction_slope_px_per_frame": correction_slope,
        "registration_loss_summary": summarize_numeric(losses),
        "registration_loss_slope_per_frame": loss_slope,
        "warnings": warnings,
    }


def plan_map_epochs(
    frame_count: int,
    *,
    epoch_count: int | None = None,
    epoch_length_frames: int | None = None,
    overlap_frames: int = 0,
) -> list[dict[str, int | str]]:
    """Plan frame ranges for multi-epoch belt maps."""

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if overlap_frames < 0:
        raise ValueError("overlap_frames must be non-negative")
    if epoch_count is None and epoch_length_frames is None:
        epoch_count = 1
    if epoch_length_frames is None:
        assert epoch_count is not None
        if epoch_count <= 0:
            raise ValueError("epoch_count must be positive")
        epoch_length_frames = int(math.ceil(frame_count / epoch_count))
    if epoch_length_frames <= 0:
        raise ValueError("epoch_length_frames must be positive")

    epochs: list[dict[str, int | str]] = []
    start = 0
    epoch_id = 0
    while start < frame_count:
        stop = min(frame_count, start + epoch_length_frames)
        train_start = max(0, start - overlap_frames)
        train_stop = min(frame_count, stop + overlap_frames)
        epochs.append(
            {
                "epoch": epoch_id,
                "name": f"epoch_{epoch_id:03d}",
                "frame_start": start,
                "frame_stop": stop,
                "train_frame_start": train_start,
                "train_frame_stop": train_stop,
            }
        )
        epoch_id += 1
        start = stop
    return epochs


def edge_audit_rows(
    detections: Sequence[Mapping[str, Any]],
    *,
    height: int,
    width: int,
    margin_px: int = 0,
) -> list[dict[str, Any]]:
    """Annotate detections with crop-edge and truncation flags."""

    rows: list[dict[str, Any]] = []
    for row in detections:
        top = finite_float(row.get("bbox_top"))
        left = finite_float(row.get("bbox_left"))
        bottom = finite_float(row.get("bbox_bottom"))
        right = finite_float(row.get("bbox_right"))
        if None in (top, left, bottom, right):
            continue
        assert top is not None and left is not None and bottom is not None and right is not None
        touches_top = top <= margin_px
        touches_left = left <= margin_px
        touches_bottom = bottom >= height - margin_px
        touches_right = right >= width - margin_px
        annotated = dict(row)
        annotated.update(
            {
                "touches_top_edge": touches_top,
                "touches_bottom_edge": touches_bottom,
                "touches_left_edge": touches_left,
                "touches_right_edge": touches_right,
                "is_truncated": touches_top or touches_bottom or touches_left or touches_right,
            }
        )
        rows.append(annotated)
    return rows


def events_from_tracks(track_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate track detection rows into event-level rows."""

    by_track: dict[int, list[Mapping[str, Any]]] = {}
    for row in track_rows:
        track_id = finite_int(row.get("track_id"))
        if track_id is None:
            continue
        by_track.setdefault(track_id, []).append(row)

    events: list[dict[str, Any]] = []
    for event_id, (track_id, rows) in enumerate(sorted(by_track.items())):
        frames = [finite_float(row.get("frame_index")) for row in rows]
        ys = [finite_float(row.get("y")) for row in rows]
        xs = [finite_float(row.get("x")) for row in rows]
        peaks = [finite_float(row.get("peak_signal")) for row in rows]
        frames_f = [value for value in frames if value is not None]
        ys_f = [value for value in ys if value is not None]
        xs_f = [value for value in xs if value is not None]
        peaks_f = [value for value in peaks if value is not None]
        velocity_y = _linear_slope(frames_f, ys_f) if len(frames_f) == len(ys_f) else None
        velocity_x = _linear_slope(frames_f, xs_f) if len(frames_f) == len(xs_f) else None
        events.append(
            {
                "event_id": event_id,
                "track_id": track_id,
                "first_frame": min(frames_f) if frames_f else "",
                "last_frame": max(frames_f) if frames_f else "",
                "n_observations": len(rows),
                "representative_y": float(np.median(ys_f)) if ys_f else "",
                "representative_x": float(np.median(xs_f)) if xs_f else "",
                "peak_signal_median": float(np.median(peaks_f)) if peaks_f else "",
                "velocity_y_px_per_frame": "" if velocity_y is None else velocity_y,
                "velocity_x_px_per_frame": "" if velocity_x is None else velocity_x,
            }
        )
    return events


def _logistic(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def confidence_rows(
    detections: Sequence[Mapping[str, Any]],
    *,
    threshold: float = 5.0,
) -> list[dict[str, Any]]:
    """Attach a simple confidence score to detection-like rows."""

    rows: list[dict[str, Any]] = []
    for row in detections:
        peak = finite_float(row.get("peak_signal"))
        mean = finite_float(row.get("mean_signal"))
        area = finite_float(row.get("area_px"))
        overlap = finite_float(row.get("recurrent_artifact_overlap_fraction")) or 0.0
        truncated = str(row.get("is_truncated", "False")).lower() in {"1", "true", "yes"}
        signal = peak if peak is not None else mean
        signal_score = 0.5 if signal is None else _logistic((signal - threshold) / max(threshold, 1e-6))
        area_score = 0.5 if area is None else min(1.0, max(0.0, area / 20.0))
        artifact_penalty = max(0.0, 1.0 - overlap)
        truncation_penalty = 0.75 if truncated else 1.0
        confidence = signal_score * (0.5 + 0.5 * area_score) * artifact_penalty * truncation_penalty
        annotated = dict(row)
        annotated["detection_confidence"] = float(max(0.0, min(1.0, confidence)))
        rows.append(annotated)
    return rows


def physical_validation_summary(
    output_dir: Path,
    *,
    expected_particle_flux_per_s: float | None = None,
    expected_mass_flux_g_s: float | None = None,
    particle_mass_g: float | None = None,
    frame_rate_hz: float | None = None,
    analysis_duration_s: float | None = None,
) -> dict[str, Any]:
    """Compare image-derived counts or mass flux with independent measurements."""

    metadata = read_json(output_dir / "metadata.json")
    events = read_csv_rows(output_dir / "events.csv")
    if not events:
        events = events_from_tracks(read_csv_rows(output_dir / "filtered_tracks.csv"))
    detections = read_csv_rows(output_dir / "detections.csv")

    n_events = len(events)
    if n_events == 0:
        n_events = len(detections)
    duration = analysis_duration_s
    if duration is None and frame_rate_hz is not None and frame_rate_hz > 0:
        n_images = finite_float(metadata.get("n_images"))
        if n_images is not None:
            duration = n_images / frame_rate_hz

    image_flux = None if not duration or duration <= 0 else n_events / duration
    estimated_mass_flux = (
        None
        if image_flux is None or particle_mass_g is None
        else image_flux * particle_mass_g
    )

    flux_error = None
    if image_flux is not None and expected_particle_flux_per_s not in (None, 0):
        assert expected_particle_flux_per_s is not None
        flux_error = (image_flux - expected_particle_flux_per_s) / expected_particle_flux_per_s

    mass_error = None
    if estimated_mass_flux is not None and expected_mass_flux_g_s not in (None, 0):
        assert expected_mass_flux_g_s is not None
        mass_error = (estimated_mass_flux - expected_mass_flux_g_s) / expected_mass_flux_g_s

    warnings: list[str] = []
    if flux_error is not None and abs(flux_error) > 0.25:
        warnings.append("image-derived particle flux differs from expected flux by more than 25%")
    if mass_error is not None and abs(mass_error) > 0.25:
        warnings.append("estimated mass flux differs from expected mass flux by more than 25%")
    if duration is None:
        warnings.append("duration unavailable; provide frame_rate_hz or analysis_duration_s for flux validation")

    return {
        "output_dir": str(output_dir),
        "event_count": n_events,
        "duration_s": duration,
        "image_particle_flux_per_s": image_flux,
        "expected_particle_flux_per_s": expected_particle_flux_per_s,
        "relative_particle_flux_error": flux_error,
        "particle_mass_g": particle_mass_g,
        "estimated_mass_flux_g_s": estimated_mass_flux,
        "expected_mass_flux_g_s": expected_mass_flux_g_s,
        "relative_mass_flux_error": mass_error,
        "warnings": warnings,
    }


def rejection_audit_rows(output_dir: Path) -> list[dict[str, Any]]:
    """Collect rejection reasons from available recurrent-artifact and track-filter outputs."""

    rows: list[dict[str, Any]] = []
    for row in read_csv_rows(output_dir / "recurrent_artifact_detections.csv"):
        rejected = str(row.get("recurrent_artifact_rejected", "")).lower() in {"true", "1", "yes"}
        if rejected:
            rows.append(
                {
                    "source": "recurrent_artifact_detections.csv",
                    "candidate_id": len(rows),
                    "frame_index": row.get("frame_index", ""),
                    "track_id": "",
                    "reason_rejected": "recurrent_artifact",
                    "failed_area_gate": False,
                    "failed_shape_gate": False,
                    "failed_recurrent_artifact_gate": True,
                    "failed_track_gate": False,
                    "failed_velocity_gate": False,
                }
            )

    for row in read_csv_rows(output_dir / "track_scores.csv"):
        accepted = str(row.get("accepted", "")).lower() in {"true", "1", "yes"}
        if accepted:
            continue
        passes_length = str(row.get("passes_min_track_length", "")).lower() in {"true", "1", "yes"}
        passes_ratio = str(row.get("passes_velocity_ratio", "")).lower() in {"true", "1", "yes"}
        passes_lateral = str(row.get("passes_lateral_velocity", "")).lower() in {"true", "1", "yes"}
        reasons = []
        if not passes_length:
            reasons.append("track_length")
        if not passes_ratio:
            reasons.append("velocity_ratio")
        if not passes_lateral:
            reasons.append("lateral_velocity")
        rows.append(
            {
                "source": "track_scores.csv",
                "candidate_id": len(rows),
                "frame_index": "",
                "track_id": row.get("track_id", ""),
                "reason_rejected": "+".join(reasons) if reasons else "track_filter",
                "failed_area_gate": False,
                "failed_shape_gate": False,
                "failed_recurrent_artifact_gate": False,
                "failed_track_gate": not passes_length,
                "failed_velocity_gate": (not passes_ratio) or (not passes_lateral),
            }
        )
    return rows


def scale_calibration_from_points(
    point_a: Sequence[float],
    point_b: Sequence[float],
    *,
    known_distance_mm: float,
) -> ScaleCalibration:
    """Estimate pixel scale from two calibration-target points."""

    if known_distance_mm <= 0:
        raise ValueError("known_distance_mm must be positive")
    if len(point_a) != 2 or len(point_b) != 2:
        raise ValueError("point_a and point_b must contain two values each")
    ay, ax = float(point_a[0]), float(point_a[1])
    by, bx = float(point_b[0]), float(point_b[1])
    distance_px = math.hypot(by - ay, bx - ax)
    if distance_px <= 0:
        raise ValueError("calibration points must be distinct")
    return ScaleCalibration(
        px_per_mm=distance_px / known_distance_mm,
        point_a=(ay, ax),
        point_b=(by, bx),
        known_distance_mm=float(known_distance_mm),
    )


def compare_run_metadata(output_dirs: Sequence[Path]) -> dict[str, Any]:
    """Check whether several BeltMap runs are comparable."""

    keys = [
        "first_image_shape",
        "belt_region",
        "belt_velocity_px_per_frame",
        "belt_map_height_px",
        "frame_stride",
        "detection_threshold",
        "min_area_px",
        "phase_estimate_source",
    ]
    rows: list[dict[str, Any]] = []
    for output_dir in output_dirs:
        metadata = read_json(output_dir / "metadata.json")
        row = {"output_dir": str(output_dir)}
        for key in keys:
            row[key] = metadata.get(key)
        rows.append(row)

    differences: dict[str, list[Any]] = {}
    for key in keys:
        values = [_stable_repr(row.get(key)) for row in rows]
        if len(set(values)) > 1:
            differences[key] = [row.get(key) for row in rows]

    warnings = [
        f"runs differ in {key}"
        for key in sorted(differences)
    ]
    return {
        "runs": rows,
        "differences": differences,
        "comparable": not differences,
        "warnings": warnings,
    }


def write_profile(name: str, path: Path) -> None:
    """Write a named cost-sensitive tuning profile as a TOML overlay."""

    if name not in PROFILE_CONFIGS:
        choices = ", ".join(sorted(PROFILE_CONFIGS))
        raise ValueError(f"unknown profile {name!r}; choose one of {choices}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_toml_from_nested(PROFILE_CONFIGS[name]), encoding="utf-8")


def write_run_trust_artifacts(
    *,
    output_dir: Path,
    image_dir: Path | None = None,
    region: tuple[int, int, int, int] | None = None,
    frame_rate_hz: float | None = None,
    expected_particle_flux_per_s: float | None = None,
    expected_mass_flux_g_s: float | None = None,
    particle_mass_g: float | None = None,
    epoch_count: int | None = None,
) -> dict[str, Path]:
    """Write the trust/QC artifact bundle for one BeltMap output directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}

    metadata = read_json(output_dir / "metadata.json")
    if region is None:
        belt_region = metadata.get("belt_region")
        if isinstance(belt_region, dict):
            region = (
                int(belt_region["top"]),
                int(belt_region["left"]),
                int(belt_region["height"]),
                int(belt_region["width"]),
            )

    if image_dir is not None:
        sequence = sequence_report(image_dir)
        path = output_dir / "sequence_qc.json"
        write_json(path, sequence)
        artifacts["sequence_qc"] = path

        quality = quality_report(image_dir, region=region)
        path = output_dir / "frame_quality_summary.json"
        write_json(path, quality)
        artifacts["frame_quality"] = path

    speed = speed_consistency_report(output_dir)
    path = output_dir / "speed_consistency.json"
    write_json(path, speed)
    artifacts["speed_consistency"] = path

    drift = run_drift_report(output_dir)
    path = output_dir / "run_drift_summary.json"
    write_json(path, drift)
    artifacts["run_drift"] = path

    physical = physical_validation_summary(
        output_dir,
        expected_particle_flux_per_s=expected_particle_flux_per_s,
        expected_mass_flux_g_s=expected_mass_flux_g_s,
        particle_mass_g=particle_mass_g,
        frame_rate_hz=frame_rate_hz,
    )
    path = output_dir / "physical_validation.json"
    write_json(path, physical)
    artifacts["physical_validation"] = path

    detections = read_csv_rows(output_dir / "detections.csv")
    height, width = _metadata_crop_shape(metadata)
    if detections and height > 0 and width > 0:
        edge_rows = edge_audit_rows(detections, height=height, width=width)
        path = output_dir / "detection_edge_audit.csv"
        write_csv_rows(path, edge_rows, list(edge_rows[0].keys()) if edge_rows else [])
        artifacts["detection_edge_audit"] = path

        conf_rows = confidence_rows(
            edge_rows,
            threshold=finite_float_or(metadata.get("detection_threshold"), 5.0),
        )
        path = output_dir / "detection_confidence.csv"
        write_csv_rows(path, conf_rows, list(conf_rows[0].keys()) if conf_rows else [])
        artifacts["detection_confidence"] = path

    track_rows = read_csv_rows(output_dir / "filtered_tracks.csv") or read_csv_rows(output_dir / "tracks.csv")
    if track_rows:
        events = events_from_tracks(track_rows)
        path = output_dir / "events.csv"
        write_csv_rows(path, events, list(events[0].keys()) if events else [])
        artifacts["events"] = path

    rejection_rows = rejection_audit_rows(output_dir)
    path = output_dir / "rejection_audit.csv"
    write_csv_rows(
        path,
        rejection_rows,
        [
            "source",
            "candidate_id",
            "frame_index",
            "track_id",
            "reason_rejected",
            "failed_area_gate",
            "failed_shape_gate",
            "failed_recurrent_artifact_gate",
            "failed_track_gate",
            "failed_velocity_gate",
        ],
    )
    artifacts["rejection_audit"] = path

    n_images = finite_int(metadata.get("n_images"))
    if n_images is not None and n_images > 0:
        epochs = plan_map_epochs(n_images, epoch_count=epoch_count or 1)
        path = output_dir / "map_epoch_plan.csv"
        write_csv_rows(path, epochs, list(epochs[0].keys()) if epochs else [])
        artifacts["map_epoch_plan"] = path

    report_path = output_dir / "minimum_evidence_report.md"
    report_path.write_text(minimum_evidence_report(output_dir), encoding="utf-8")
    artifacts["minimum_evidence_report"] = report_path

    artifact_index = output_dir / "trust_artifacts.json"
    write_json(artifact_index, {key: str(path) for key, path in artifacts.items()})
    artifacts["trust_artifacts"] = artifact_index
    return artifacts


def minimum_evidence_report(output_dir: Path) -> str:
    """Create a compact publication-oriented evidence report."""

    metadata = read_json(output_dir / "metadata.json")
    validation = read_json(output_dir / "validation_summary.json")
    speed = read_json(output_dir / "speed_consistency.json")
    drift = read_json(output_dir / "run_drift_summary.json")
    physical = read_json(output_dir / "physical_validation.json")
    quality = read_json(output_dir / "frame_quality_summary.json")
    sequence = read_json(output_dir / "sequence_qc.json")
    rejection_rows = read_csv_rows(output_dir / "rejection_audit.csv")
    events = read_csv_rows(output_dir / "events.csv")
    confidence = read_csv_rows(output_dir / "detection_confidence.csv")

    lines = [
        "# BeltMap minimum evidence report",
        "",
        "## Run identity",
        "",
        f"- Output directory: `{output_dir}`",
        f"- Images processed: {metadata.get('n_images', 'n/a')}",
        f"- Belt velocity px/frame: {metadata.get('belt_velocity_px_per_frame', 'n/a')}",
        f"- Belt map height px: {metadata.get('belt_map_height_px', 'n/a')}",
        f"- Detection threshold: {metadata.get('detection_threshold', 'n/a')}",
        "",
        "## Preflight and image quality",
        "",
        f"- Sequence warnings: {', '.join(sequence.get('warnings', [])) or 'none recorded'}",
        f"- Frame-quality warnings: {', '.join(quality.get('warnings', [])) or 'none recorded'}",
        "",
        "## Phase and speed stability",
        "",
        f"- Correction slope px/frame: {speed.get('correction_slope_px_per_frame', 'n/a')}",
        f"- Inferred belt velocity px/frame: {speed.get('inferred_belt_velocity_px_per_frame', 'n/a')}",
        f"- Speed warnings: {', '.join(speed.get('warnings', [])) or 'none recorded'}",
        "",
        "## Drift and map staleness",
        "",
        f"- Detection-count slope/frame: {drift.get('detection_count_slope_per_frame', 'n/a')}",
        f"- Drift warnings: {', '.join(drift.get('warnings', [])) or 'none recorded'}",
        "",
        "## Detections, tracks, and events",
        "",
        f"- Raw detections: {metadata.get('n_detections', 'n/a')}",
        f"- Tracks: {metadata.get('n_tracks', 'n/a')}",
        f"- Filtered velocity estimates: {metadata.get('n_filtered_velocity_estimates', 'n/a')}",
        f"- Event rows: {len(events)}",
        f"- Rejection rows: {len(rejection_rows)}",
        f"- Detection confidence rows: {len(confidence)}",
        "",
        "## Physical cross-check",
        "",
        f"- Image particle flux / s: {physical.get('image_particle_flux_per_s', 'n/a')}",
        f"- Expected particle flux / s: {physical.get('expected_particle_flux_per_s', 'n/a')}",
        f"- Relative particle-flux error: {physical.get('relative_particle_flux_error', 'n/a')}",
        f"- Estimated mass flux g/s: {physical.get('estimated_mass_flux_g_s', 'n/a')}",
        f"- Expected mass flux g/s: {physical.get('expected_mass_flux_g_s', 'n/a')}",
        f"- Physical-validation warnings: {', '.join(physical.get('warnings', [])) or 'none recorded'}",
        "",
        "## Existing validation summary",
        "",
        f"- Validation keys available: {', '.join(sorted(validation.keys())) if validation else 'none'}",
        "",
    ]
    return "\n".join(lines)


def _metadata_crop_shape(metadata: Mapping[str, Any]) -> tuple[int, int]:
    region = metadata.get("belt_region")
    if isinstance(region, dict):
        height = finite_int(region.get("height")) or 0
        width = finite_int(region.get("width")) or 0
        return height, width
    shape = metadata.get("first_image_shape")
    if isinstance(shape, list) and len(shape) >= 2:
        return int(shape[0]), int(shape[1])
    return 0, 0


def _stable_repr(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, ScaleCalibration):
        return {
            "px_per_mm": value.px_per_mm,
            "mm_per_px": value.mm_per_px,
            "point_a": list(value.point_a),
            "point_b": list(value.point_b),
            "known_distance_mm": value.known_distance_mm,
        }
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _toml_from_nested(config: Mapping[str, Mapping[str, object]]) -> str:
    lines: list[str] = []
    for section, values in config.items():
        lines.append(f"[{section}]")
        for key, value in values.items():
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines)


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value))
