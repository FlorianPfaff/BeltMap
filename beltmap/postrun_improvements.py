"""Post-run result-improvement helpers for BeltMap.

This module collects opt-in utilities that improve the *operational* quality loop
without changing the default image driver.  The helpers consume standard BeltMap
outputs and write additional diagnostics, confidence scores, frame-selection
plans, and quality-contract reports.  They are intentionally NumPy/Pillow-only so
that they can be used from CI, parameter sweeps, and manual review workflows.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from PIL import Image

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class MetricSummary:
    """A compact scalar metric with an optional pass/fail state."""

    name: str
    value: float | int | None
    threshold: float | int | None = None
    passed: bool | None = None
    unit: str = ""
    description: str = ""


@dataclass(frozen=True)
class QualityFlag:
    """Machine-readable warning emitted by post-run auditing."""

    severity: str
    code: str
    message: str
    value: float | int | None = None
    threshold: float | int | None = None


@dataclass(frozen=True)
class ContractResult:
    """Evaluation result for one quality-contract check."""

    name: str
    passed: bool
    value: float | int | None
    threshold: float | int | None
    comparison: str
    description: str = ""


DEFAULT_QUALITY_CONTRACT: dict[str, Any] = {
    "max_registration_boundary_share": 0.05,
    "max_many_tiny_component_share": 0.50,
    "min_velocity_ratio_in_range_share": 0.50,
    "max_recurrent_rejection_share": 0.75,
    "min_filtered_velocity_estimates": 1,
    "velocity_ratio_y_range": [0.0, 1.1],
}

SYNTHETIC_REGRESSION_CONTRACT_TEMPLATE: dict[str, Any] = {
    "phase_mae_px_max": 1.0,
    "belt_map_rmse_gray_max": 8.0,
    "detection_f1_min": 0.75,
    "velocity_mae_px_per_frame_max": 5.0,
    "registration_boundary_share_max": 0.05,
    "runtime_s_max": None,
    "peak_memory_mb_max": None,
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into dictionaries, returning an empty list if missing."""

    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    """Write rows to a CSV file, preserving the supplied field order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def finite_int(value: Any) -> int | None:
    parsed = finite_float(value)
    return None if parsed is None else int(parsed)


def load_metadata(output_dir: Path) -> dict[str, Any]:
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.is_file():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def normalized_png(path: Path, image: ArrayLike) -> None:
    """Save an array as a linearly normalized 8-bit PNG."""

    arr = np.asarray(image, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        scaled = np.zeros(arr.shape, dtype=np.uint8)
    else:
        lo = float(np.percentile(finite, 1))
        hi = float(np.percentile(finite, 99))
        if hi <= lo:
            hi = lo + 1.0
        scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
        scaled = (255.0 * scaled).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(scaled).save(path)


def phase_rows(output_dir: Path) -> list[dict[str, str]]:
    return read_csv_rows(output_dir / "phase_estimates.csv")


def detection_count_by_frame(output_dir: Path) -> dict[int, int]:
    rows = read_csv_rows(output_dir / "detections_per_frame.csv")
    result: dict[int, int] = {}
    for row in rows:
        frame = finite_int(row.get("frame_index"))
        count = finite_int(row.get("n_detections"))
        if frame is not None:
            result[frame] = 0 if count is None else count
    return result


def load_detection_rows(output_dir: Path) -> list[dict[str, str]]:
    return read_csv_rows(output_dir / "detections.csv")


def map_shape_from_outputs(output_dir: Path, metadata: Mapping[str, Any] | None = None) -> tuple[int, int] | None:
    metadata = load_metadata(output_dir) if metadata is None else metadata
    map_height = finite_int(metadata.get("belt_map_height_px"))
    region = metadata.get("belt_region")
    width = None
    if isinstance(region, Mapping):
        width = finite_int(region.get("width"))
    if map_height is not None and width is not None:
        return map_height, width
    belt_map_path = output_dir / "belt_map.npy"
    if belt_map_path.is_file():
        belt_map = np.load(belt_map_path, mmap_mode="r")
        if belt_map.ndim == 2:
            return int(belt_map.shape[0]), int(belt_map.shape[1])
    return None


def crop_height_from_metadata(metadata: Mapping[str, Any]) -> int | None:
    region = metadata.get("belt_region")
    if isinstance(region, Mapping):
        return finite_int(region.get("height"))
    return None


def compute_phase_row_counts(
    phases_px: Sequence[float],
    *,
    map_height: int,
    crop_height: int,
) -> NDArray[np.uint32]:
    """Approximate belt-coordinate row exposure counts from per-frame phases."""

    if map_height <= 0:
        raise ValueError("map_height must be positive")
    if crop_height <= 0:
        raise ValueError("crop_height must be positive")
    counts = np.zeros(map_height, dtype=np.uint32)
    image_rows = np.arange(crop_height, dtype=np.float64)
    for phase in phases_px:
        if not np.isfinite(phase):
            continue
        rows = np.mod(np.floor(image_rows + float(phase)).astype(np.int64), map_height)
        counts += np.bincount(rows, minlength=map_height).astype(np.uint32)
    return counts


def uncertainty_from_counts(
    counts: ArrayLike,
    *,
    scale: float = 1.0,
    min_count: float = 1.0,
) -> FloatArray:
    """Convert observation counts to an uncertainty floor proportional to 1/sqrt(N)."""

    if scale <= 0:
        raise ValueError("scale must be positive")
    if min_count <= 0:
        raise ValueError("min_count must be positive")
    arr = np.asarray(counts, dtype=np.float64)
    safe = np.maximum(arr, min_count)
    uncertainty = scale / np.sqrt(safe)
    uncertainty[~np.isfinite(arr) | (arr <= 0)] = scale
    return uncertainty


def write_map_uncertainty_outputs(
    output_dir: Path,
    *,
    report_dir: Path | None = None,
    scale: float = 1.0,
    write_full_counts: bool = False,
) -> dict[str, Any]:
    """Write row-level and optional full map-coverage/uncertainty outputs."""

    output_dir = Path(output_dir)
    report_dir = output_dir if report_dir is None else Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(output_dir)
    shape = map_shape_from_outputs(output_dir, metadata)
    crop_height = crop_height_from_metadata(metadata)
    phases = [finite_float(row.get("phase_px")) for row in phase_rows(output_dir)]
    phase_values = [value for value in phases if value is not None]
    if shape is None or crop_height is None or not phase_values:
        summary = {
            "available": False,
            "reason": "requires metadata.json with belt map shape/crop height and phase_estimates.csv",
        }
        write_json(report_dir / "map_uncertainty_summary.json", summary)
        return summary

    map_height, map_width = shape
    row_counts = compute_phase_row_counts(
        phase_values,
        map_height=map_height,
        crop_height=crop_height,
    )
    row_uncertainty = uncertainty_from_counts(row_counts, scale=scale)
    np.save(report_dir / "belt_map_row_counts.npy", row_counts)
    np.save(report_dir / "belt_map_row_uncertainty.npy", row_uncertainty.astype(np.float32))
    normalized_png(report_dir / "belt_map_row_counts.png", row_counts[:, None])
    normalized_png(report_dir / "belt_map_row_uncertainty.png", row_uncertainty[:, None])

    if write_full_counts:
        full_counts = np.repeat(row_counts[:, None], map_width, axis=1)
        full_uncertainty = np.repeat(row_uncertainty[:, None], map_width, axis=1).astype(np.float32)
        np.save(report_dir / "belt_map_counts.npy", full_counts)
        np.save(report_dir / "belt_map_uncertainty.npy", full_uncertainty)
        normalized_png(report_dir / "belt_map_coverage.png", full_counts)
        normalized_png(report_dir / "belt_map_uncertainty.png", full_uncertainty)

    finite_counts = row_counts[row_counts > 0]
    summary = {
        "available": True,
        "map_height": map_height,
        "map_width": map_width,
        "crop_height": crop_height,
        "phase_estimates": len(phase_values),
        "row_count_min": int(np.min(finite_counts)) if finite_counts.size else 0,
        "row_count_median": float(np.median(finite_counts)) if finite_counts.size else 0.0,
        "row_count_p05": float(np.percentile(finite_counts, 5)) if finite_counts.size else 0.0,
        "unobserved_rows": int(np.count_nonzero(row_counts == 0)),
        "scale": scale,
        "wrote_full_counts": write_full_counts,
    }
    write_json(report_dir / "map_uncertainty_summary.json", summary)
    return summary


def seam_discontinuity_profile(
    belt_map: ArrayLike,
    *,
    window_px: int = 8,
) -> dict[str, Any]:
    """Measure seam discontinuity and suggest a lower-discontinuity roll row."""

    belt = np.asarray(belt_map, dtype=np.float64)
    if belt.ndim != 2:
        raise ValueError("belt_map must be 2-D")
    if window_px < 1:
        raise ValueError("window_px must be positive")
    jumps = _seam_discontinuity_jumps(belt, window_px=window_px)
    best_row = int(np.argmin(jumps))
    current_jump = float(jumps[0])
    best_jump = float(jumps[best_row])
    return {
        "current_seam_row": 0,
        "current_mean_abs_jump_gray": current_jump,
        "best_roll_row": best_row,
        "best_mean_abs_jump_gray": best_jump,
        "relative_improvement": None if current_jump <= 0 else float((current_jump - best_jump) / current_jump),
        "p95_row_jump_gray": float(np.percentile(jumps, 95)),
        "median_row_jump_gray": float(np.median(jumps)),
    }


def _seam_discontinuity_jumps(belt: FloatArray, *, window_px: int) -> FloatArray:
    """Return candidate seam costs using a symmetric row window around each seam."""

    height = belt.shape[0]
    if height <= 0:
        raise ValueError("belt_map must have at least one row")
    window = max(1, min(int(window_px), max(1, height // 2)))
    before_offsets = np.arange(window, 0, -1, dtype=np.int64)
    after_offsets = np.arange(window, dtype=np.int64)
    jumps = np.empty(height, dtype=np.float64)
    for row in range(height):
        before_rows = (row - before_offsets) % height
        after_rows = (row + after_offsets) % height
        jumps[row] = float(np.mean(np.abs(belt[before_rows] - belt[after_rows])))
    return jumps


def write_seam_diagnostics(output_dir: Path, *, report_dir: Path | None = None) -> dict[str, Any]:
    output_dir = Path(output_dir)
    report_dir = output_dir if report_dir is None else Path(report_dir)
    belt_map_path = output_dir / "belt_map.npy"
    if not belt_map_path.is_file():
        result = {"available": False, "reason": "belt_map.npy is missing"}
        write_json(report_dir / "seam_diagnostics.json", result)
        return result
    belt_map = np.load(belt_map_path)
    result = {"available": True, **seam_discontinuity_profile(belt_map)}
    write_json(report_dir / "seam_diagnostics.json", result)
    jumps = _seam_discontinuity_jumps(
        np.asarray(belt_map, dtype=np.float64),
        window_px=8,
    )
    normalized_png(report_dir / "seam_discontinuity_profile.png", jumps[:, None])
    return result


def _frame_metric_rows(mapping: Mapping[int, float | int], *, metric: str, reverse: bool, top_n: int) -> list[dict[str, Any]]:
    rows = [
        {"frame_index": frame, "metric": metric, "value": value}
        for frame, value in mapping.items()
        if value is not None and np.isfinite(float(value))
    ]
    rows.sort(key=lambda row: float(row["value"]), reverse=reverse)
    return rows[:top_n]


def worst_frame_tables(output_dir: Path, *, top_n: int = 10) -> dict[str, list[dict[str, Any]]]:
    """Rank frames by common failure-mode proxy metrics."""

    output_dir = Path(output_dir)
    phase = phase_rows(output_dir)
    low_score: dict[int, float] = {}
    abs_correction: dict[int, float] = {}
    for row in phase:
        frame = finite_int(row.get("frame_index"))
        if frame is None:
            continue
        score = finite_float(row.get("score"))
        correction = finite_float(row.get("correction_px"))
        if score is not None:
            low_score[frame] = score
        if correction is not None:
            abs_correction[frame] = abs(correction)

    detection_counts = detection_count_by_frame(output_dir)
    recurrent_rows = read_csv_rows(output_dir / "recurrent_artifact_detections.csv")
    rejected_by_frame: dict[int, int] = {}
    for row in recurrent_rows:
        frame = finite_int(row.get("frame_index"))
        rejected = str(row.get("recurrent_artifact_rejected", "")).strip().lower() in {"1", "true", "yes"}
        if frame is not None and rejected:
            rejected_by_frame[frame] = rejected_by_frame.get(frame, 0) + 1

    photometric_rows = read_csv_rows(output_dir / "photometric_fits.csv")
    rmse_by_frame: dict[int, float] = {}
    for row in photometric_rows:
        frame = finite_int(row.get("frame_index"))
        rmse = finite_float(row.get("rmse_gray"))
        if frame is not None and rmse is not None:
            rmse_by_frame[frame] = rmse

    tables = {
        "low_registration_score": _frame_metric_rows(low_score, metric="registration_score", reverse=False, top_n=top_n),
        "large_phase_correction": _frame_metric_rows(abs_correction, metric="abs_correction_px", reverse=True, top_n=top_n),
        "detection_spikes": _frame_metric_rows(detection_counts, metric="n_detections", reverse=True, top_n=top_n),
        "empty_or_low_detection_frames": _frame_metric_rows(detection_counts, metric="n_detections", reverse=False, top_n=top_n),
        "recurrent_rejections": _frame_metric_rows(rejected_by_frame, metric="recurrent_rejected", reverse=True, top_n=top_n),
        "photometric_rmse": _frame_metric_rows(rmse_by_frame, metric="photometric_rmse_gray", reverse=True, top_n=top_n),
    }
    return tables


def write_worst_frame_tables(output_dir: Path, *, report_dir: Path, top_n: int = 10) -> dict[str, list[dict[str, Any]]]:
    tables = worst_frame_tables(output_dir, top_n=top_n)
    for name, rows in tables.items():
        write_csv(
            report_dir / f"worst_{name}.csv",
            rows,
            ["frame_index", "metric", "value"],
        )
    write_json(report_dir / "worst_frames.json", tables)
    return tables


def quality_flags_from_outputs(output_dir: Path) -> list[QualityFlag]:
    """Return warnings for common poor-result modes."""

    output_dir = Path(output_dir)
    metadata = load_metadata(output_dir)
    flags: list[QualityFlag] = []

    corrections = np.asarray([
        value for value in (finite_float(row.get("correction_px")) for row in phase_rows(output_dir))
        if value is not None
    ], dtype=np.float64)
    search_radius = finite_float(metadata.get("registration_search_radius_px")) or 8.0
    search_step = finite_float(metadata.get("registration_search_step_px")) or 1.0
    boundary_tolerance = max(1e-9, 0.5 * search_step)
    if corrections.size:
        boundary_share = float(np.mean(np.abs(np.abs(corrections) - search_radius) <= boundary_tolerance))
        if boundary_share > 0.05:
            flags.append(QualityFlag("warning", "registration_boundary", "phase corrections often hit the search boundary", boundary_share, 0.05))

    counts = np.asarray(list(detection_count_by_frame(output_dir).values()), dtype=np.float64)
    if counts.size:
        median = float(np.median(counts))
        p95 = float(np.percentile(counts, 95))
        if p95 > max(25.0, 5.0 * (median + 1.0)):
            flags.append(QualityFlag("warning", "detection_spikes", "detection counts have large frame-to-frame spikes", p95, max(25.0, 5.0 * (median + 1.0))))

    detections = load_detection_rows(output_dir)
    areas = np.asarray([
        value for value in (finite_float(row.get("area_px")) for row in detections)
        if value is not None
    ], dtype=np.float64)
    if areas.size:
        tiny_share = float(np.mean(areas <= 2.0))
        if tiny_share > 0.5:
            flags.append(QualityFlag("warning", "many_tiny_components", "most detections are tiny components", tiny_share, 0.5))

    velocities = read_csv_rows(output_dir / "velocities.csv")
    ratios = np.asarray([
        value for value in (finite_float(row.get("velocity_ratio_y")) for row in velocities)
        if value is not None
    ], dtype=np.float64)
    if ratios.size:
        in_range = float(np.mean((0.0 <= ratios) & (ratios <= 1.1)))
        if in_range < 0.5:
            flags.append(QualityFlag("warning", "implausible_velocity_ratios", "many velocity ratios are outside [0, 1.1]", in_range, 0.5))

    recurrent_rejected = finite_int(metadata.get("n_recurrent_artifact_rejected")) or 0
    n_detections = finite_int(metadata.get("n_detections")) or len(detections)
    denom = recurrent_rejected + n_detections
    if denom > 0:
        rejected_share = recurrent_rejected / denom
        if rejected_share > 0.75:
            flags.append(QualityFlag("info", "heavy_recurrent_filtering", "recurrent artifact filtering rejected most first-pass detections", rejected_share, 0.75))

    photometric_rows = read_csv_rows(output_dir / "photometric_fits.csv")
    gains = np.asarray([value for value in (finite_float(row.get("gain")) for row in photometric_rows) if value is not None], dtype=np.float64)
    if gains.size:
        extreme_gain_share = float(np.mean((gains < 0.75) | (gains > 1.25)))
        if extreme_gain_share > 0.1:
            flags.append(QualityFlag("warning", "photometric_gain_drift", "many frames required large photometric gain correction", extreme_gain_share, 0.1))
    return flags


def write_quality_flags(output_dir: Path, *, report_dir: Path) -> list[QualityFlag]:
    flags = quality_flags_from_outputs(output_dir)
    write_json(report_dir / "quality_flags.json", [asdict(flag) for flag in flags])
    lines = ["# Quality flags", ""]
    if not flags:
        lines.append("No quality flags were triggered.")
    for flag in flags:
        suffix = "" if flag.value is None else f" value={flag.value:.4g}"
        lines.append(f"- **{flag.severity} / {flag.code}**: {flag.message}{suffix}")
    (report_dir / "quality_flags.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return flags


def flux_summary(output_dir: Path, *, frame_rate_hz: float | None = None) -> dict[str, Any]:
    """Summarize accepted particle-track and velocity outputs."""

    output_dir = Path(output_dir)
    rows = read_csv_rows(output_dir / "filtered_velocities.csv") or read_csv_rows(output_dir / "velocities.csv")
    ratios = np.asarray([value for value in (finite_float(row.get("velocity_ratio_y")) for row in rows) if value is not None], dtype=np.float64)
    speeds = np.asarray([value for value in (finite_float(row.get("speed_px_per_frame")) for row in rows) if value is not None], dtype=np.float64)
    metadata = load_metadata(output_dir)
    n_images = finite_int(metadata.get("n_images")) or 0
    duration_s = None
    tracks_per_second = None
    if frame_rate_hz is not None and frame_rate_hz > 0 and n_images > 0:
        duration_s = n_images / frame_rate_hz
        tracks_per_second = len(rows) / duration_s if duration_s > 0 else None
    return {
        "velocity_rows": len(rows),
        "velocity_ratio_y_median": None if ratios.size == 0 else float(np.median(ratios)),
        "velocity_ratio_y_iqr": None if ratios.size == 0 else float(np.percentile(ratios, 75) - np.percentile(ratios, 25)),
        "speed_px_per_frame_median": None if speeds.size == 0 else float(np.median(speeds)),
        "frame_rate_hz": frame_rate_hz,
        "duration_s": duration_s,
        "tracks_per_second": tracks_per_second,
    }


def detection_confidence_rows(output_dir: Path) -> list[dict[str, Any]]:
    """Compute a simple per-detection confidence score from existing columns."""

    output_dir = Path(output_dir)
    reg_score_by_frame: dict[int, float] = {}
    for row in phase_rows(output_dir):
        frame = finite_int(row.get("frame_index"))
        score = finite_float(row.get("score"))
        if frame is not None and score is not None:
            reg_score_by_frame[frame] = max(0.0, min(1.0, score))

    rows: list[dict[str, Any]] = []
    for row in load_detection_rows(output_dir):
        frame = finite_int(row.get("frame_index"))
        peak = finite_float(row.get("peak_signal")) or 0.0
        mean_signal = finite_float(row.get("mean_signal")) or 0.0
        area = finite_float(row.get("area_px")) or 0.0
        top = finite_float(row.get("bbox_top"))
        bottom = finite_float(row.get("bbox_bottom"))
        left = finite_float(row.get("bbox_left"))
        right = finite_float(row.get("bbox_right"))
        overlap = finite_float(row.get("recurrent_artifact_overlap_fraction")) or 0.0
        probability = finite_float(row.get("recurrent_artifact_probability")) or 0.0
        reg_score = 1.0 if frame is None else reg_score_by_frame.get(frame, 1.0)
        signal_score = 1.0 - math.exp(-max(0.0, peak) / 5.0)
        mean_score = 1.0 - math.exp(-max(0.0, mean_signal) / 5.0)
        shape_score = 1.0
        if None not in (top, bottom, left, right) and area > 0:
            bbox_area = max(1.0, (bottom - top) * (right - left))
            extent = max(0.0, min(1.0, area / bbox_area))
            shape_score = math.sqrt(extent)
        artifact_penalty = max(0.0, 1.0 - max(overlap, probability))
        confidence = signal_score * (0.5 + 0.5 * mean_score) * shape_score * reg_score * artifact_penalty
        out = dict(row)
        out.update(
            {
                "registration_score": reg_score,
                "signal_score": signal_score,
                "shape_score": shape_score,
                "artifact_penalty": artifact_penalty,
                "detection_confidence": confidence,
            }
        )
        rows.append(out)
    return rows


def write_detection_confidence(output_dir: Path, *, report_dir: Path) -> list[dict[str, Any]]:
    rows = detection_confidence_rows(output_dir)
    fieldnames = [
        "frame_index", "image", "label", "y", "x", "area_px", "peak_signal", "mean_signal",
        "registration_score", "signal_score", "shape_score", "artifact_penalty", "detection_confidence",
    ]
    write_csv(report_dir / "detection_confidence.csv", rows, fieldnames)
    return rows


def adaptive_map_frame_plan(output_dir: Path, *, frame_count: int | None = None, top_n: int = 120) -> list[dict[str, Any]]:
    """Suggest low-contamination, phase-diverse frames for map reconstruction."""

    metadata = load_metadata(output_dir)
    if frame_count is None:
        frame_count = finite_int(metadata.get("n_images")) or 0
    counts = detection_count_by_frame(output_dir)
    phase_by_frame: dict[int, float] = {}
    score_by_frame: dict[int, float] = {}
    for row in phase_rows(output_dir):
        frame = finite_int(row.get("frame_index"))
        phase = finite_float(row.get("phase_px"))
        score = finite_float(row.get("score"))
        if frame is not None and phase is not None:
            phase_by_frame[frame] = phase
            if score is not None:
                score_by_frame[frame] = score
    frames = sorted(set(range(frame_count)) | set(counts) | set(phase_by_frame))
    if not frames:
        return []
    map_height = finite_float(metadata.get("belt_map_height_px")) or 1.0
    bucket_count = max(1, min(top_n, 64))
    buckets: dict[int, list[tuple[float, int]]] = {}
    for frame in frames:
        phase = phase_by_frame.get(frame, float(frame))
        bucket = int((phase % map_height) / map_height * bucket_count)
        density = counts.get(frame, 0)
        registration_penalty = 1.0 - score_by_frame.get(frame, 1.0)
        rank_cost = density + 10.0 * max(0.0, registration_penalty)
        buckets.setdefault(bucket, []).append((rank_cost, frame))
    selected: list[dict[str, Any]] = []
    while len(selected) < top_n and buckets:
        progressed = False
        for bucket in sorted(list(buckets)):
            candidates = buckets[bucket]
            if not candidates:
                buckets.pop(bucket, None)
                continue
            candidates.sort()
            cost, frame = candidates.pop(0)
            selected.append({
                "frame_index": frame,
                "phase_bucket": bucket,
                "rank_cost": cost,
                "n_detections": counts.get(frame, 0),
                "registration_score": score_by_frame.get(frame, ""),
            })
            progressed = True
            if len(selected) >= top_n:
                break
        if not progressed:
            break
    return selected[:top_n]


def suggest_label_frames(output_dir: Path, *, frame_count: int = 50) -> list[dict[str, Any]]:
    """Return a diverse set of frames to annotate for real-data validation."""

    tables = worst_frame_tables(output_dir, top_n=max(5, frame_count))
    selected: dict[int, set[str]] = {}
    bucket_order = [
        "detection_spikes",
        "empty_or_low_detection_frames",
        "low_registration_score",
        "large_phase_correction",
        "recurrent_rejections",
        "photometric_rmse",
    ]
    per_bucket = max(1, math.ceil(frame_count / max(1, len(bucket_order))))
    for bucket in bucket_order:
        for row in tables.get(bucket, [])[:per_bucket]:
            frame = finite_int(row.get("frame_index"))
            if frame is not None:
                selected.setdefault(frame, set()).add(bucket)
    for bucket in bucket_order:
        for row in tables.get(bucket, []):
            if len(selected) >= frame_count:
                break
            frame = finite_int(row.get("frame_index"))
            if frame is not None:
                selected.setdefault(frame, set()).add(bucket)
        if len(selected) >= frame_count:
            break
    return [
        {"frame_index": frame, "reason": ";".join(sorted(reasons))}
        for frame, reasons in sorted(selected.items())[:frame_count]
    ]


def write_label_plan(output_dir: Path, *, output_path: Path, frame_count: int = 50) -> list[dict[str, Any]]:
    rows = suggest_label_frames(output_dir, frame_count=frame_count)
    write_csv(output_path, rows, ["frame_index", "reason"])
    return rows


def evaluate_quality_contract(output_dir: Path, contract: Mapping[str, Any] | None = None) -> list[ContractResult]:
    """Evaluate a compact quality contract against standard output files."""

    output_dir = Path(output_dir)
    contract = DEFAULT_QUALITY_CONTRACT if contract is None else contract
    metadata = load_metadata(output_dir)
    results: list[ContractResult] = []

    corrections = np.asarray([
        value for value in (finite_float(row.get("correction_px")) for row in phase_rows(output_dir))
        if value is not None
    ], dtype=np.float64)
    search_radius = finite_float(metadata.get("registration_search_radius_px")) or 8.0
    search_step = finite_float(metadata.get("registration_search_step_px")) or 1.0
    boundary_tolerance = max(1e-9, 0.5 * search_step)
    boundary_share = None if corrections.size == 0 else float(np.mean(np.abs(np.abs(corrections) - search_radius) <= boundary_tolerance))
    threshold = finite_float(contract.get("max_registration_boundary_share"))
    if threshold is not None:
        results.append(ContractResult("registration_boundary_share", boundary_share is None or boundary_share <= threshold, boundary_share, threshold, "<=", "share of phase corrections at search boundary"))

    detections = load_detection_rows(output_dir)
    areas = np.asarray([value for value in (finite_float(row.get("area_px")) for row in detections) if value is not None], dtype=np.float64)
    tiny_share = None if areas.size == 0 else float(np.mean(areas <= 2.0))
    threshold = finite_float(contract.get("max_many_tiny_component_share"))
    if threshold is not None:
        results.append(ContractResult("many_tiny_component_share", tiny_share is None or tiny_share <= threshold, tiny_share, threshold, "<=", "share of detections with area <= 2 px"))

    velocities = read_csv_rows(output_dir / "velocities.csv")
    ratios = np.asarray([value for value in (finite_float(row.get("velocity_ratio_y")) for row in velocities) if value is not None], dtype=np.float64)
    ratio_range = contract.get("velocity_ratio_y_range", [0.0, 1.1])
    lower, upper = float(ratio_range[0]), float(ratio_range[1])
    in_range_share = None if ratios.size == 0 else float(np.mean((lower <= ratios) & (ratios <= upper)))
    threshold = finite_float(contract.get("min_velocity_ratio_in_range_share"))
    if threshold is not None:
        results.append(ContractResult("velocity_ratio_in_range_share", in_range_share is None or in_range_share >= threshold, in_range_share, threshold, ">=", f"share of velocity ratios in [{lower}, {upper}]"))

    recurrent_rejected = finite_int(metadata.get("n_recurrent_artifact_rejected")) or 0
    n_detections = finite_int(metadata.get("n_detections")) or len(detections)
    rejection_share = None if recurrent_rejected + n_detections == 0 else recurrent_rejected / (recurrent_rejected + n_detections)
    threshold = finite_float(contract.get("max_recurrent_rejection_share"))
    if threshold is not None:
        results.append(ContractResult("recurrent_rejection_share", rejection_share is None or rejection_share <= threshold, rejection_share, threshold, "<=", "share rejected by recurrent artifact filter"))

    min_filtered = finite_int(contract.get("min_filtered_velocity_estimates"))
    if min_filtered is not None:
        filtered = read_csv_rows(output_dir / "filtered_velocities.csv")
        results.append(ContractResult("filtered_velocity_estimates", len(filtered) >= min_filtered, len(filtered), min_filtered, ">=", "accepted filtered velocity rows"))
    return results


def write_quality_contract_report(output_dir: Path, *, report_dir: Path, contract: Mapping[str, Any] | None = None) -> list[ContractResult]:
    results = evaluate_quality_contract(output_dir, contract=contract)
    write_json(report_dir / "quality_contract.json", [asdict(result) for result in results])
    lines = ["# Quality contract", ""]
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(f"- **{status}** `{result.name}`: {result.value} {result.comparison} {result.threshold} — {result.description}")
    (report_dir / "quality_contract.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results


def belt_edge_ignore_mask(shape: tuple[int, int], *, margin_px: int) -> NDArray[np.bool_]:
    """Return a boolean allow-mask that excludes belt-edge margins."""

    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError("shape must be positive")
    if margin_px < 0:
        raise ValueError("margin_px must be non-negative")
    mask = np.ones((height, width), dtype=bool)
    if margin_px == 0:
        return mask
    margin = min(margin_px, height // 2, width // 2)
    mask[:margin, :] = False
    mask[-margin:, :] = False
    mask[:, :margin] = False
    mask[:, -margin:] = False
    return mask


def load_ignore_mask(path: Path, *, shape: tuple[int, int] | None = None) -> NDArray[np.bool_]:
    """Load a PNG/NumPy ignore mask as an allow-mask; nonzero pixels are allowed."""

    path = Path(path)
    if path.suffix.lower() == ".npy":
        mask = np.asarray(np.load(path), dtype=bool)
    else:
        mask = np.asarray(Image.open(path).convert("L")) > 0
    if shape is not None and tuple(mask.shape) != tuple(shape):
        raise ValueError(f"ignore mask shape {mask.shape} does not match expected {shape}")
    return mask


def apply_ignore_mask(values: ArrayLike, allow_mask: ArrayLike, *, fill_value: float = np.nan) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64).copy()
    mask = np.asarray(allow_mask, dtype=bool)
    if arr.shape != mask.shape:
        raise ValueError("values and allow_mask must have the same shape")
    arr[~mask] = fill_value
    return arr


def color_residual_score(
    observed_rgb: ArrayLike,
    expected_rgb: ArrayLike,
    *,
    min_scale: float = 1e-6,
) -> FloatArray:
    """Return an RGB residual magnitude score with robust per-channel scaling."""

    observed = np.asarray(observed_rgb, dtype=np.float64)
    expected = np.asarray(expected_rgb, dtype=np.float64)
    if observed.shape != expected.shape:
        raise ValueError("observed_rgb and expected_rgb must have the same shape")
    if observed.ndim != 3 or observed.shape[2] not in (3, 4):
        raise ValueError("color residual scoring expects an HxWx3 or HxWx4 image")
    residual = observed[..., :3] - expected[..., :3]
    scales = []
    for channel in range(3):
        values = residual[..., channel]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            scales.append(1.0)
            continue
        center = float(np.median(finite))
        mad = float(np.median(np.abs(finite - center)))
        scales.append(max(1.4826 * mad, min_scale))
    normalized = residual / np.asarray(scales, dtype=np.float64)[None, None, :]
    return np.sqrt(np.sum(np.square(normalized), axis=2))


def warp_perspective(
    image: ArrayLike,
    homography: ArrayLike,
    output_shape: tuple[int, int],
    *,
    fill_value: float = 0.0,
) -> FloatArray:
    """Warp an image with a 3x3 homography using bilinear sampling."""

    src = np.asarray(image, dtype=np.float64)
    if src.ndim not in (2, 3):
        raise ValueError("image must be 2-D or 3-D")
    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("homography must be 3x3")
    inv = np.linalg.inv(matrix)
    out_h, out_w = output_shape
    yy, xx = np.meshgrid(np.arange(out_h, dtype=np.float64), np.arange(out_w, dtype=np.float64), indexing="ij")
    ones = np.ones_like(xx)
    coords = np.stack([xx.ravel(), yy.ravel(), ones.ravel()], axis=0)
    src_coords = inv @ coords
    src_coords /= src_coords[2:3]
    x = src_coords[0].reshape(output_shape)
    y = src_coords[1].reshape(output_shape)
    return _bilinear_sample(src, y, x, fill_value=fill_value)


def _bilinear_sample(image: np.ndarray, y: np.ndarray, x: np.ndarray, *, fill_value: float) -> FloatArray:
    h, w = image.shape[:2]
    valid = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    x0 = np.clip(np.floor(x).astype(np.int64), 0, w - 1)
    y0 = np.clip(np.floor(y).astype(np.int64), 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = x - x0
    wy = y - y0
    if image.ndim == 2:
        result = (
            (1 - wx) * (1 - wy) * image[y0, x0]
            + wx * (1 - wy) * image[y0, x1]
            + (1 - wx) * wy * image[y1, x0]
            + wx * wy * image[y1, x1]
        )
        result[~valid] = fill_value
        return result
    result = (
        (1 - wx)[..., None] * (1 - wy)[..., None] * image[y0, x0]
        + wx[..., None] * (1 - wy)[..., None] * image[y0, x1]
        + (1 - wx)[..., None] * wy[..., None] * image[y1, x0]
        + wx[..., None] * wy[..., None] * image[y1, x1]
    )
    result[~valid] = fill_value
    return result


def short_horizon_link_detections(
    detections: Sequence[Mapping[str, Any]],
    *,
    max_frame_gap: int = 2,
    max_distance_px: float = 25.0,
) -> list[dict[str, Any]]:
    """Lightweight short-horizon linking for post-run fragmentation diagnostics."""

    if max_frame_gap < 1:
        raise ValueError("max_frame_gap must be positive")
    if max_distance_px <= 0:
        raise ValueError("max_distance_px must be positive")
    items = []
    for idx, row in enumerate(detections):
        frame = finite_int(row.get("frame_index"))
        y = finite_float(row.get("y"))
        x = finite_float(row.get("x"))
        if frame is None or y is None or x is None:
            continue
        items.append({"index": idx, "frame_index": frame, "y": y, "x": x})
    items.sort(key=lambda row: (row["frame_index"], row["y"], row["x"]))
    tracks: list[list[dict[str, Any]]] = []
    for item in items:
        best_track = None
        best_distance = max_distance_px
        for track_index, track in enumerate(tracks):
            last = track[-1]
            gap = item["frame_index"] - last["frame_index"]
            if gap < 1 or gap > max_frame_gap:
                continue
            distance = math.hypot(item["y"] - last["y"], item["x"] - last["x"])
            if distance < best_distance:
                best_distance = distance
                best_track = track_index
        if best_track is None:
            tracks.append([item])
        else:
            tracks[best_track].append(item)
    rows = []
    for track_id, track in enumerate(tracks):
        for det_index, item in enumerate(track):
            rows.append({
                "diagnostic_track_id": track_id,
                "track_detection_index": det_index,
                **item,
            })
    return rows


def write_postrun_audit(
    output_dir: Path,
    *,
    report_dir: Path | None = None,
    top_n: int = 10,
    frame_rate_hz: float | None = None,
    write_full_counts: bool = False,
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a bundle of post-run result-improvement diagnostics."""

    output_dir = Path(output_dir)
    report_dir = output_dir / "postrun_audit" if report_dir is None else Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    flags = write_quality_flags(output_dir, report_dir=report_dir)
    worst = write_worst_frame_tables(output_dir, report_dir=report_dir, top_n=top_n)
    map_summary = write_map_uncertainty_outputs(output_dir, report_dir=report_dir, write_full_counts=write_full_counts)
    seam = write_seam_diagnostics(output_dir, report_dir=report_dir)
    confidence = write_detection_confidence(output_dir, report_dir=report_dir)
    label_plan = write_label_plan(output_dir, output_path=report_dir / "label_plan.csv", frame_count=max(20, top_n * 2))
    map_plan = adaptive_map_frame_plan(output_dir, top_n=120)
    write_csv(report_dir / "adaptive_map_frame_plan.csv", map_plan, ["frame_index", "phase_bucket", "rank_cost", "n_detections", "registration_score"])
    flux = flux_summary(output_dir, frame_rate_hz=frame_rate_hz)
    write_json(report_dir / "flux_summary.json", flux)
    contract_results = write_quality_contract_report(output_dir, report_dir=report_dir, contract=contract)

    detections = load_detection_rows(output_dir)
    diagnostic_links = short_horizon_link_detections(detections, max_frame_gap=2) if detections else []
    write_csv(report_dir / "short_horizon_track_diagnostics.csv", diagnostic_links, ["diagnostic_track_id", "track_detection_index", "index", "frame_index", "y", "x"])

    summary = {
        "output_dir": str(output_dir),
        "report_dir": str(report_dir),
        "quality_flags": len(flags),
        "worst_frame_tables": {key: len(value) for key, value in worst.items()},
        "map_uncertainty": map_summary,
        "seam_diagnostics": seam,
        "detection_confidence_rows": len(confidence),
        "label_plan_frames": len(label_plan),
        "adaptive_map_frame_candidates": len(map_plan),
        "flux_summary": flux,
        "quality_contract_passed": all(result.passed for result in contract_results),
    }
    write_json(report_dir / "postrun_audit_summary.json", summary)
    return summary


def write_quality_contract_template(path: Path, *, synthetic: bool = False) -> None:
    template = SYNTHETIC_REGRESSION_CONTRACT_TEMPLATE if synthetic else DEFAULT_QUALITY_CONTRACT
    write_json(path, template)
