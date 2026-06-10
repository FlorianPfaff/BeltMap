"""Advanced result-improvement helpers for BeltMap experiments.

The functions in this module are intentionally self-contained and NumPy-only so
that they can be used from scripts, validation reports, and future driver
integration without adding runtime dependencies.  They cover several higher-risk
improvements that are best evaluated as opt-in diagnostics before being enabled
by default in the image driver.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.floating]
REVIEWED_GROUND_TRUTH_STATUS = "reviewed_ground_truth"


@dataclass(frozen=True)
class GainOffsetFit:
    """Robust per-frame photometric fit ``observed ~= gain * expected + offset``."""

    gain: float
    offset: float
    n_pixels: int
    rmse_gray: float
    trimmed_fraction: float


@dataclass(frozen=True)
class ShiftEstimate:
    """Integer two-dimensional registration diagnostic result."""

    shift_y_px: int
    shift_x_px: int
    loss: float
    score: float


@dataclass(frozen=True)
class RealLabelMetrics:
    """Detection metrics against sparse manually annotated real-data boxes."""

    frames: int
    truth_boxes: int
    detection_boxes: int
    matches: int
    precision: float | None
    recall: float | None
    f1: float | None
    mean_iou: float | None
    mean_centroid_error_px: float | None


@dataclass(frozen=True)
class Provenance:
    """Run provenance that can be embedded into metadata or validation reports."""

    python_version: str
    platform: str
    executable: str
    cwd: str
    git_commit: str | None
    git_dirty: bool | None
    environment: dict[str, str]
    input_manifest_sha256: str | None


def as_float_image(image: ArrayLike, *, name: str) -> FloatArray:
    arr = np.asarray(image, dtype=np.float64)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return arr


def finite_mask(*arrays: ArrayLike) -> NDArray[np.bool_]:
    if not arrays:
        raise ValueError("at least one array is required")
    converted = [np.asarray(array, dtype=np.float64) for array in arrays]
    shape = converted[0].shape
    valid = np.ones(shape, dtype=bool)
    for array in converted:
        if array.shape != shape:
            raise ValueError("all arrays must have the same shape")
        valid &= np.isfinite(array)
    return valid


def robust_gain_offset(
    observed: ArrayLike,
    expected: ArrayLike,
    *,
    mask: ArrayLike | None = None,
    trim_fraction: float = 0.05,
    max_iterations: int = 3,
    min_pixels: int = 128,
) -> GainOffsetFit:
    """Fit robust gain/offset for one frame.

    This is intended for illumination drift, LED flicker, and camera exposure
    changes.  The fit repeatedly solves a least-squares line and removes the
    largest residuals according to ``trim_fraction``.
    """

    if not 0.0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction must be in [0, 0.5)")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    obs = as_float_image(observed, name="observed")
    exp = as_float_image(expected, name="expected")
    valid = finite_mask(obs, exp)
    if mask is not None:
        user_mask = np.asarray(mask, dtype=bool)
        if user_mask.shape != obs.shape:
            raise ValueError("mask must have the same shape as observed")
        valid &= user_mask
    x = exp[valid].ravel()
    y = obs[valid].ravel()
    if x.size < min_pixels:
        raise ValueError(f"not enough valid pixels for photometric fit: {x.size} < {min_pixels}")

    keep = np.ones(x.size, dtype=bool)
    gain = 1.0
    offset = 0.0
    for _iteration in range(max_iterations):
        kept_x = x[keep]
        kept_y = y[keep]
        if kept_x.size < min_pixels:
            break
        design = np.column_stack([kept_x, np.ones(kept_x.size)])
        gain, offset = (float(v) for v in np.linalg.lstsq(design, kept_y, rcond=None)[0])
        residual = kept_y - (gain * kept_x + offset)
        if trim_fraction <= 0:
            break
        cutoff = np.quantile(np.abs(residual), 1.0 - trim_fraction)
        new_keep_kept = np.abs(residual) <= cutoff
        new_keep = np.zeros_like(keep)
        new_keep[np.flatnonzero(keep)[new_keep_kept]] = True
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep

    fitted = gain * x[keep] + offset
    rmse = float(np.sqrt(np.mean(np.square(y[keep] - fitted)))) if np.any(keep) else float("nan")
    return GainOffsetFit(
        gain=gain,
        offset=offset,
        n_pixels=int(np.count_nonzero(keep)),
        rmse_gray=rmse,
        trimmed_fraction=float(1.0 - np.count_nonzero(keep) / x.size),
    )


def apply_gain_offset(expected: ArrayLike, fit: GainOffsetFit) -> FloatArray:
    """Return ``fit.gain * expected + fit.offset`` as float64."""

    return fit.gain * as_float_image(expected, name="expected") + fit.offset


def quadratic_subpixel_minimum(offsets: Sequence[float], losses: Sequence[float]) -> float:
    """Estimate the sub-grid minimum of a sampled registration loss curve.

    The function fits a parabola through the best sample and its two neighbors.
    If the best sample is at the boundary, or the parabola is degenerate, the
    best sampled offset is returned unchanged.
    """

    x = np.asarray(offsets, dtype=np.float64)
    y = np.asarray(losses, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size == 0:
        raise ValueError("offsets and losses must be one-dimensional arrays of equal nonzero length")
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        raise ValueError("no finite registration losses")
    x = x[finite]
    y = y[finite]
    best = int(np.argmin(y))
    if best == 0 or best == x.size - 1:
        return float(x[best])
    xs = x[best - 1 : best + 2]
    ys = y[best - 1 : best + 2]
    a, b, _c = np.polyfit(xs, ys, deg=2)
    if not np.isfinite(a) or a <= 0:
        return float(x[best])
    candidate = float(-b / (2.0 * a))
    if candidate < float(np.min(xs)) or candidate > float(np.max(xs)):
        return float(x[best])
    return candidate


def _trimmed_loss(values: np.ndarray, *, trim_fraction: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("inf")
    squared = np.square(finite)
    if trim_fraction > 0 and squared.size > 1:
        cutoff = np.quantile(squared, 1.0 - trim_fraction)
        squared = squared[squared <= cutoff]
    return float(np.mean(squared))


def estimate_integer_xy_shift(
    observed: ArrayLike,
    expected: ArrayLike,
    *,
    mask: ArrayLike | None = None,
    max_shift_y_px: int = 4,
    max_shift_x_px: int = 4,
    trim_fraction: float = 0.08,
) -> ShiftEstimate:
    """Diagnostic 2-D registration by small integer-shift grid search.

    This deliberately does not change BeltMap's one-dimensional phase model.  It
    is a sanity check for crop drift, horizontal misalignment, or camera motion.
    """

    if max_shift_y_px < 0 or max_shift_x_px < 0:
        raise ValueError("max shifts must be non-negative")
    obs = as_float_image(observed, name="observed")
    exp = as_float_image(expected, name="expected")
    if obs.shape != exp.shape:
        raise ValueError("observed and expected must have the same shape")
    valid = finite_mask(obs, exp)
    if mask is not None:
        user_mask = np.asarray(mask, dtype=bool)
        if user_mask.shape != obs.shape:
            raise ValueError("mask must have the same shape as observed")
        valid &= user_mask

    losses: list[tuple[float, int, int]] = []
    for dy in range(-max_shift_y_px, max_shift_y_px + 1):
        for dx in range(-max_shift_x_px, max_shift_x_px + 1):
            shifted = np.roll(np.roll(exp, dy, axis=0), dx, axis=1)
            shifted_valid = np.roll(np.roll(valid, dy, axis=0), dx, axis=1) & valid
            loss = _trimmed_loss((obs - shifted)[shifted_valid], trim_fraction=trim_fraction)
            losses.append((loss, dy, dx))
    best_loss, best_y, best_x = min(losses, key=lambda item: item[0])
    median_loss = float(np.median([loss for loss, _dy, _dx in losses if np.isfinite(loss)]))
    score = 0.0 if median_loss <= 0 else max(0.0, 1.0 - best_loss / median_loss)
    return ShiftEstimate(best_y, best_x, best_loss, score)


def unwrap_periodic(values: ArrayLike, period: float) -> FloatArray:
    """Unwrap periodic pixel phases into a continuous sequence."""

    if period <= 0:
        raise ValueError("period must be positive")
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr.astype(np.float64)
    radians = arr / period * 2.0 * np.pi
    return np.unwrap(radians) / (2.0 * np.pi) * period


def smooth_phase_velocity(
    measured_phase_px: ArrayLike,
    *,
    period_px: float,
    scores: ArrayLike | None = None,
    process_noise_px: float = 0.1,
    measurement_noise_px: float = 2.0,
) -> dict[str, list[float]]:
    """Simple phase/velocity state-space smoother for registration diagnostics.

    The state is ``[phase, velocity]`` with a constant-velocity transition.  It
    is intentionally lightweight: good enough to evaluate whether a full driver
    integration is worthwhile.
    """

    if process_noise_px <= 0 or measurement_noise_px <= 0:
        raise ValueError("noise scales must be positive")
    measured = unwrap_periodic(measured_phase_px, period_px)
    n = measured.size
    if n == 0:
        return {"phase_px": [], "velocity_px_per_frame": []}
    weights = np.ones(n, dtype=np.float64)
    if scores is not None:
        score_arr = np.asarray(scores, dtype=np.float64)
        if score_arr.shape != measured.shape:
            raise ValueError("scores must match measured phases")
        weights = np.clip(np.nan_to_num(score_arr, nan=0.0), 0.05, 1.0)

    state = np.array([measured[0], 0.0], dtype=np.float64)
    covariance = np.diag([measurement_noise_px**2, measurement_noise_px**2])
    transition = np.array([[1.0, -1.0], [0.0, 1.0]], dtype=np.float64)
    observation = np.array([[1.0, 0.0]], dtype=np.float64)
    process = np.diag([process_noise_px**2, process_noise_px**2])
    phase = np.empty(n, dtype=np.float64)
    velocity = np.empty(n, dtype=np.float64)

    for index, measurement in enumerate(measured):
        if index > 0:
            state = transition @ state
            covariance = transition @ covariance @ transition.T + process
        measurement_var = (measurement_noise_px / weights[index]) ** 2
        innovation = measurement - float((observation @ state)[0])
        innovation_var = float((observation @ covariance @ observation.T)[0, 0] + measurement_var)
        kalman_gain = covariance @ observation.T / innovation_var
        state = state + (kalman_gain[:, 0] * innovation)
        covariance = (np.eye(2) - kalman_gain @ observation) @ covariance
        phase[index] = state[0] % period_px
        velocity[index] = state[1]
    return {
        "phase_px": [float(v) for v in phase],
        "velocity_px_per_frame": [float(v) for v in velocity],
    }


def map_uncertainty_from_counts(
    counts: ArrayLike,
    *,
    min_count: float = 1.0,
    scale: float = 1.0,
) -> FloatArray:
    """Convert belt-map observation counts to a noise/uncertainty floor."""

    if min_count <= 0 or scale <= 0:
        raise ValueError("min_count and scale must be positive")
    count_arr = np.asarray(counts, dtype=np.float64)
    safe = np.maximum(count_arr, min_count)
    uncertainty = scale / np.sqrt(safe)
    uncertainty[~np.isfinite(count_arr) | (count_arr <= 0)] = scale
    return uncertainty


def seam_discontinuity_profile(belt_map: ArrayLike, *, seam_row: int = 0, window_px: int = 8) -> dict[str, float]:
    """Measure how discontinuous a cyclic belt map is around a seam row."""

    belt = as_float_image(belt_map, name="belt_map")
    if belt.ndim != 2:
        raise ValueError("belt_map must be 2-D")
    if window_px < 1:
        raise ValueError("window_px must be positive")
    height = belt.shape[0]
    row = int(seam_row) % height
    before = np.asarray([belt[(row - i - 1) % height] for i in range(window_px)], dtype=np.float64)
    after = np.asarray([belt[(row + i) % height] for i in range(window_px)], dtype=np.float64)
    jump = after[0] - before[0]
    local = after - before
    return {
        "seam_row": float(row),
        "mean_abs_jump_gray": float(np.mean(np.abs(jump))),
        "p95_abs_jump_gray": float(np.percentile(np.abs(jump), 95)),
        "window_mean_abs_difference_gray": float(np.mean(np.abs(local))),
    }


def theil_sen_slope(times: ArrayLike, values: ArrayLike) -> float:
    """Robust median pairwise slope estimator for velocity fitting."""

    t = np.asarray(times, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if t.shape != y.shape or t.ndim != 1:
        raise ValueError("times and values must be one-dimensional arrays of equal length")
    finite = np.isfinite(t) & np.isfinite(y)
    t = t[finite]
    y = y[finite]
    if np.unique(t).size < 2:
        raise ValueError("at least two distinct times are required")
    slopes: list[float] = []
    for i in range(t.size):
        dt = t[i + 1 :] - t[i]
        dy = y[i + 1 :] - y[i]
        valid = dt != 0
        slopes.extend((dy[valid] / dt[valid]).tolist())
    if not slopes:
        raise ValueError("no valid slope pairs")
    return float(np.median(np.asarray(slopes, dtype=np.float64)))


def track_confidence_score(
    *,
    n_detections: int,
    min_track_length: int,
    mean_peak_signal: float | None = None,
    velocity_fit_rmse_px: float | None = None,
    velocity_ratio_y: float | None = None,
) -> float:
    """Continuous confidence score for a particle track."""

    if min_track_length <= 0:
        raise ValueError("min_track_length must be positive")
    length_score = min(1.0, max(0.0, n_detections / min_track_length))
    signal_score = 1.0 if mean_peak_signal is None else 1.0 - math.exp(-max(0.0, mean_peak_signal) / 5.0)
    fit_score = 1.0 if velocity_fit_rmse_px is None else 1.0 / (1.0 + max(0.0, velocity_fit_rmse_px))
    if velocity_ratio_y is None:
        ratio_score = 1.0
    else:
        ratio_score = 1.0 if 0.0 <= velocity_ratio_y <= 1.1 else max(0.0, 1.0 - abs(velocity_ratio_y - 0.55))
    return float(length_score * signal_score * fit_score * ratio_score)


def bbox_iou(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    """Intersection-over-union for boxes with top/left/bottom/right keys."""

    top = max(float(a["top"]), float(b["top"]))
    left = max(float(a["left"]), float(b["left"]))
    bottom = min(float(a["bottom"]), float(b["bottom"]))
    right = min(float(a["right"]), float(b["right"]))
    inter = max(0.0, bottom - top) * max(0.0, right - left)
    area_a = max(0.0, float(a["bottom"]) - float(a["top"])) * max(0.0, float(a["right"]) - float(a["left"]))
    area_b = max(0.0, float(b["bottom"]) - float(b["top"])) * max(0.0, float(b["right"]) - float(b["left"]))
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else float(inter / union)


def _box_centroid(box: Mapping[str, float]) -> tuple[float, float]:
    return ((float(box["top"]) + float(box["bottom"])) / 2.0, (float(box["left"]) + float(box["right"])) / 2.0)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def detection_boxes_by_frame(output_dir: Path) -> dict[int, list[dict[str, float]]]:
    rows = read_csv_rows(output_dir / "detections.csv")
    grouped: dict[int, list[dict[str, float]]] = {}
    for row in rows:
        try:
            frame = finite_int(row["frame_index"])
            if frame is None:
                continue
            box = _parse_detection_box(row)
            if box is None:
                continue
        except (KeyError, TypeError, ValueError):
            continue
        grouped.setdefault(frame, []).append(box)
    return grouped


def load_real_label_boxes(path: Path) -> dict[int, list[dict[str, float]]]:
    """Load sparse real-data boxes from a compact JSON format.

    Accepted format::

        {"frames": [{"frame_index": 0,
                      "boxes": [{"top": 1, "left": 2, "bottom": 3, "right": 4}]}]}
    """

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        _validate_reviewed_truth_status(data)
    frames = data.get("frames") if isinstance(data, dict) else None
    if not isinstance(frames, list):
        raise ValueError("label JSON must contain a 'frames' list")
    result: dict[int, list[dict[str, float]]] = {}
    for frame in frames:
        if not isinstance(frame, Mapping):
            raise ValueError("label frames must be objects")
        frame_index = finite_int(frame.get("frame_index"))
        if frame_index is None:
            raise ValueError("label frame_index values must be finite integers")
        boxes: list[dict[str, float]] = []
        for box in frame.get("boxes", []):
            boxes.append(_parse_real_label_box(box))
        result[frame_index] = boxes
    return result


def _review_flag_is_true(value: Any) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "0", "false", "no", "n"}:
            return False
        if normalized in {"1", "true", "yes", "y"}:
            return True
    return bool(value)


def _validate_reviewed_truth_status(data: dict[str, Any]) -> None:
    status = data.get("status")
    requires_manual_review = data.get("requires_manual_review")
    if status is None and requires_manual_review is None:
        return
    if status != REVIEWED_GROUND_TRUTH_STATUS or _review_flag_is_true(
        requires_manual_review,
    ):
        raise ValueError(
            "label JSON is not reviewed ground truth; set status="
            f"{REVIEWED_GROUND_TRUTH_STATUS!r} and requires_manual_review=false "
            "only after all scored frames have been reviewed"
        )


def _parse_real_label_box(box: Mapping[str, Any]) -> dict[str, float]:
    try:
        top = float(box["top"])
        left = float(box["left"])
        bottom = float(box["bottom"])
        right = float(box["right"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("label boxes must contain numeric top/left/bottom/right") from exc
    if not all(math.isfinite(value) for value in (top, left, bottom, right)):
        raise ValueError("label box coordinates must be finite")
    if bottom <= top or right <= left:
        raise ValueError("label boxes must have positive half-open area")
    return {"top": top, "left": left, "bottom": bottom, "right": right}


def _parse_detection_box(row: Mapping[str, Any]) -> dict[str, float] | None:
    values = [
        finite_float(row.get(key))
        for key in ("bbox_top", "bbox_left", "bbox_bottom", "bbox_right")
    ]
    if any(value is None for value in values):
        return None
    top, left, bottom, right = values
    assert top is not None and left is not None
    assert bottom is not None and right is not None
    if bottom <= top or right <= left:
        return None
    return {"top": top, "left": left, "bottom": bottom, "right": right}


def evaluate_real_detections(output_dir: Path, labels_path: Path, *, iou_threshold: float = 0.5) -> RealLabelMetrics:
    """Evaluate BeltMap detections against sparse real-data annotations."""

    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    detections = detection_boxes_by_frame(output_dir)
    truth = load_real_label_boxes(labels_path)
    matches = 0
    ious: list[float] = []
    centroid_errors: list[float] = []
    labeled_frames = set(truth)
    detection_count = sum(len(detections.get(frame_index, [])) for frame_index in labeled_frames)
    truth_count = sum(len(v) for v in truth.values())
    for frame_index, truth_boxes in truth.items():
        frame_detections = detections.get(frame_index, [])
        candidates: list[tuple[float, int, int]] = []
        for t_idx, t_box in enumerate(truth_boxes):
            for d_idx, d_box in enumerate(frame_detections):
                iou = bbox_iou(t_box, d_box)
                if iou >= iou_threshold:
                    candidates.append((iou, t_idx, d_idx))
        used_truth: set[int] = set()
        used_detections: set[int] = set()
        for iou, t_idx, d_idx in sorted(candidates, reverse=True):
            if t_idx in used_truth or d_idx in used_detections:
                continue
            used_truth.add(t_idx)
            used_detections.add(d_idx)
            matches += 1
            ious.append(iou)
            ty, tx = _box_centroid(truth_boxes[t_idx])
            dy, dx = _box_centroid(frame_detections[d_idx])
            centroid_errors.append(float(math.hypot(ty - dy, tx - dx)))
    if detection_count == 0 and truth_count == 0:
        precision = 1.0
        recall = 1.0
        f1 = 1.0
    else:
        precision = None if detection_count == 0 else matches / detection_count
        recall = None if truth_count == 0 else matches / truth_count
        if precision is None or recall is None:
            f1 = None
        elif precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
    return RealLabelMetrics(
        frames=len(truth),
        truth_boxes=truth_count,
        detection_boxes=detection_count,
        matches=matches,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_iou=None if not ious else float(np.mean(ious)),
        mean_centroid_error_px=None if not centroid_errors else float(np.mean(centroid_errors)),
    )


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
    if parsed is None or not parsed.is_integer():
        return None
    return int(parsed)


def quality_flags(output_dir: Path) -> dict[str, Any]:
    """Return machine-readable warnings for common poor-result modes."""

    metadata_path = output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    detections_per_frame = read_csv_rows(output_dir / "detections_per_frame.csv")
    detections = read_csv_rows(output_dir / "detections.csv")
    velocities = read_csv_rows(output_dir / "velocities.csv")
    phase_rows = read_csv_rows(output_dir / "phase_estimates.csv")
    flags: list[dict[str, Any]] = []

    correction_values = [finite_float(row.get("correction_px")) for row in phase_rows]
    corrections = np.asarray([v for v in correction_values if v is not None], dtype=np.float64)
    search_radius = finite_float(metadata.get("registration_search_radius_px")) or 8.0
    search_step = finite_float(metadata.get("registration_search_step_px")) or 1.0
    boundary_tolerance = max(1e-9, 0.5 * search_step)
    if corrections.size:
        boundary_share = float(np.mean(np.abs(np.abs(corrections) - search_radius) <= boundary_tolerance))
        if boundary_share > 0.05:
            flags.append({"severity": "warning", "code": "registration_boundary", "message": "phase corrections often hit the search boundary", "share": boundary_share})

    counts = np.asarray([finite_float(row.get("n_detections")) or 0.0 for row in detections_per_frame], dtype=np.float64)
    if counts.size and float(np.percentile(counts, 95)) > max(25.0, 5.0 * float(np.median(counts) + 1.0)):
        flags.append({"severity": "warning", "code": "detection_spikes", "message": "detection counts have large frame-to-frame spikes", "p95": float(np.percentile(counts, 95)), "median": float(np.median(counts))})

    areas = np.asarray([finite_float(row.get("area_px")) for row in detections if finite_float(row.get("area_px")) is not None], dtype=np.float64)
    if areas.size and float(np.mean(areas <= 2.0)) > 0.5:
        flags.append({"severity": "warning", "code": "many_tiny_components", "message": "most detections are tiny components", "share_area_le_2": float(np.mean(areas <= 2.0))})

    ratios = np.asarray([finite_float(row.get("velocity_ratio_y")) for row in velocities if finite_float(row.get("velocity_ratio_y")) is not None], dtype=np.float64)
    if ratios.size and float(np.mean((0.0 <= ratios) & (ratios <= 1.1))) < 0.5:
        flags.append({"severity": "warning", "code": "implausible_velocity_ratios", "message": "many velocity ratios are outside the expected belt-relative interval", "share_0_to_1p1": float(np.mean((0.0 <= ratios) & (ratios <= 1.1)))})

    recurrent_rejected = finite_int(metadata.get("n_recurrent_artifact_rejected")) or 0
    metadata_detection_count = finite_int(metadata.get("n_detections"))
    n_detections = (
        metadata_detection_count
        if metadata_detection_count is not None
        else len(detections)
    )
    recurrent_denominator = recurrent_rejected + n_detections
    if recurrent_denominator > 0:
        recurrent_share = recurrent_rejected / recurrent_denominator
        if recurrent_share > 0.75:
            flags.append({"severity": "info", "code": "heavy_recurrent_filtering", "message": "recurrent artifact filtering rejected most first-pass detections", "rejected": recurrent_rejected, "share": recurrent_share})

    return {"output_dir": str(output_dir), "metadata_present": bool(metadata), "flags": flags}


def _git(args: Sequence[str], *, cwd: Path) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=cwd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def dataset_manifest_sha256(root: Path, *, limit: int | None = None) -> str | None:
    if not root.exists():
        return None
    entries: list[str] = []
    files = [path for path in root.rglob("*") if path.is_file()]
    files.sort()
    if limit is not None:
        files = files[:limit]
    for path in files:
        stat = path.stat()
        entries.append(f"{path.relative_to(root)}\t{stat.st_size}\t{int(stat.st_mtime)}")
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()


def collect_provenance(*, cwd: Path | None = None, image_dir: Path | None = None, env_prefixes: Sequence[str] = ("BELTMAP_", "DETECTION_", "MAP_", "STATIC_", "RECURRENT_", "PHASE_")) -> Provenance:
    root = Path.cwd() if cwd is None else cwd
    env = {key: value for key, value in sorted(os.environ.items()) if any(key.startswith(prefix) for prefix in env_prefixes)}
    dirty_text = _git(["status", "--porcelain"], cwd=root)
    return Provenance(
        python_version=sys.version,
        platform=platform.platform(),
        executable=sys.executable,
        cwd=str(root),
        git_commit=_git(["rev-parse", "HEAD"], cwd=root),
        git_dirty=None if dirty_text is None else bool(dirty_text),
        environment=env,
        input_manifest_sha256=None if image_dir is None else dataset_manifest_sha256(image_dir),
    )


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_provenance(path: Path, *, cwd: Path | None = None, image_dir: Path | None = None) -> Provenance:
    provenance = collect_provenance(cwd=cwd, image_dir=image_dir)
    write_json(path, asdict(provenance))
    return provenance
