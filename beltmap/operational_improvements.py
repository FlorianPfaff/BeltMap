"""Operational calibration, QC, and reporting helpers for BeltMap.

This module collects pragmatic utilities that sit around the core BeltMap
algorithm.  They intentionally avoid heavy optional dependencies so they can be
used from command-line preflight tools, validation reports, notebooks, or future
pipeline stages.  Some helpers are production-ready utilities, while others are
small reference implementations that provide a safe starting point for larger
features such as geometric rectification, streaming, or learned detector plugins.
"""

from __future__ import annotations

import csv
import hashlib
import html
import importlib
import json
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from PIL import Image, ImageFilter

FloatArray = NDArray[np.floating]
BoolArray = NDArray[np.bool_]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class BeltRegionSuggestion:
    """Suggested belt crop in full-frame image coordinates."""

    top: int
    left: int
    height: int
    width: int
    score: float
    method: str
    threshold: float
    moving_pixel_fraction: float

    @property
    def region(self) -> tuple[int, int, int, int]:
        return self.top, self.left, self.height, self.width

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HomographyModel:
    """Planar projective transform from source image points to target points."""

    matrix: NDArray[np.float64]
    source_points: tuple[tuple[float, float], ...]
    target_points: tuple[tuple[float, float], ...]

    @property
    def inverse_matrix(self) -> NDArray[np.float64]:
        return np.linalg.inv(self.matrix)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix": np.asarray(self.matrix, dtype=float).tolist(),
            "source_points": [list(p) for p in self.source_points],
            "target_points": [list(p) for p in self.target_points],
        }


@dataclass(frozen=True)
class PeriodEstimate:
    """Autocorrelation-based estimate of a belt period in pixels."""

    period_px: int
    score: float
    candidates: tuple[tuple[int, float], ...]
    method: str = "vertical-profile-autocorrelation"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AdaptiveSample:
    """One frame selected for belt-map reconstruction."""

    frame_index: int
    phase_px: float
    bin_index: int
    score: float
    coverage_gain: int
    reason: str


@dataclass(frozen=True)
class ParticleDescriptor:
    """Shape and intensity descriptor for one particle component."""

    area_px: int
    bbox_top: int
    bbox_left: int
    bbox_bottom: int
    bbox_right: int
    centroid_y: float
    centroid_x: float
    equivalent_diameter_px: float
    major_axis_px: float
    minor_axis_px: float
    orientation_rad: float
    integrated_signal: float | None
    mean_signal: float | None
    peak_signal: float | None
    extent: float


@dataclass(frozen=True)
class DetectionUncertainty:
    """Approximate centroid uncertainty from signal strength and local noise."""

    centroid_y_std_px: float | None
    centroid_x_std_px: float | None
    effective_signal: float
    effective_noise: float
    n_pixels: int


@dataclass(frozen=True)
class VelocityUncertainty:
    """Linear-fit uncertainty for a track velocity estimate."""

    slope_px_per_time: float
    intercept_px: float
    slope_std_px_per_time: float | None
    residual_std_px: float | None
    n_points: int


@dataclass(frozen=True)
class FluxSummary:
    """Experiment-level particle flux and velocity summary."""

    n_velocity_rows: int
    n_accepted_rows: int
    duration_s: float | None
    frame_count: int | None
    frame_rate_hz: float | None
    particle_flux_per_s: float | None
    median_velocity_ratio_y: float | None
    q25_velocity_ratio_y: float | None
    q75_velocity_ratio_y: float | None
    mean_velocity_y_px_per_s: float | None
    belt_velocity_px_per_s: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EventClassification:
    """Rule-based classification of a detection or track-level event."""

    label: str
    confidence: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ReviewItem:
    """One image or overlay to include in a manual QC review page."""

    frame_index: int
    image: str
    title: str
    n_detections: int = 0
    notes: str = ""


@dataclass(frozen=True)
class DatasetFileRecord:
    """Manifest row for one image file."""

    path: str
    size_bytes: int
    sha256: str
    width: int | None
    height: int | None
    mode: str | None
    mtime_ns: int


@dataclass(frozen=True)
class DatasetManifest:
    """Reproducibility manifest for an image directory."""

    root: str
    files: tuple[DatasetFileRecord, ...]
    manifest_sha256: str
    created_unix_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "files": [asdict(row) for row in self.files],
            "manifest_sha256": self.manifest_sha256,
            "created_unix_s": self.created_unix_s,
        }


@dataclass(frozen=True)
class TimestampTable:
    """Frame timestamp lookup loaded from CSV or metadata."""

    frame_to_time_s: Mapping[int, float]

    def time_for_frame(self, frame_index: int) -> float:
        try:
            return float(self.frame_to_time_s[int(frame_index)])
        except KeyError as exc:
            raise KeyError(f"No timestamp for frame {frame_index}") from exc


@dataclass
class StreamingFrameState:
    """State for directory-polling based online processing prototypes."""

    seen_paths: set[str] = field(default_factory=set)
    last_scan_unix_s: float = 0.0


@dataclass(frozen=True)
class PipelineStageResult:
    """Small provenance record for a pure pipeline stage call."""

    name: str
    elapsed_s: float
    outputs: Mapping[str, Any]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class SyntheticRandomizationConfig:
    """Domain-randomization settings for synthetic frames."""

    gain_range: tuple[float, float] = (0.9, 1.1)
    offset_range: tuple[float, float] = (-5.0, 5.0)
    gaussian_noise_sigma: float = 2.0
    blur_radius_px: float = 0.0
    scratch_count: int = 0
    scratch_intensity: float = 20.0


@dataclass(frozen=True)
class MultiCameraEvent:
    """One stitched event candidate across camera streams."""

    event_id: int
    camera_rows: tuple[dict[str, Any], ...]
    mean_time_s: float | None
    mean_belt_phase_px: float | None


# ---------------------------------------------------------------------------
# Image loading and ROI calibration
# ---------------------------------------------------------------------------


def natural_key(path: Path) -> list[int | str]:
    import re

    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(path))]


def list_image_paths(image_dir: Path, *, max_frames: int | None = None) -> list[Path]:
    paths = sorted(
        [path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS and not path.name.startswith("._")],
        key=natural_key,
    )
    if max_frames is not None and max_frames > 0:
        paths = paths[:max_frames]
    return paths


def read_gray_image(path: Path) -> NDArray[np.float64]:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float64)


def suggest_belt_region_from_frames(
    frames: Sequence[ArrayLike],
    *,
    percentile: float = 80.0,
    margin_px: int = 16,
    min_moving_fraction: float = 0.002,
) -> BeltRegionSuggestion:
    """Suggest a belt ROI from temporal motion/variance energy.

    The detector uses frame-to-frame absolute differences when at least two
    frames are supplied.  With one frame it falls back to local intensity
    variation.  The returned region is a conservative bounding box around high
    motion-energy pixels plus ``margin_px``.
    """

    arrays = [np.asarray(frame, dtype=np.float64) for frame in frames]
    if not arrays:
        raise ValueError("frames must contain at least one image")
    shape = arrays[0].shape
    if len(shape) != 2:
        raise ValueError("frames must be 2-D grayscale images")
    if any(array.shape != shape for array in arrays):
        raise ValueError("all frames must have the same shape")
    if not 0 <= percentile < 100:
        raise ValueError("percentile must be in [0, 100)")
    if margin_px < 0:
        raise ValueError("margin_px must be non-negative")

    if len(arrays) >= 2:
        diffs = [np.abs(b - a) for a, b in zip(arrays[:-1], arrays[1:])]
        energy = np.mean(diffs, axis=0)
        method = "temporal-motion-energy"
    else:
        image = arrays[0]
        gy = np.zeros_like(image)
        gx = np.zeros_like(image)
        gy[1:] = np.abs(np.diff(image, axis=0))
        gx[:, 1:] = np.abs(np.diff(image, axis=1))
        energy = gy + gx
        method = "single-frame-texture-energy"

    finite = energy[np.isfinite(energy)]
    if finite.size == 0:
        raise ValueError("motion energy contains no finite pixels")
    threshold = float(np.percentile(finite, percentile))
    moving = np.isfinite(energy) & (energy > threshold)
    moving_fraction = float(np.count_nonzero(moving) / moving.size)
    if moving_fraction < min_moving_fraction:
        top, left, height, width = 0, 0, shape[0], shape[1]
        score = 0.0
    else:
        rows, cols = np.nonzero(moving)
        top = max(0, int(rows.min()) - margin_px)
        left = max(0, int(cols.min()) - margin_px)
        bottom = min(shape[0], int(rows.max()) + margin_px + 1)
        right = min(shape[1], int(cols.max()) + margin_px + 1)
        height = bottom - top
        width = right - left
        inside = energy[top:bottom, left:right]
        score = float(np.nanmean(inside) / max(float(np.nanmean(energy)), 1e-12))
    return BeltRegionSuggestion(
        top=int(top),
        left=int(left),
        height=int(height),
        width=int(width),
        score=score,
        method=method,
        threshold=threshold,
        moving_pixel_fraction=moving_fraction,
    )


# ---------------------------------------------------------------------------
# Perspective correction / homography
# ---------------------------------------------------------------------------


def estimate_homography(
    source_points: Sequence[Sequence[float]],
    target_points: Sequence[Sequence[float]],
) -> HomographyModel:
    """Estimate a 3x3 source-to-target homography using normalized DLT."""

    src = np.asarray(source_points, dtype=np.float64)
    dst = np.asarray(target_points, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2 or src.shape[0] < 4:
        raise ValueError("source_points and target_points must be Nx2 arrays with N >= 4")

    src_norm, src_transform = _normalize_points(src)
    dst_norm, dst_transform = _normalize_points(dst)
    rows: list[list[float]] = []
    for (x, y), (u, v) in zip(src_norm, dst_norm):
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
    _, _, vh = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    h_norm = vh[-1].reshape(3, 3)
    matrix = np.linalg.inv(dst_transform) @ h_norm @ src_transform
    matrix /= matrix[2, 2]
    return HomographyModel(
        matrix=matrix.astype(np.float64),
        source_points=tuple((float(x), float(y)) for x, y in src),
        target_points=tuple((float(x), float(y)) for x, y in dst),
    )


def _normalize_points(points: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    center = np.mean(points, axis=0)
    shifted = points - center
    mean_distance = float(np.mean(np.sqrt(np.sum(shifted * shifted, axis=1))))
    scale = math.sqrt(2.0) / max(mean_distance, 1e-12)
    transform = np.array(
        [[scale, 0.0, -scale * center[0]], [0.0, scale, -scale * center[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    homogeneous = np.c_[points, np.ones(points.shape[0])]
    normalized = (transform @ homogeneous.T).T
    return normalized[:, :2], transform


def warp_perspective(
    image: ArrayLike,
    homography: HomographyModel | ArrayLike,
    output_shape: tuple[int, int],
    *,
    fill_value: float = 0.0,
    interpolation: str = "bilinear",
) -> NDArray[np.float64]:
    """Warp ``image`` into ``output_shape`` using a source-to-target homography."""

    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("warp_perspective currently expects a 2-D grayscale image")
    matrix = homography.matrix if isinstance(homography, HomographyModel) else np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("homography matrix must be 3x3")
    if interpolation not in {"nearest", "bilinear"}:
        raise ValueError("interpolation must be 'nearest' or 'bilinear'")
    out_h, out_w = output_shape
    yy, xx = np.indices((out_h, out_w), dtype=np.float64)
    target = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)], axis=0)
    source = np.linalg.inv(matrix) @ target
    w = source[2]
    finite_w = np.isfinite(w) & (np.abs(w) > 1e-12)
    sx_flat = np.full(w.shape, -1.0, dtype=np.float64)
    sy_flat = np.full(w.shape, -1.0, dtype=np.float64)
    np.divide(source[0], w, out=sx_flat, where=finite_w)
    np.divide(source[1], w, out=sy_flat, where=finite_w)
    sx = sx_flat.reshape(output_shape)
    sy = sy_flat.reshape(output_shape)
    if interpolation == "nearest":
        return _sample_nearest(arr, sy, sx, fill_value=fill_value)
    return _sample_bilinear(arr, sy, sx, fill_value=fill_value)


def _sample_nearest(image: NDArray[np.float64], y: NDArray[np.float64], x: NDArray[np.float64], *, fill_value: float) -> NDArray[np.float64]:
    yi = np.rint(y).astype(np.int64)
    xi = np.rint(x).astype(np.int64)
    valid = (0 <= yi) & (yi < image.shape[0]) & (0 <= xi) & (xi < image.shape[1])
    out = np.full(y.shape, fill_value, dtype=np.float64)
    out[valid] = image[yi[valid], xi[valid]]
    return out


def _sample_bilinear(image: NDArray[np.float64], y: NDArray[np.float64], x: NDArray[np.float64], *, fill_value: float) -> NDArray[np.float64]:
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = y0 + 1
    x1 = x0 + 1
    valid = (0 <= y0) & (y1 < image.shape[0]) & (0 <= x0) & (x1 < image.shape[1])
    out = np.full(y.shape, fill_value, dtype=np.float64)
    wy = y - y0
    wx = x - x0
    out[valid] = (
        (1 - wy[valid]) * (1 - wx[valid]) * image[y0[valid], x0[valid]]
        + (1 - wy[valid]) * wx[valid] * image[y0[valid], x1[valid]]
        + wy[valid] * (1 - wx[valid]) * image[y1[valid], x0[valid]]
        + wy[valid] * wx[valid] * image[y1[valid], x1[valid]]
    )
    return out


# ---------------------------------------------------------------------------
# Belt-period estimation and adaptive sampling
# ---------------------------------------------------------------------------


def estimate_period_from_profile(
    profile: ArrayLike,
    *,
    min_period_px: int = 8,
    max_period_px: int | None = None,
    top_k: int = 5,
) -> PeriodEstimate:
    """Estimate a periodicity from a 1-D belt texture profile."""

    values = np.asarray(profile, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size < 2 * min_period_px:
        raise ValueError("profile is too short for the requested minimum period")
    max_period = values.size // 2 if max_period_px is None else min(int(max_period_px), values.size - 1)
    if min_period_px <= 0 or max_period < min_period_px:
        raise ValueError("invalid period search range")
    centered = values - float(np.mean(values))
    std = float(np.std(centered))
    if std <= 0:
        raise ValueError("profile has no variation")
    centered /= std
    candidates: list[tuple[int, float]] = []
    for period in range(int(min_period_px), int(max_period) + 1):
        a = centered[:-period]
        b = centered[period:]
        denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
        score = 0.0 if denom <= 0 else float(np.sum(a * b) / denom)
        candidates.append((period, score))
    candidates.sort(key=lambda item: item[1], reverse=True)
    best_period, best_score = candidates[0]
    return PeriodEstimate(
        period_px=int(best_period),
        score=float(best_score),
        candidates=tuple((int(p), float(s)) for p, s in candidates[:top_k]),
    )


def estimate_period_from_belt_map(
    belt_map: ArrayLike,
    *,
    min_period_px: int = 8,
    max_period_px: int | None = None,
) -> PeriodEstimate:
    """Estimate belt period from the vertical texture profile of a reconstructed map."""

    arr = np.asarray(belt_map, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("belt_map must be a 2-D array")
    gradient = np.zeros_like(arr)
    gradient[1:] = np.abs(np.diff(arr, axis=0))
    profile = np.nanmean(gradient, axis=1)
    return estimate_period_from_profile(profile, min_period_px=min_period_px, max_period_px=max_period_px)


def select_adaptive_map_frames(
    phases_px: Sequence[float],
    *,
    map_height_px: int,
    sample_count: int,
    crop_height_px: int = 1,
    quality_scores: Sequence[float] | None = None,
    bin_count: int | None = None,
) -> list[AdaptiveSample]:
    """Select frames that cover belt-map phase bins with optional quality weighting."""

    phases = np.asarray(phases_px, dtype=np.float64)
    if phases.size == 0:
        return []
    if map_height_px <= 0 or sample_count <= 0 or crop_height_px <= 0:
        raise ValueError("map_height_px, sample_count, and crop_height_px must be positive")
    scores = np.ones(phases.size, dtype=np.float64) if quality_scores is None else np.asarray(quality_scores, dtype=np.float64)
    if scores.shape != phases.shape:
        raise ValueError("quality_scores must have one value per phase")
    bins = int(bin_count or min(map_height_px, max(sample_count * 2, 8)))
    occupied = np.zeros(bins, dtype=np.int64)
    selected: list[AdaptiveSample] = []
    remaining = set(range(phases.size))
    phase_bins = np.floor((np.mod(phases, map_height_px) / map_height_px) * bins).astype(int)
    phase_bins = np.clip(phase_bins, 0, bins - 1)
    span_bins = max(1, int(math.ceil(crop_height_px / map_height_px * bins)))

    while remaining and len(selected) < min(sample_count, phases.size):
        best: tuple[float, int, int, int] | None = None
        for index in remaining:
            start = int(phase_bins[index])
            covered = (start + np.arange(span_bins)) % bins
            gain = int(np.count_nonzero(occupied[covered] == 0))
            quality = float(scores[index]) if np.isfinite(scores[index]) else 0.0
            objective = gain + 0.001 * quality
            key = (objective, gain, quality, -index)
            if best is None or key > best:
                best = key
                best_index = index
                best_covered = covered
        assert best is not None
        remaining.remove(best_index)
        occupied[best_covered] += 1
        selected.append(
            AdaptiveSample(
                frame_index=int(best_index),
                phase_px=float(phases[best_index]),
                bin_index=int(phase_bins[best_index]),
                score=float(scores[best_index]) if np.isfinite(scores[best_index]) else 0.0,
                coverage_gain=int(best[1]),
                reason="new-phase-coverage" if best[1] > 0 else "quality-fill",
            )
        )
    return selected


# ---------------------------------------------------------------------------
# Masks, residual statistics, and threshold selection
# ---------------------------------------------------------------------------


def load_ignore_mask(path: Path, *, expected_shape: tuple[int, int] | None = None) -> BoolArray:
    """Load an ignore mask from an image. Nonzero pixels are ignored."""

    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"), dtype=np.uint8) > 0
    if expected_shape is not None and mask.shape != expected_shape:
        raise ValueError(f"ignore mask shape {mask.shape} does not match expected shape {expected_shape}")
    return mask


def apply_ignore_mask(valid_mask: ArrayLike, ignore_mask: ArrayLike | None) -> BoolArray:
    valid = np.asarray(valid_mask, dtype=bool)
    if ignore_mask is None:
        return valid.copy()
    ignore = np.asarray(ignore_mask, dtype=bool)
    if ignore.shape != valid.shape:
        raise ValueError("ignore_mask must have the same shape as valid_mask")
    return valid & ~ignore


def belt_edge_ignore_mask(
    shape: tuple[int, int],
    *,
    top_px: int = 0,
    bottom_px: int = 0,
    left_px: int = 0,
    right_px: int = 0,
) -> BoolArray:
    """Return a boolean ignore mask for fixed belt-edge margins."""

    height, width = shape
    mask = np.zeros(shape, dtype=bool)
    if top_px > 0:
        mask[: min(height, top_px), :] = True
    if bottom_px > 0:
        mask[max(0, height - bottom_px) :, :] = True
    if left_px > 0:
        mask[:, : min(width, left_px)] = True
    if right_px > 0:
        mask[:, max(0, width - right_px) :] = True
    return mask


def particle_density_score(
    residual: ArrayLike,
    *,
    threshold: float,
    polarity: str = "bright",
    mask: ArrayLike | None = None,
) -> float:
    """Return the share of valid pixels that look particle-contaminated."""

    values = np.asarray(residual, dtype=np.float64)
    valid = np.isfinite(values)
    if mask is not None:
        user_mask = np.asarray(mask, dtype=bool)
        if user_mask.shape != values.shape:
            raise ValueError("mask must have the same shape as residual")
        valid &= user_mask
    if not np.any(valid):
        return 1.0
    signal = _polarity_signal(values, polarity)
    return float(np.count_nonzero(valid & (signal > threshold)) / np.count_nonzero(valid))


def rank_frames_by_particle_density(
    residuals: Sequence[ArrayLike],
    *,
    threshold: float,
    polarity: str = "bright",
) -> list[tuple[int, float]]:
    scores = [(index, particle_density_score(residual, threshold=threshold, polarity=polarity)) for index, residual in enumerate(residuals)]
    return sorted(scores, key=lambda item: item[1])


def recommend_threshold(
    residual: ArrayLike,
    *,
    expected_false_pixels_per_frame: float = 1.0,
    polarity: str = "bright",
    mask: ArrayLike | None = None,
) -> float:
    """Recommend a residual threshold from an empirical tail quantile."""

    values = np.asarray(residual, dtype=np.float64)
    valid = np.isfinite(values)
    if mask is not None:
        user_mask = np.asarray(mask, dtype=bool)
        if user_mask.shape != values.shape:
            raise ValueError("mask must have the same shape as residual")
        valid &= user_mask
    signal = _polarity_signal(values, polarity)[valid]
    if signal.size == 0:
        raise ValueError("no valid residual values")
    tail_probability = min(max(expected_false_pixels_per_frame / signal.size, 0.0), 1.0)
    quantile = 1.0 - tail_probability
    return float(np.quantile(signal, quantile))


def empirical_p_values(
    values: ArrayLike,
    *,
    background_values: ArrayLike | None = None,
    polarity: str = "bright",
) -> NDArray[np.float64]:
    """Assign empirical upper-tail p-values to residual-like scores."""

    arr = np.asarray(values, dtype=np.float64)
    bg = arr if background_values is None else np.asarray(background_values, dtype=np.float64)
    bg_signal = _polarity_signal(bg, polarity).ravel()
    bg_signal = np.sort(bg_signal[np.isfinite(bg_signal)])
    if bg_signal.size == 0:
        raise ValueError("background_values contain no finite values")
    signal = _polarity_signal(arr, polarity)
    ranks = np.searchsorted(bg_signal, signal, side="left")
    p = (bg_signal.size - ranks + 1) / (bg_signal.size + 1)
    p[~np.isfinite(signal)] = np.nan
    return p.astype(np.float64)


def fdr_threshold_from_p_values(p_values: ArrayLike, scores: ArrayLike, *, alpha: float = 0.01) -> float | None:
    """Return the minimum score accepted by Benjamini-Hochberg FDR control."""

    p = np.asarray(p_values, dtype=np.float64).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    valid = np.isfinite(p) & np.isfinite(s)
    p = p[valid]
    s = s[valid]
    if p.size == 0:
        return None
    order = np.argsort(p)
    thresholds = alpha * (np.arange(1, p.size + 1) / p.size)
    accepted = p[order] <= thresholds
    if not np.any(accepted):
        return None
    accepted_scores = s[order][accepted]
    return float(np.min(accepted_scores))


def _polarity_signal(values: NDArray[np.float64], polarity: str) -> NDArray[np.float64]:
    if polarity == "bright":
        return values
    if polarity == "dark":
        return -values
    if polarity in {"absolute", "abs"}:
        return np.abs(values)
    raise ValueError("polarity must be bright, dark, or absolute")


# ---------------------------------------------------------------------------
# Component splitting, particle descriptors, and uncertainty
# ---------------------------------------------------------------------------


def split_merged_components(
    mask: ArrayLike,
    *,
    max_area_px: int,
    min_gap_px: int = 1,
) -> list[BoolArray]:
    """Split large components by empty projection gaps when possible."""

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("mask must be 2-D")
    components = _connected_component_masks(binary)
    result: list[BoolArray] = []
    for component in components:
        if int(np.count_nonzero(component)) <= max_area_px:
            result.append(component)
            continue
        result.extend(_split_component_by_projection(component, min_gap_px=min_gap_px))
    return result


def _split_component_by_projection(component: BoolArray, *, min_gap_px: int) -> list[BoolArray]:
    rows, cols = np.nonzero(component)
    if rows.size == 0:
        return []
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(cols.min()), int(cols.max()) + 1
    crop = component[top:bottom, left:right]
    row_projection = np.count_nonzero(crop, axis=1)
    col_projection = np.count_nonzero(crop, axis=0)
    row_splits = _projection_split_indices(row_projection, min_gap_px=min_gap_px)
    col_splits = _projection_split_indices(col_projection, min_gap_px=min_gap_px)
    if len(col_splits) >= len(row_splits) and len(col_splits) > 1:
        chunks = [(slice(None), slice(start, stop)) for start, stop in col_splits]
    elif len(row_splits) > 1:
        chunks = [(slice(start, stop), slice(None)) for start, stop in row_splits]
    else:
        return [component]
    pieces: list[BoolArray] = []
    for row_slice, col_slice in chunks:
        piece_crop = np.zeros_like(crop)
        piece_crop[row_slice, col_slice] = crop[row_slice, col_slice]
        if np.any(piece_crop):
            piece = np.zeros_like(component)
            piece[top:bottom, left:right] = piece_crop
            pieces.extend(_connected_component_masks(piece))
    return pieces or [component]


def _projection_split_indices(projection: NDArray[np.integer], *, min_gap_px: int) -> list[tuple[int, int]]:
    occupied = projection > 0
    if not np.any(occupied):
        return []
    segments: list[tuple[int, int]] = []
    start: int | None = None
    gap_start: int | None = None
    for index, is_occupied in enumerate(occupied):
        if is_occupied:
            if start is None:
                start = index
            gap_start = None
        elif start is not None and gap_start is None:
            gap_start = index
        if start is not None and gap_start is not None and index - gap_start + 1 >= min_gap_px:
            segments.append((start, gap_start))
            start = None
            gap_start = None
    if start is not None:
        segments.append((start, len(projection)))
    return segments


def particle_descriptor_from_mask(mask: ArrayLike, *, signal: ArrayLike | None = None) -> ParticleDescriptor:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not np.any(binary):
        raise ValueError("mask must be a non-empty 2-D component mask")
    rows, cols = np.nonzero(binary)
    values = None if signal is None else np.asarray(signal, dtype=np.float64)[rows, cols]
    weights = np.ones(rows.size, dtype=np.float64)
    if values is not None:
        weights = np.clip(values, 0.0, None)
        if float(np.sum(weights)) <= 0:
            weights = np.ones(rows.size, dtype=np.float64)
    weight_sum = float(np.sum(weights))
    cy = float(np.sum(rows * weights) / weight_sum)
    cx = float(np.sum(cols * weights) / weight_sum)
    centered = np.vstack([rows - cy, cols - cx])
    covariance = (centered * weights[None, :]) @ centered.T / max(weight_sum, 1e-12)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    major = float(4.0 * math.sqrt(max(eigvals[0], 0.0)))
    minor = float(4.0 * math.sqrt(max(eigvals[-1], 0.0)))
    orientation = float(math.atan2(eigvecs[0, 0], eigvecs[1, 0]))
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(cols.min()), int(cols.max()) + 1
    area = int(rows.size)
    return ParticleDescriptor(
        area_px=area,
        bbox_top=top,
        bbox_left=left,
        bbox_bottom=bottom,
        bbox_right=right,
        centroid_y=cy,
        centroid_x=cx,
        equivalent_diameter_px=float(2.0 * math.sqrt(area / math.pi)),
        major_axis_px=major,
        minor_axis_px=minor,
        orientation_rad=orientation,
        integrated_signal=None if values is None else float(np.sum(values)),
        mean_signal=None if values is None else float(np.mean(values)),
        peak_signal=None if values is None else float(np.max(values)),
        extent=float(area / ((bottom - top) * (right - left))),
    )


def estimate_centroid_uncertainty(
    component_mask: ArrayLike,
    *,
    signal: ArrayLike | None = None,
    local_noise: ArrayLike | float | None = None,
) -> DetectionUncertainty:
    mask = np.asarray(component_mask, dtype=bool)
    if not np.any(mask):
        return DetectionUncertainty(None, None, 0.0, 0.0, 0)
    rows, cols = np.nonzero(mask)
    sig = np.ones(rows.size, dtype=np.float64) if signal is None else np.clip(np.asarray(signal, dtype=np.float64)[rows, cols], 0, None)
    effective_signal = float(np.sum(sig))
    if np.isscalar(local_noise) and local_noise is not None:
        noise = np.full(rows.size, float(local_noise), dtype=np.float64)
    elif local_noise is None:
        noise = np.ones(rows.size, dtype=np.float64)
    else:
        noise = np.asarray(local_noise, dtype=np.float64)[rows, cols]
    effective_noise = float(np.sqrt(np.mean(np.square(noise[np.isfinite(noise)])))) if np.any(np.isfinite(noise)) else 0.0
    if effective_signal <= 0 or effective_noise <= 0:
        return DetectionUncertainty(None, None, effective_signal, effective_noise, int(rows.size))
    descriptor = particle_descriptor_from_mask(mask, signal=signal)
    snr = effective_signal / max(effective_noise * math.sqrt(rows.size), 1e-12)
    cy_std = descriptor.minor_axis_px / max(2.0 * snr, 1e-12)
    cx_std = descriptor.major_axis_px / max(2.0 * snr, 1e-12)
    return DetectionUncertainty(float(cy_std), float(cx_std), effective_signal, effective_noise, int(rows.size))


def robust_velocity_fit(times: Sequence[float], positions: Sequence[float]) -> VelocityUncertainty:
    """Estimate slope with median pairwise velocity and a robust residual scale."""

    t = np.asarray(times, dtype=np.float64)
    y = np.asarray(positions, dtype=np.float64)
    valid = np.isfinite(t) & np.isfinite(y)
    t = t[valid]
    y = y[valid]
    if t.size < 2 or np.unique(t).size < 2:
        raise ValueError("at least two distinct finite times are required")
    per_point_slopes: list[float] = []
    for i in range(t.size):
        dt = t - t[i]
        dy = y - y[i]
        good = dt != 0
        if np.any(good):
            per_point_slopes.append(float(np.median(dy[good] / dt[good])))
    slope = float(np.median(per_point_slopes))
    intercept = float(np.median(y - slope * t))
    residuals = y - (slope * t + intercept)
    mad = float(np.median(np.abs(residuals - np.median(residuals))))
    residual_std = 1.4826 * mad
    denominator = float(np.sum((t - np.mean(t)) ** 2))
    slope_std = None if denominator <= 0 else float(residual_std / math.sqrt(denominator))
    return VelocityUncertainty(slope, intercept, slope_std, residual_std, int(t.size))


# ---------------------------------------------------------------------------
# Timing, metadata, streaming, and map updates
# ---------------------------------------------------------------------------


def read_image_metadata(path: Path) -> dict[str, Any]:
    """Read basic image metadata and safe EXIF fields."""

    stat = path.stat()
    with Image.open(path) as image:
        exif = {}
        try:
            raw_exif = image.getexif()
            exif = {str(key): str(value) for key, value in raw_exif.items()}
        except Exception:
            exif = {}
        return {
            "path": str(path),
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "format": image.format,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "exif": exif,
        }


def load_timestamps_csv(path: Path, *, frame_column: str = "frame_index", time_column: str = "time_s") -> TimestampTable:
    mapping: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            mapping[int(row[frame_column])] = float(row[time_column])
    if not mapping:
        raise ValueError("timestamp CSV contains no rows")
    return TimestampTable(mapping)


def discover_new_stream_frames(image_dir: Path, state: StreamingFrameState, *, max_new: int | None = None) -> list[Path]:
    paths = list_image_paths(image_dir)
    new_paths = [path for path in paths if str(path) not in state.seen_paths]
    if max_new is not None:
        new_paths = new_paths[:max_new]
    for path in new_paths:
        state.seen_paths.add(str(path))
    state.last_scan_unix_s = time.time()
    return new_paths


def incremental_update_map(
    current_map: ArrayLike,
    observation: ArrayLike,
    valid_mask: ArrayLike,
    *,
    learning_rate: float = 0.01,
) -> NDArray[np.float64]:
    """Slowly update a belt map from trusted non-particle observations."""

    old = np.asarray(current_map, dtype=np.float64)
    obs = np.asarray(observation, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if old.shape != obs.shape or old.shape != valid.shape:
        raise ValueError("current_map, observation, and valid_mask must have the same shape")
    if not 0 <= learning_rate <= 1:
        raise ValueError("learning_rate must be in [0, 1]")
    updated = old.copy()
    updated[valid] = (1.0 - learning_rate) * old[valid] + learning_rate * obs[valid]
    return updated


# ---------------------------------------------------------------------------
# Manifest, provenance, quality tooling, and templates
# ---------------------------------------------------------------------------


def dataset_manifest(image_dir: Path, *, hash_chunk_size: int = 1024 * 1024) -> DatasetManifest:
    records: list[DatasetFileRecord] = []
    for path in list_image_paths(image_dir):
        stat = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(hash_chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
        width = height = None
        mode = None
        try:
            with Image.open(path) as image:
                width, height, mode = image.width, image.height, image.mode
        except Exception:
            pass
        records.append(
            DatasetFileRecord(
                path=str(path.relative_to(image_dir)),
                size_bytes=int(stat.st_size),
                sha256=digest.hexdigest(),
                width=width,
                height=height,
                mode=mode,
                mtime_ns=int(stat.st_mtime_ns),
            )
        )
    payload = json.dumps([asdict(record) for record in records], sort_keys=True).encode("utf-8")
    return DatasetManifest(
        root=str(image_dir),
        files=tuple(records),
        manifest_sha256=hashlib.sha256(payload).hexdigest(),
        created_unix_s=time.time(),
    )


def runtime_provenance(*, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    data = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "argv": sys.argv,
        "numpy_version": np.__version__,
        "pillow_version": Image.__version__,
        "env": {key: os.environ.get(key) for key in sorted(os.environ) if key.startswith("BELTMAP") or key in {"GITHUB_SHA", "GITHUB_REF"}},
    }
    if extra:
        data.update(dict(extra))
    return data


def write_workflow_templates(output_dir: Path) -> dict[str, Path]:
    """Write minimal Snakemake and Nextflow templates for BeltMap runs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    snakefile = output_dir / "Snakefile"
    snakefile.write_text(
        """configfile: \"beltmap-workflow-config.yaml\"\n\nrule all:\n    input:\n        \"outputs/validation_report.md\"\n\nrule apply_beltmap:\n    input:\n        config[\"config\"]\n    output:\n        \"outputs/metadata.json\"\n    shell:\n        \"beltmap-apply --config {input} --print-config\"\n\nrule validate:\n    input:\n        \"outputs/metadata.json\"\n    output:\n        \"outputs/validation_report.md\"\n    shell:\n        \"beltmap-validate --output-dir outputs\"\n""",
        encoding="utf-8",
    )
    nextflow = output_dir / "main.nf"
    nextflow.write_text(
        """params.config = 'beltmap.toml'\n\nprocess APPLY_BELTMAP {\n  output:\n  path 'outputs'\n\n  script:\n  \"\"\"\n  beltmap-apply --config ${params.config} --print-config\n  beltmap-validate --output-dir outputs\n  \"\"\"\n}\n\nworkflow { APPLY_BELTMAP() }\n""",
        encoding="utf-8",
    )
    return {"snakemake": snakefile, "nextflow": nextflow}


def write_container_templates(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dockerfile = output_dir / "Dockerfile"
    dockerfile.write_text(
        """FROM python:3.12-slim\nWORKDIR /app\nCOPY . /app\nRUN python -m pip install --upgrade pip && python -m pip install -e '.[speed]'\nENTRYPOINT [\"beltmap-apply\"]\n""",
        encoding="utf-8",
    )
    apptainer = output_dir / "beltmap.def"
    apptainer.write_text(
        """Bootstrap: docker\nFrom: python:3.12-slim\n\n%post\n    python -m pip install --upgrade pip\n    cd /app && python -m pip install -e '.[speed]'\n\n%files\n    . /app\n\n%runscript\n    exec beltmap-apply \"$@\"\n""",
        encoding="utf-8",
    )
    return {"dockerfile": dockerfile, "apptainer": apptainer}


def write_quality_tooling_templates(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    precommit = output_dir / ".pre-commit-config.yaml"
    precommit.write_text(
        """repos:\n  - repo: https://github.com/astral-sh/ruff-pre-commit\n    rev: v0.8.0\n    hooks:\n      - id: ruff\n      - id: ruff-format\n""",
        encoding="utf-8",
    )
    ruff = output_dir / "ruff.toml"
    ruff.write_text(
        """line-length = 100\ntarget-version = \"py310\"\n[lint]\nselect = [\"E\", \"F\", \"I\", \"B\"]\n""",
        encoding="utf-8",
    )
    return {"precommit": precommit, "ruff": ruff}


# ---------------------------------------------------------------------------
# Failure modes, event classification, reporting, and science exports
# ---------------------------------------------------------------------------


def classify_failure_modes(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return rule-based QC warnings from run summary metrics."""

    warnings: list[dict[str, Any]] = []

    def add(code: str, severity: str, message: str) -> None:
        warnings.append({"code": code, "severity": severity, "message": message})

    if _summary_metric(summary, "phase_boundary_fraction", default=0.0) > 0.1:
        add("registration-search-boundary", "high", "Many phase corrections hit the search boundary; increase registration radius or improve velocity calibration.")
    if _summary_metric(summary, "registration_score_median", default=1.0) < 0.2:
        add("low-registration-score", "high", "Median phase-registration score is low; check belt texture, crop, and map quality.")
    if _summary_metric(summary, "small_component_share_area_le_8", default=0.0) > 0.5:
        add("many-small-components", "medium", "Most detections are tiny; raise threshold or enable shape/edge gates.")
    if _summary_metric(summary, "velocity_ratio_share_0_to_1", default=1.0) < 0.5:
        add("implausible-velocity-ratios", "high", "Few velocities are in the expected [0, 1] belt-relative range.")
    if _summary_metric(summary, "map_low_coverage_fraction", default=0.0) > 0.05:
        add("low-map-coverage", "medium", "Some belt-map pixels have low coverage; increase sample frames or use adaptive sampling.")
    if _summary_metric(summary, "track_fragmentation", default=0.0) > 0.5:
        add("track-fragmentation", "medium", "Tracks are fragmented; use wider PyRecEst matching or tracklet stitching.")
    return warnings


def _summary_metric(summary: Mapping[str, Any], key: str, *, default: float) -> float:
    value = _finite_float(summary.get(key))
    return default if value is None else value


def classify_event(
    *,
    recurrent_overlap_fraction: float | None = None,
    velocity_ratio_y: float | None = None,
    peak_signal: float | None = None,
    map_uncertainty: float | None = None,
) -> EventClassification:
    reasons: list[str] = []
    score = 0.5
    label = "uncertain"
    if recurrent_overlap_fraction is not None and recurrent_overlap_fraction > 0.5:
        label = "belt-fixed-artifact"
        score += 0.3
        reasons.append("high recurrent belt-coordinate overlap")
    if velocity_ratio_y is not None and 0.0 <= velocity_ratio_y <= 1.1:
        if label == "uncertain":
            label = "loose-particle"
        score += 0.2
        reasons.append("physically plausible velocity ratio")
    elif velocity_ratio_y is not None:
        reasons.append("velocity ratio outside expected range")
        score -= 0.15
    if peak_signal is not None and peak_signal >= 5.0:
        score += 0.1
        reasons.append("strong residual peak")
    if map_uncertainty is not None and map_uncertainty > 5.0:
        if label == "uncertain":
            label = "map-uncertainty-artifact"
        score -= 0.1
        reasons.append("high rendered map uncertainty")
    return EventClassification(label=label, confidence=float(np.clip(score, 0.0, 1.0)), reasons=tuple(reasons))


def summarize_flux(
    velocity_rows: Sequence[Mapping[str, Any]],
    *,
    frame_count: int | None = None,
    frame_rate_hz: float | None = None,
    duration_s: float | None = None,
    belt_velocity_px_per_s: float | None = None,
    accepted_only: bool = False,
) -> FluxSummary:
    rows = list(velocity_rows)
    if accepted_only:
        rows = [row for row in rows if str(row.get("accepted", "true")).lower() in {"1", "true", "yes", "on"}]
    ratios = [_finite_float(row.get("velocity_ratio_y")) for row in rows]
    ratios = [value for value in ratios if value is not None]
    velocities = [_finite_float(row.get("velocity_y_px_per_frame")) for row in rows]
    velocities = [value for value in velocities if value is not None]
    if duration_s is None and frame_count is not None and frame_rate_hz is not None and frame_rate_hz > 0:
        duration_s = frame_count / frame_rate_hz
    flux = None if duration_s is None or duration_s <= 0 else len(rows) / duration_s
    mean_v = None
    if velocities:
        multiplier = frame_rate_hz if frame_rate_hz is not None else 1.0
        mean_v = float(np.mean(velocities) * multiplier)
    return FluxSummary(
        n_velocity_rows=len(velocity_rows),
        n_accepted_rows=len(rows),
        duration_s=None if duration_s is None else float(duration_s),
        frame_count=frame_count,
        frame_rate_hz=frame_rate_hz,
        particle_flux_per_s=flux,
        median_velocity_ratio_y=None if not ratios else float(np.median(ratios)),
        q25_velocity_ratio_y=None if not ratios else float(np.percentile(ratios, 25)),
        q75_velocity_ratio_y=None if not ratios else float(np.percentile(ratios, 75)),
        mean_velocity_y_px_per_s=mean_v,
        belt_velocity_px_per_s=belt_velocity_px_per_s,
    )


def write_science_exports(
    output_dir: Path,
    velocity_rows: Sequence[Mapping[str, Any]],
    *,
    frame_count: int | None = None,
    frame_rate_hz: float | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_flux(velocity_rows, frame_count=frame_count, frame_rate_hz=frame_rate_hz)
    summary_path = output_dir / "particle_flux_summary.json"
    summary_path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    distribution_path = output_dir / "velocity_distribution.csv"
    with distribution_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["track_id", "velocity_ratio_y", "velocity_y_px_per_frame"])
        writer.writeheader()
        for row in velocity_rows:
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames or []})
    return {"flux_summary": summary_path, "velocity_distribution": distribution_path}


def build_review_items(overlay_paths: Sequence[Path], *, detection_counts: Mapping[int, int] | None = None) -> list[ReviewItem]:
    items: list[ReviewItem] = []
    counts = detection_counts or {}
    for index, path in enumerate(sorted(overlay_paths, key=natural_key)):
        frame_index = _last_int_in_stem(path, fallback=index)
        items.append(ReviewItem(frame_index=frame_index, image=str(path), title=path.name, n_detections=int(counts.get(frame_index, 0))))
    return items


def write_html_review(path: Path, items: Sequence[ReviewItem], *, title: str = "BeltMap review") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in items:
        rows.append(
            f"<section><h2>{html.escape(item.title)}</h2>"
            f"<p>frame={item.frame_index}, detections={item.n_detections}</p>"
            f"<img src='{html.escape(item.image)}' style='max-width:100%;height:auto'>"
            f"<p>{html.escape(item.notes)}</p></section>"
        )
    path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><h1>{html.escape(title)}</h1>"
        "<p>Use this page to mark false positives, missed particles, bad tracks, and ignore regions in a separate annotation file.</p>"
        + "\n".join(rows),
        encoding="utf-8",
    )
    return path


def write_html_qc_report(path: Path, summary: Mapping[str, Any], items: Sequence[ReviewItem] = ()) -> Path:
    warnings = classify_failure_modes(summary)
    body = ["<!doctype html><meta charset='utf-8'><title>BeltMap QC</title><h1>BeltMap QC report</h1>"]
    body.append("<h2>Summary</h2><table>")
    for key, value in sorted(summary.items()):
        body.append(f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>")
    body.append("</table><h2>Warnings</h2><ul>")
    for warning in warnings:
        body.append(f"<li><b>{html.escape(warning['severity'])}</b> {html.escape(warning['code'])}: {html.escape(warning['message'])}</li>")
    body.append("</ul>")
    if items:
        body.append("<h2>Review images</h2>")
        for item in items:
            body.append(f"<h3>{html.escape(item.title)}</h3><img src='{html.escape(item.image)}' style='max-width:100%'>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Pipeline, plugins, synthetic randomization, and multi-camera events
# ---------------------------------------------------------------------------


def run_pipeline_stage(name: str, function: Callable[..., Any], /, *args: Any, **kwargs: Any) -> PipelineStageResult:
    start = time.perf_counter()
    output = function(*args, **kwargs)
    elapsed = time.perf_counter() - start
    outputs = output if isinstance(output, Mapping) else {"result": output}
    return PipelineStageResult(name=name, elapsed_s=elapsed, outputs=outputs, metadata=runtime_provenance())


def load_detector_plugin(spec: str) -> Callable[..., Any]:
    """Load a detector plugin from ``module:function``."""

    if ":" not in spec:
        raise ValueError("plugin spec must be 'module:function'")
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    if not callable(function):
        raise TypeError(f"{spec} is not callable")
    return function


def run_detector_plugin(spec: str, residual: ArrayLike, **kwargs: Any) -> Any:
    return load_detector_plugin(spec)(residual, **kwargs)


def randomize_synthetic_frame(frame: ArrayLike, config: SyntheticRandomizationConfig, *, rng: np.random.Generator | None = None) -> NDArray[np.float64]:
    generator = rng or np.random.default_rng()
    image = np.asarray(frame, dtype=np.float64).copy()
    gain = generator.uniform(*config.gain_range)
    offset = generator.uniform(*config.offset_range)
    image = image * gain + offset
    if config.gaussian_noise_sigma > 0:
        image += generator.normal(0.0, config.gaussian_noise_sigma, size=image.shape)
    if config.scratch_count > 0:
        for _ in range(config.scratch_count):
            row = int(generator.integers(0, image.shape[0]))
            image[row : min(image.shape[0], row + 2), :] += config.scratch_intensity
    if config.blur_radius_px > 0:
        pil = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
        image = np.asarray(pil.filter(ImageFilter.GaussianBlur(config.blur_radius_px)), dtype=np.float64)
    return image


def stitch_multicamera_events(
    rows_by_camera: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    time_tolerance_s: float = 0.05,
    phase_tolerance_px: float = 10.0,
) -> list[MultiCameraEvent]:
    """Greedily stitch camera-specific detections by time and belt phase."""

    candidates: list[tuple[str, Mapping[str, Any]]] = []
    for camera, rows in rows_by_camera.items():
        for row in rows:
            enriched = dict(row)
            enriched["camera"] = camera
            candidates.append((camera, enriched))
    used: set[int] = set()
    events: list[MultiCameraEvent] = []
    for i, (_camera, row) in enumerate(candidates):
        if i in used:
            continue
        group = [dict(row)]
        used.add(i)
        time_i = _finite_float(row.get("time_s"))
        phase_i = _finite_float(row.get("belt_phase_px", row.get("phase_px")))
        for j, (_other_camera, other) in enumerate(candidates):
            if j in used:
                continue
            time_j = _finite_float(other.get("time_s"))
            phase_j = _finite_float(other.get("belt_phase_px", other.get("phase_px")))
            time_ok = time_i is None or time_j is None or abs(time_i - time_j) <= time_tolerance_s
            phase_ok = phase_i is None or phase_j is None or abs(phase_i - phase_j) <= phase_tolerance_px
            if time_ok and phase_ok:
                group.append(dict(other))
                used.add(j)
        times = [_finite_float(item.get("time_s")) for item in group]
        phases = [_finite_float(item.get("belt_phase_px", item.get("phase_px"))) for item in group]
        events.append(
            MultiCameraEvent(
                event_id=len(events),
                camera_rows=tuple(group),
                mean_time_s=_mean_optional(times),
                mean_belt_phase_px=_mean_optional(phases),
            )
        )
    return events


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _connected_component_masks(mask: BoolArray) -> list[BoolArray]:
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[BoolArray] = []
    height, width = mask.shape
    for start_row, start_col in np.argwhere(mask):
        row = int(start_row)
        col = int(start_col)
        if visited[row, col]:
            continue
        component = np.zeros(mask.shape, dtype=bool)
        stack = [(row, col)]
        visited[row, col] = True
        while stack:
            y, x = stack.pop()
            component[y, x] = True
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        components.append(component)
    return components


def _finite_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _last_int_in_stem(path: Path, *, fallback: int) -> int:
    import re

    matches = re.findall(r"\d+", path.stem)
    return int(matches[-1]) if matches else fallback


def _mean_optional(values: Iterable[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return None if not finite else float(statistics.mean(finite))
