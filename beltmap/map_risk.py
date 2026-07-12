"""Belt-coordinate support/risk diagnostics for BeltMap detections.

The belt map is interpolated wherever no sampled, unmasked pixel contributed to a
given belt-coordinate location.  Those locations are useful for diagnostics and
can also be used as a conservative detection prior: components supported mostly
by interpolated or weakly observed belt-map pixels are more likely to be map
artifacts than components over well-observed texture.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from .phase import render_belt_view
from .tracking import ParticleDetection


@dataclass(frozen=True)
class BeltMapRiskMaps:
    """Belt-coordinate support and risk images derived from map accumulation."""

    support: np.ndarray
    observed_mask: np.ndarray
    interpolated_mask: np.ndarray
    low_support_mask: np.ndarray
    risk: np.ndarray


@dataclass(frozen=True)
class MapRiskDetectionScore:
    """One detection scored against the belt-coordinate support/risk maps."""

    detection: ParticleDetection
    rejected: bool


def load_belt_map_support(path: Path, *, map_shape: tuple[int, int]) -> np.ndarray:
    """Load a previously written ``belt_map_support.npy`` with shape checks."""

    support = _support_array(np.load(path), "REUSE_MAP_SUPPORT_PATH")
    if support.shape != map_shape:
        raise ValueError(
            "reused belt-map support shape does not match belt_map.npy: "
            f"{support.shape} != {map_shape}"
        )
    return support.astype(np.float32, copy=False)


def compute_belt_map_risk_maps(
    support: np.ndarray,
    *,
    min_support: float = 1.0,
) -> BeltMapRiskMaps:
    """Return belt-coordinate observation support and normalized risk maps.

    ``support`` is the accumulation weight per belt-coordinate pixel.  With the
    default fractional splat this is an effective observation count, not
    necessarily an integer.  The normalized risk is 1 for interpolated pixels and
    decreases linearly to 0 once ``support >= min_support``.
    """

    support_arr = _support_array(support, "support")
    min_support_value = _finite_float(min_support, "min_support")
    if min_support_value < 0.0:
        raise ValueError("min_support must be finite and non-negative")

    observed = np.isfinite(support_arr) & (support_arr > 0.0)
    clean_support = np.where(observed, support_arr, 0.0).astype(np.float32, copy=False)
    interpolated = ~observed
    if min_support_value <= 0.0:
        low_support = interpolated.copy()
        risk = interpolated.astype(np.float32)
    else:
        low_support = clean_support < min_support_value
        deficit = (min_support_value - clean_support) / min_support_value
        risk = np.clip(deficit, 0.0, 1.0).astype(np.float32, copy=False)
        risk[interpolated] = 1.0

    return BeltMapRiskMaps(
        support=clean_support,
        observed_mask=observed,
        interpolated_mask=interpolated,
        low_support_mask=low_support,
        risk=risk,
    )


def score_map_risk_detections(
    detections: Sequence[ParticleDetection],
    *,
    phase_px: float,
    frame_shape: tuple[int, int],
    maps: BeltMapRiskMaps,
    reject_max_mean_risk: float = 1.0,
    reject_max_interpolated_fraction: float = 1.0,
    reject_max_low_support_fraction: float = 1.0,
) -> list[MapRiskDetectionScore]:
    """Score detections by the support/risk of their belt-coordinate bbox.

    Rejection thresholds are disabled by default because all thresholds default
    to the maximum possible score/fraction.  Set any threshold below 1 to turn the
    diagnostic into a conservative component-level gate.
    """

    _validate_probability_threshold(
        "reject_max_mean_risk",
        reject_max_mean_risk,
    )
    _validate_probability_threshold(
        "reject_max_interpolated_fraction",
        reject_max_interpolated_fraction,
    )
    _validate_probability_threshold(
        "reject_max_low_support_fraction",
        reject_max_low_support_fraction,
    )
    _validate_risk_map_shapes(maps)
    phase_px = _finite_float(phase_px, "phase_px")
    height, width = _validate_frame_shape(frame_shape)
    if maps.support.shape[1] != width:
        raise ValueError(
            "belt-map support width must match detection frame width: "
            f"{maps.support.shape[1]} != {width}"
        )

    support_view = render_belt_view(maps.support, phase_px, height)
    risk_view = render_belt_view(maps.risk, phase_px, height)
    interpolated_view = render_belt_view(
        maps.interpolated_mask.astype(np.float32),
        phase_px,
        height,
    )
    low_support_view = render_belt_view(
        maps.low_support_mask.astype(np.float32),
        phase_px,
        height,
    )

    scores: list[MapRiskDetectionScore] = []
    for detection in detections:
        stats = _bbox_map_risk_stats(
            detection,
            support_view=support_view,
            risk_view=risk_view,
            interpolated_view=interpolated_view,
            low_support_view=low_support_view,
        )
        scored_detection = replace(detection, **stats)
        mean_risk = scored_detection.map_risk_mean
        interpolated_fraction = scored_detection.map_interpolated_fraction
        low_support_fraction = scored_detection.map_low_support_fraction
        rejected = bool(
            mean_risk is not None
            and interpolated_fraction is not None
            and low_support_fraction is not None
            and (
                mean_risk > reject_max_mean_risk
                or interpolated_fraction > reject_max_interpolated_fraction
                or low_support_fraction > reject_max_low_support_fraction
            )
        )
        scores.append(MapRiskDetectionScore(scored_detection, rejected))
    return scores


def _bbox_map_risk_stats(
    detection: ParticleDetection,
    *,
    support_view: np.ndarray,
    risk_view: np.ndarray,
    interpolated_view: np.ndarray,
    low_support_view: np.ndarray,
) -> dict[str, float | None]:
    height, width = support_view.shape
    top = max(0, _integer_coordinate(detection.bbox_top, "bbox_top"))
    left = max(0, _integer_coordinate(detection.bbox_left, "bbox_left"))
    bottom = min(height, _integer_coordinate(detection.bbox_bottom, "bbox_bottom"))
    right = min(width, _integer_coordinate(detection.bbox_right, "bbox_right"))
    if bottom <= top or right <= left:
        return {
            "map_support_min": None,
            "map_support_mean": None,
            "map_risk_mean": None,
            "map_risk_max": None,
            "map_interpolated_fraction": None,
            "map_low_support_fraction": None,
        }

    support_values = support_view[top:bottom, left:right]
    risk_values = risk_view[top:bottom, left:right]
    interpolated_values = interpolated_view[top:bottom, left:right]
    low_support_values = low_support_view[top:bottom, left:right]

    # Non-periodic finite-strip rendering marks rows beyond reconstructed map
    # support with NaN.  Those pixels are not missing observations to ignore:
    # they have zero support and maximal map uncertainty.  Counting them
    # conservatively prevents boundary detections from appearing better supported
    # than they are, and keeps fully out-of-support boxes scoreable and filterable.
    support_values = np.where(np.isfinite(support_values), support_values, 0.0)
    risk_values = np.where(np.isfinite(risk_values), risk_values, 1.0)
    interpolated_values = np.where(
        np.isfinite(interpolated_values),
        interpolated_values,
        1.0,
    )
    low_support_values = np.where(
        np.isfinite(low_support_values),
        low_support_values,
        1.0,
    )

    return {
        "map_support_min": float(np.min(support_values)),
        "map_support_mean": float(np.mean(support_values)),
        "map_risk_mean": float(np.mean(risk_values)),
        "map_risk_max": float(np.max(risk_values)),
        "map_interpolated_fraction": _finite_mean_clipped_fraction(interpolated_values),
        "map_low_support_fraction": _finite_mean_clipped_fraction(low_support_values),
    }


def _finite_mean_clipped_fraction(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.mean(np.clip(finite, 0.0, 1.0)))


def _validate_risk_map_shapes(maps: BeltMapRiskMaps) -> None:
    support = _support_array(maps.support, "support")
    shape = support.shape
    risk = _probability_array(maps.risk, "risk")
    if risk.shape != shape:
        raise ValueError(f"risk shape must match support shape: {risk.shape} != {shape}")
    for name in ("observed_mask", "interpolated_mask", "low_support_mask"):
        arr = getattr(maps, name)
        mask = np.asarray(arr)
        if mask.shape != shape:
            raise ValueError(f"{name} shape must match support shape: {arr.shape} != {shape}")
        if mask.dtype != np.bool_:
            raise ValueError(f"{name} must be a boolean array")


def _validate_frame_shape(frame_shape: tuple[int, int]) -> tuple[int, int]:
    if len(frame_shape) != 2:
        raise ValueError("frame_shape must contain height and width")
    height = _positive_integer_dimension(frame_shape[0], "frame_shape height")
    width = _positive_integer_dimension(frame_shape[1], "frame_shape width")
    return height, width


def _positive_integer_dimension(value: int, name: str) -> int:
    parsed = _finite_float(value, name)
    if not parsed.is_integer() or parsed < 1:
        raise ValueError(f"{name} must be a positive finite integer")
    return int(parsed)


def _integer_coordinate(value: int, name: str) -> int:
    parsed = _finite_float(value, name)
    if not parsed.is_integer():
        raise ValueError(f"{name} must be a finite integer")
    return int(parsed)


def _validate_probability_threshold(name: str, value: float) -> None:
    parsed = _finite_float(value, name)
    if parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_, str)):
        raise ValueError(f"{name} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _support_array(value: object, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype == np.bool_ or not np.issubdtype(raw.dtype, np.number):
        raise ValueError(f"{name} must contain finite non-negative support values")
    support = raw.astype(np.float32, copy=False)
    if support.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array")
    if not np.all(np.isfinite(support)) or np.any(support < 0.0):
        raise ValueError(f"{name} must contain finite non-negative support values")
    return support


def _probability_array(value: object, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array")
    if not np.all(np.isfinite(arr)) or np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError(f"{name} must contain finite values in [0, 1]")
    return arr
