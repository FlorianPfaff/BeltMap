"""Prevent sparse finite overlap from winning cyclic map alignment.

The benchmark compares a reconstructed belt map against synthetic ground truth
under every cyclic row shift.  The historical implementation minimized RMSE
alone even though each shift can contain a different number of paired finite
pixels.  A shift with one coincident pixel could therefore beat the physically
correct shift supported by many pixels.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from . import benchmark as _benchmark

_PATCHED_ATTR = "_beltmap_map_overlap_patched"
_ORIGINAL_ATTR = "_beltmap_original_map_metrics"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the original map metric implementation behind this patch."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_map_metrics = _unwrap_patched_callable(_benchmark.map_metrics)


def coverage_first_map_metrics(
    output_dir: Path,
    truth_path: Path,
    truth: dict[str, Any],
) -> dict[str, Any]:
    """Compare belt maps while preferring shifts with maximum finite support.

    Candidate shifts are ranked lexicographically by paired finite-pixel count
    and then by RMSE.  This keeps the error comparison on the largest available
    common support instead of allowing a tiny overlap to win by chance.
    """

    reconstructed_path = output_dir / "belt_map.npy"
    true_map_path = _benchmark.resolve_truth_path(
        truth_path,
        truth.get("true_belt_map_npy"),
        "true_belt_map.npy",
    )
    if not reconstructed_path.is_file():
        return {"available": False, "reason": f"Missing {reconstructed_path}"}
    if not true_map_path.is_file():
        return {"available": False, "reason": f"Missing {true_map_path}"}

    reconstructed = np.asarray(np.load(reconstructed_path), dtype=np.float64)
    target = np.asarray(np.load(true_map_path), dtype=np.float64)
    if reconstructed.shape != target.shape:
        return {
            "available": False,
            "reason": "Shape mismatch",
            "reconstructed_shape": list(reconstructed.shape),
            "truth_shape": list(target.shape),
        }
    if reconstructed.ndim != 2:
        return {
            "available": False,
            "reason": "Expected 2-D belt maps",
            "reconstructed_shape": list(reconstructed.shape),
        }
    if reconstructed.shape[0] == 0 or reconstructed.shape[1] == 0:
        return {
            "available": False,
            "reason": "Empty belt maps",
            "reconstructed_shape": list(reconstructed.shape),
        }

    best_shift = 0
    best_rmse = float("inf")
    best_mae = float("inf")
    best_finite_pixels = 0
    for shift in range(target.shape[0]):
        shifted = np.roll(reconstructed, shift=shift, axis=0)
        valid = np.isfinite(shifted) & np.isfinite(target)
        finite_pixels = int(np.count_nonzero(valid))
        if finite_pixels == 0:
            continue
        error = shifted[valid] - target[valid]
        rmse = float(np.sqrt(np.mean(np.square(error))))
        if finite_pixels > best_finite_pixels or (
            finite_pixels == best_finite_pixels and rmse < best_rmse
        ):
            best_shift = shift
            best_rmse = rmse
            best_mae = float(np.mean(np.abs(error)))
            best_finite_pixels = finite_pixels

    if best_finite_pixels == 0 or not math.isfinite(best_rmse):
        return {
            "available": False,
            "reason": "No finite paired belt-map pixels",
            "reconstructed_map": str(reconstructed_path),
            "truth_map": str(true_map_path),
            "shape": list(reconstructed.shape),
        }

    return {
        "available": True,
        "truth_map": str(true_map_path),
        "reconstructed_map": str(reconstructed_path),
        "shape": list(reconstructed.shape),
        "finite_pixels": best_finite_pixels,
        "best_cyclic_shift_px": int(best_shift),
        "rmse_gray": best_rmse,
        "mean_abs_error_gray": best_mae,
    }


setattr(coverage_first_map_metrics, _PATCHED_ATTR, True)
setattr(coverage_first_map_metrics, _ORIGINAL_ATTR, _original_map_metrics)
_benchmark.map_metrics = coverage_first_map_metrics
