"""Ignore unavailable detection counts in advanced spike diagnostics.

The legacy quality report substituted zero for blank or malformed values in
``detections_per_frame.csv``.  Missing measurements then changed both the median
and upper percentile used by the spike heuristic, which could fabricate a
``detection_spikes`` warning.  This compatibility patch keeps real zero-count
frames but excludes unavailable counts from that diagnostic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import advanced_quality as _advanced_quality

_PATCHED_ATTR = "_beltmap_missing_detection_counts_patched"
_ORIGINAL_ATTR = "_beltmap_original_quality_flags"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the report function behind this compatibility patch, if present."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_quality_flags = _unwrap_patched_callable(_advanced_quality.quality_flags)


def _detection_spike_flag(output_dir: Path) -> dict[str, Any] | None:
    rows = _advanced_quality.read_csv_rows(output_dir / "detections_per_frame.csv")
    counts = np.asarray(
        [
            count
            for row in rows
            if (
                count := _advanced_quality.finite_float(row.get("n_detections"))
            )
            is not None
        ],
        dtype=np.float64,
    )
    if counts.size == 0:
        return None

    percentile_95 = float(np.percentile(counts, 95))
    median = float(np.median(counts))
    if percentile_95 <= max(25.0, 5.0 * (median + 1.0)):
        return None
    return {
        "severity": "warning",
        "code": "detection_spikes",
        "message": "detection counts have large frame-to-frame spikes",
        "p95": percentile_95,
        "median": median,
    }


def quality_flags_ignoring_missing_detection_counts(
    output_dir: Path,
) -> dict[str, Any]:
    """Return quality flags without treating unavailable counts as zero frames."""

    output_dir = Path(output_dir)
    result = dict(_original_quality_flags(output_dir))
    replacement = _detection_spike_flag(output_dir)
    raw_flags = result.get("flags", [])
    flags: list[Any] = []
    inserted = False
    if isinstance(raw_flags, list):
        for flag in raw_flags:
            is_spike_flag = (
                isinstance(flag, Mapping) and flag.get("code") == "detection_spikes"
            )
            if not is_spike_flag:
                flags.append(flag)
                continue
            if replacement is not None and not inserted:
                flags.append(replacement)
                inserted = True
    if replacement is not None and not inserted:
        flags.append(replacement)
    result["flags"] = flags
    return result


setattr(quality_flags_ignoring_missing_detection_counts, _PATCHED_ATTR, True)
setattr(
    quality_flags_ignoring_missing_detection_counts,
    _ORIGINAL_ATTR,
    _original_quality_flags,
)
_advanced_quality.quality_flags = quality_flags_ignoring_missing_detection_counts
