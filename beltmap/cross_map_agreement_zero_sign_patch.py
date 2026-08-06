"""Treat residual regions without a net polarity as signless.

Cross-map agreement can require the raw residual polarity to agree between the
primary and confirming maps.  The original helper returned integer ``0`` when a
detection's fallback bounding-box mean was exactly zero.  Two such signless
regions therefore compared equal and satisfied the sign-consistency gate even
though neither supplied positive or negative polarity evidence.
"""

from __future__ import annotations

from typing import Any

from . import cross_map_agreement as _agreement

_PATCHED_ATTR = "_beltmap_cross_map_zero_sign_patched"
_ORIGINAL_ATTR = "_beltmap_original_detection_raw_sign"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the raw-sign helper behind this compatibility patch."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_detection_raw_sign = _unwrap_patched_callable(
    _agreement.detection_raw_sign
)


def nonzero_detection_raw_sign(
    detection: _agreement.ParticleDetection,
    residual: _agreement.ResidualImage | None,
) -> int | None:
    """Return positive/negative polarity, or ``None`` when no sign is defined."""

    sign = _original_detection_raw_sign(detection, residual)
    return None if sign == 0 else sign


setattr(nonzero_detection_raw_sign, _PATCHED_ATTR, True)
setattr(
    nonzero_detection_raw_sign,
    _ORIGINAL_ATTR,
    _original_detection_raw_sign,
)
_agreement.detection_raw_sign = nonzero_detection_raw_sign
