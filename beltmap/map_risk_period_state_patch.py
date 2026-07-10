"""Keep map-risk rendering non-periodic for inferred finite-strip driver runs.

This module is imported for its side effects from :mod:`beltmap.__init__` after
:mod:`beltmap.driver_period_state_patch` has installed the driver-run period
context.  Map-risk scoring renders belt-coordinate support images separately
from the main map-building path, so it must consume the same context explicitly.
"""

from __future__ import annotations

from typing import Any

from . import driver_period_state_patch as _driver_period_state
from . import map_risk as _map_risk

_PATCHED_ATTR = "_beltmap_map_risk_period_state_patched"
_ORIGINAL_ATTR = "_beltmap_map_risk_original_render_belt_view"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the original renderer behind this patch, if already installed."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_render_belt_view = _unwrap_patched_callable(_map_risk.render_belt_view)


def period_aware_map_risk_render_belt_view(
    belt_map,
    phase_px,
    height,
    *,
    x_slice=None,
    periodic: bool = True,
):
    """Render map-risk views without wrapping an inferred finite strip.

    Outside a packaged driver run, preserve the caller's periodic argument.  A
    driver run records ``None`` when the map height is only finite support rather
    than a known physical belt period; in that context, disable wrapping for the
    support, risk, interpolated-mask, and low-support-mask views scored by
    :func:`beltmap.map_risk.score_map_risk_detections`.
    """

    driver_period = _driver_period_state._DRIVER_MODEL_PERIOD_PX[0]
    if (
        periodic
        and driver_period is not _driver_period_state._DRIVER_MODEL_PERIOD_UNKNOWN
        and driver_period is None
    ):
        periodic = False
    return _original_render_belt_view(
        belt_map,
        phase_px,
        height,
        x_slice=x_slice,
        periodic=periodic,
    )


setattr(period_aware_map_risk_render_belt_view, _PATCHED_ATTR, True)
setattr(
    period_aware_map_risk_render_belt_view,
    _ORIGINAL_ATTR,
    _original_render_belt_view,
)
_map_risk.render_belt_view = period_aware_map_risk_render_belt_view
