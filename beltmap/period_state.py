"""Helpers for preserving known-vs-inferred belt period state."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi
from typing import Any, Mapping


@dataclass(frozen=True)
class BeltPeriodState:
    """Driver-facing period state for a reconstructed belt map.

    ``map_height_px`` is the support height of the reconstructed belt-map array.
    ``model_period_px`` is the physical cyclic belt period to use in the motion
    model. It is ``None`` when the map is a finite inferred strip rather than a
    trusted full belt circumference.
    """

    map_height_px: int
    model_period_px: float | None
    period_known: bool
    source: str

    @property
    def periodic(self) -> bool:
        """Whether downstream rendering and revolution logic may wrap rows."""

        return self.model_period_px is not None


def fresh_period_state(*, map_height_px: int, model_period_px: float | None) -> BeltPeriodState:
    """Return period state for a freshly reconstructed map."""

    height = _positive_int(map_height_px, name="map_height_px")
    period = _positive_float_or_none(model_period_px, name="model_period_px")
    return BeltPeriodState(
        map_height_px=height,
        model_period_px=period,
        period_known=period is not None,
        source="supplied" if period is not None else "inferred_finite_strip",
    )


def reused_period_state(
    *,
    map_height_px: int,
    supplied_period_px: int | None,
    metadata: Mapping[str, Any] | None,
) -> BeltPeriodState:
    """Resolve period state for a reused ``belt_map.npy`` and metadata pair.

    New metadata keys are trusted first. For older runs without explicit period
    metadata, a supplied ``BELT_PERIOD_PX`` is trusted only when it matches the
    loaded map height. Otherwise the reused map is treated as a finite strip to
    avoid silently wrapping inferred support.
    """

    height = _positive_int(map_height_px, name="map_height_px")
    supplied = _positive_float_or_none(supplied_period_px, name="supplied_period_px")
    meta = dict(metadata or {})
    _validate_metadata_map_height(meta, height)

    metadata_period = _metadata_period(meta)
    if metadata_period is not None:
        _validate_reused_period_matches_height(
            metadata_period,
            height,
            source="metadata model_period_px",
        )
        return BeltPeriodState(
            map_height_px=height,
            model_period_px=metadata_period,
            period_known=True,
            source="metadata",
        )

    if _metadata_requires_missing_period(meta):
        raise ValueError(
            "metadata declares a periodic belt map but does not provide a "
            "model_period_px or belt_period_px_input"
        )

    if _metadata_declares_finite(meta):
        return BeltPeriodState(
            map_height_px=height,
            model_period_px=None,
            period_known=False,
            source="metadata_finite_strip",
        )

    if supplied is not None and int(round(supplied)) == height:
        return BeltPeriodState(
            map_height_px=height,
            model_period_px=float(supplied),
            period_known=True,
            source="supplied_matching_reuse_height",
        )

    return BeltPeriodState(
        map_height_px=height,
        model_period_px=None,
        period_known=False,
        source="legacy_reuse_unknown",
    )


def phase_fraction_and_radians(
    phase_px: float,
    model_period_px: float | None,
) -> tuple[float | str, float | str]:
    """Return phase fraction/radians fields for CSV output.

    Finite strips do not have a physical cycle, so the derived cyclic fields are
    represented as empty strings rather than normalized by the finite support
    height.
    """

    period = _positive_float_or_none(model_period_px, name="model_period_px")
    if period is None:
        return "", ""
    fraction = float(phase_px) / period
    return fraction, fraction * 2.0 * pi


def require_period_known(state: BeltPeriodState, *, feature: str) -> None:
    """Raise a clear error when a cyclic-only feature is used on a finite strip."""

    if state.model_period_px is None:
        raise ValueError(
            f"{feature} requires a known physical BELT_PERIOD_PX; "
            "the current belt map is an inferred finite strip"
        )


def metadata_fields(state: BeltPeriodState) -> dict[str, float | bool | str | int | None]:
    """Return stable metadata fields preserving finite-vs-periodic semantics."""

    return {
        "belt_map_height_px": state.map_height_px,
        "model_period_px": state.model_period_px,
        "belt_period_known": state.period_known,
        "belt_map_periodic": state.periodic,
        "belt_period_state_source": state.source,
    }


def _metadata_period(metadata: Mapping[str, Any]) -> float | None:
    for key in ("model_period_px", "belt_model_period_px"):
        if key in metadata and metadata[key] not in (None, ""):
            return _positive_float_or_none(metadata[key], name=key)
    if metadata.get("belt_period_known") is True:
        value = metadata.get("belt_period_px_input")
        if value not in (None, ""):
            return _positive_float_or_none(value, name="belt_period_px_input")
    return None


def _metadata_requires_missing_period(metadata: Mapping[str, Any]) -> bool:
    return metadata.get("belt_period_known") is True or metadata.get("belt_map_periodic") is True


def _metadata_declares_finite(metadata: Mapping[str, Any]) -> bool:
    if metadata.get("belt_period_known") is False:
        return True
    if metadata.get("belt_map_periodic") is False:
        return True
    value = metadata.get("model_period_px")
    return value in (None, "") and "model_period_px" in metadata


def _validate_metadata_map_height(metadata: Mapping[str, Any], height: int) -> None:
    value = metadata.get("belt_map_height_px")
    if value in (None, ""):
        return
    recorded_height = _positive_int(value, name="metadata belt_map_height_px")
    if recorded_height != height:
        raise ValueError(
            "metadata belt_map_height_px must match reused belt-map height; "
            f"got metadata height {recorded_height} for loaded height {height}"
        )


def _validate_reused_period_matches_height(period: float, height: int, *, source: str) -> None:
    if abs(float(period) - float(height)) > 1e-6:
        raise ValueError(
            f"{source} must match reused belt-map height; "
            f"got period {period:g} for height {height}"
        )


def _positive_int(value: Any, *, name: str) -> int:
    parsed_float = float(value)
    if not isfinite(parsed_float) or not parsed_float.is_integer():
        raise ValueError(f"{name} must be a positive integer")
    parsed = int(parsed_float)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _positive_float_or_none(value: Any, *, name: str) -> float | None:
    if value in (None, ""):
        return None
    parsed = float(value)
    if not isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be a positive finite value when set")
    return parsed
