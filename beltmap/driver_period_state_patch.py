"""Driver-side preservation of known-vs-inferred belt-period state.

This module is imported for its side effects from :mod:`beltmap.__init__`.
It keeps the legacy driver from treating the finite support height of an
inferred belt map as a trusted cyclic belt circumference.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import _driver_map
from . import driver as _driver
from .period_state import (
    BeltPeriodState,
    fresh_period_state,
    metadata_fields,
    phase_fraction_and_radians,
    require_period_known,
    reused_period_state,
)
from .phase import BeltMotionModel as _BeltMotionModel
from .phase import PhaseDriftFilter as _PhaseDriftFilter
from .phase import render_belt_view as _render_belt_view

_MAP_BUILD_PERIOD_KNOWN: list[bool | None] = [None]
_MAP_ACCUMULATION_PERIODIC: list[bool | None] = [None]
_DRIVER_MODEL_PERIOD_UNKNOWN = object()
_DRIVER_MODEL_PERIOD_PX = [_DRIVER_MODEL_PERIOD_UNKNOWN]
_ORIGINALS_ATTR = "_beltmap_driver_period_state_originals"

_originals = getattr(_driver, _ORIGINALS_ATTR, None)
if _originals is None:
    _originals = {
        "build_belt_map_result": _driver.build_belt_map_result,
        "accumulate_belt_map": _driver_map.accumulate_belt_map,
        "belt_motion_model": _BeltMotionModel,
        "phase_drift_filter": _PhaseDriftFilter,
        "phase_estimate_row": _driver.phase_estimate_row,
        "texture_phase_velocity_summary": _driver.texture_phase_velocity_summary,
        "score_recurrent_artifact_detections": _driver.score_recurrent_artifact_detections,
        "driver_main": _driver.main,
    }
    setattr(_driver, _ORIGINALS_ATTR, _originals)

_original_build_belt_map_result = _originals["build_belt_map_result"]
_original_accumulate_belt_map = _originals["accumulate_belt_map"]
_original_belt_motion_model = _originals["belt_motion_model"]
_original_phase_drift_filter = _originals["phase_drift_filter"]
_original_phase_estimate_row = _originals["phase_estimate_row"]
_original_texture_phase_velocity_summary = _originals["texture_phase_velocity_summary"]
_original_score_recurrent_artifact_detections = _originals[
    "score_recurrent_artifact_detections"
]
_original_driver_main = _originals["driver_main"]


def _env_int(name: str) -> int | None:
    value = os.getenv(name)
    if value in (None, ""):
        return None
    return int(value)


def _load_metadata_for_reused_map(path: Path) -> dict[str, Any]:
    metadata_path = path.with_name("metadata.json")
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _height_from_period(period_px: float | int | None) -> int | None:
    if period_px is None:
        return None
    try:
        period = float(period_px)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(period) or period <= 0.0:
        return None
    rounded = int(round(period))
    if abs(period - float(rounded)) > 1e-6:
        return None
    return rounded


def _state_for_map_height(map_height_px: int) -> BeltPeriodState:
    supplied_period = _env_int("BELT_PERIOD_PX")
    reuse_path_value = os.getenv("REUSE_BELT_MAP_PATH", "").strip()
    if reuse_path_value:
        reuse_path = Path(reuse_path_value)
        return reused_period_state(
            map_height_px=map_height_px,
            supplied_period_px=supplied_period,
            metadata=_load_metadata_for_reused_map(reuse_path),
        )
    model_period = (
        float(supplied_period)
        if supplied_period is not None and supplied_period == map_height_px
        else None
    )
    return fresh_period_state(
        map_height_px=map_height_px,
        model_period_px=model_period,
    )


def _state_for_period(period_px: float | int | None) -> BeltPeriodState | None:
    map_height = _height_from_period(period_px)
    if map_height is None:
        return None
    return _state_for_map_height(map_height)


def _model_period(period_px: float | int | None) -> float | None:
    state = _state_for_period(period_px)
    if state is None:
        return None if period_px is None else float(period_px)
    return state.model_period_px


def _output_model_period(period_px: float | int | None) -> float | None:
    driver_period = _DRIVER_MODEL_PERIOD_PX[0]
    if driver_period is not _DRIVER_MODEL_PERIOD_UNKNOWN:
        return None if driver_period is None else float(driver_period)
    return None if period_px is None else float(period_px)


def _patched_build_belt_map_result(*args, **kwargs):
    previous = _MAP_BUILD_PERIOD_KNOWN[0]
    _MAP_BUILD_PERIOD_KNOWN[0] = kwargs.get("supplied_period") is not None
    try:
        return _original_build_belt_map_result(*args, **kwargs)
    finally:
        _MAP_BUILD_PERIOD_KNOWN[0] = previous


def _patched_accumulate_belt_map(*args, **kwargs):
    previous = _MAP_ACCUMULATION_PERIODIC[0]
    _MAP_ACCUMULATION_PERIODIC[0] = bool(kwargs.get("model_period"))
    try:
        return _original_accumulate_belt_map(*args, **kwargs)
    finally:
        _MAP_ACCUMULATION_PERIODIC[0] = previous


def _patched_driver_map_render_belt_view(
    belt_map,
    phase_px,
    height,
    *,
    x_slice=None,
    periodic: bool = True,
):
    accumulation_periodic = _MAP_ACCUMULATION_PERIODIC[0]
    if periodic and accumulation_periodic is False:
        periodic = False
    build_period_known = _MAP_BUILD_PERIOD_KNOWN[0]
    if periodic and build_period_known is False:
        periodic = False
    return _render_belt_view(
        belt_map,
        phase_px,
        height,
        x_slice=x_slice,
        periodic=periodic,
    )


def _patched_belt_motion_model(*args, **kwargs):
    model = _original_belt_motion_model(*args, **kwargs)
    resolved_period = _model_period(model.period_px)
    if resolved_period == model.period_px:
        return model
    return replace(model, period_px=resolved_period)


def _patched_phase_drift_filter(*args, **kwargs):
    if "period_px" in kwargs:
        kwargs = dict(kwargs)
        kwargs["period_px"] = _model_period(kwargs["period_px"])
    return _original_phase_drift_filter(*args, **kwargs)


def _patched_phase_estimate_row(frame_index: int, path, residual, period_px: float | None) -> dict:
    resolved_period = _output_model_period(period_px)
    row_period = resolved_period if resolved_period is not None else 1.0
    row = _original_phase_estimate_row(
        frame_index,
        path,
        residual,
        row_period,
    )
    phase_fraction, phase_rad = phase_fraction_and_radians(
        float(row["phase_px"]),
        resolved_period,
    )
    row["phase_fraction"] = phase_fraction
    row["phase_rad"] = phase_rad
    return row


def _registered_phase_row_count(phase_rows: Sequence[Mapping[str, Any]]) -> tuple[int, bool]:
    frames = 0
    has_registration = False
    for row in phase_rows:
        try:
            float(row["frame_index"])
            float(row["phase_px"])
        except (TypeError, ValueError, KeyError):
            continue
        frames += 1
        has_registration = has_registration or "registration" in str(row.get("method", ""))
    return frames, has_registration


def _patched_texture_phase_velocity_summary(
    phase_rows,
    *,
    period_px: float | None,
    nominal_velocity_px_per_frame: float,
):
    resolved_period = _output_model_period(period_px)
    if resolved_period is None:
        samples, has_registration = _registered_phase_row_count(phase_rows)
        if samples >= 2 and has_registration:
            return {
                "texture_phase_velocity_status": "unknown_period",
                "texture_phase_velocity_samples": samples,
            }
        resolved_period = 1.0
    return _original_texture_phase_velocity_summary(
        phase_rows,
        period_px=resolved_period,
        nominal_velocity_px_per_frame=nominal_velocity_px_per_frame,
    )


def _artifact_map_height(recurrent_artifact_map) -> int | None:
    if hasattr(recurrent_artifact_map, "mask"):
        recurrent_artifact_map = recurrent_artifact_map.mask
    try:
        return int(np.asarray(recurrent_artifact_map).shape[0])
    except (TypeError, ValueError, IndexError):
        return None


def _patched_score_recurrent_artifact_detections(
    detections_by_frame,
    phase_px_by_frame,
    recurrent_artifact_map,
    *args,
    **kwargs,
):
    map_height = _artifact_map_height(recurrent_artifact_map)
    if map_height is not None:
        require_period_known(
            _state_for_map_height(map_height),
            feature="recurrent-artifact filtering",
        )
    return _original_score_recurrent_artifact_detections(
        detections_by_frame,
        phase_px_by_frame,
        recurrent_artifact_map,
        *args,
        **kwargs,
    )


def _recurrent_artifact_requested() -> bool:
    if os.getenv("REUSE_RECURRENT_ARTIFACT_MAP_PATH", "").strip():
        return True
    min_revolutions = _env_int("RECURRENT_ARTIFACT_MIN_REVOLUTIONS")
    return min_revolutions is not None and min_revolutions > 0


def _period_state_for_driver_preflight() -> BeltPeriodState | None:
    reuse_path_value = os.getenv("REUSE_BELT_MAP_PATH", "").strip()
    if reuse_path_value:
        map_shape = np.load(Path(reuse_path_value), mmap_mode="r").shape
        return _state_for_map_height(int(map_shape[0]))
    supplied_period = _env_int("BELT_PERIOD_PX")
    if supplied_period is None:
        return None
    return fresh_period_state(
        map_height_px=supplied_period,
        model_period_px=float(supplied_period),
    )


def _metadata_output_path() -> Path:
    output_dir = os.getenv("BELTMAP_OUTPUT_DIR", "outputs")
    return Path(output_dir) / "metadata.json"


def _patch_metadata_file() -> None:
    metadata_path = _metadata_output_path()
    if not metadata_path.exists():
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    map_height = metadata.get("belt_map_height_px")
    if map_height in (None, ""):
        return
    state = _state_for_map_height(int(map_height))
    metadata.update(metadata_fields(state))
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _patched_main(*args, **kwargs):
    state = _period_state_for_driver_preflight()
    if _recurrent_artifact_requested():
        if state is None:
            raise ValueError(
                "recurrent-artifact filtering requires a known physical BELT_PERIOD_PX; "
                "the current belt map is an inferred finite strip"
            )
        require_period_known(state, feature="recurrent-artifact filtering")

    previous_output_period = _DRIVER_MODEL_PERIOD_PX[0]
    _DRIVER_MODEL_PERIOD_PX[0] = None if state is None else state.model_period_px
    try:
        result = _original_driver_main(*args, **kwargs)
    finally:
        _DRIVER_MODEL_PERIOD_PX[0] = previous_output_period
    _patch_metadata_file()
    return result


_driver.build_belt_map_result = _patched_build_belt_map_result
_driver.BeltMotionModel = _patched_belt_motion_model
_driver.PhaseDriftFilter = _patched_phase_drift_filter
_driver.phase_estimate_row = _patched_phase_estimate_row
_driver.texture_phase_velocity_summary = _patched_texture_phase_velocity_summary
_driver.score_recurrent_artifact_detections = _patched_score_recurrent_artifact_detections
_driver.main = _patched_main
_driver_map.accumulate_belt_map = _patched_accumulate_belt_map
_driver_map.render_belt_view = _patched_driver_map_render_belt_view
