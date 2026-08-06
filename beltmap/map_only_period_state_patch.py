"""Prevent cyclic map-only diagnostics on inferred finite belt-map strips."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from beltmap import map_only_negative_control as _map_only

_ORIGINAL_ATTR = "_beltmap_map_only_period_state_original_generate"
_PATCHED_ATTR = "_beltmap_map_only_period_state_guarded"


def _unwrap_patched_callable(func: Any) -> Any:
    return getattr(func, _ORIGINAL_ATTR, func)


_original_generate_map_only_negative_control_report = _unwrap_patched_callable(
    _map_only.generate_map_only_negative_control_report
)


def _metadata_path(*, output_dir: Path, belt_map_path: Path | None) -> Path:
    belt_path = (
        Path(belt_map_path)
        if belt_map_path is not None
        else Path(output_dir) / "belt_map.npy"
    )
    sibling = belt_path.with_name("metadata.json")
    if sibling.is_file():
        return sibling
    return Path(output_dir) / "metadata.json"


def _load_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _declares_finite_strip(metadata: dict[str, Any]) -> bool:
    if metadata.get("belt_map_periodic") is False:
        return True
    if metadata.get("belt_period_known") is False:
        return True
    return "model_period_px" in metadata and metadata.get("model_period_px") in (
        None,
        "",
    )


def period_safe_generate_map_only_negative_control_report(
    *,
    output_dir: Path,
    config=None,
    belt_map_path: Path | None = None,
    phase_estimates_path: Path | None = None,
    metrics_path: Path | None = None,
    report_path: Path | None = None,
    detections_path: Path | None = None,
    detections_per_frame_path: Path | None = None,
    tracks_path: Path | None = None,
    velocities_path: Path | None = None,
    track_scores_path: Path | None = None,
) -> Any:
    metadata_path = _metadata_path(
        output_dir=Path(output_dir),
        belt_map_path=None if belt_map_path is None else Path(belt_map_path),
    )
    metadata = _load_metadata(metadata_path)
    if _declares_finite_strip(metadata):
        raise ValueError(
            "map-only negative-control rendering requires a known physical "
            "BELT_PERIOD_PX; metadata marks the belt map as an inferred finite "
            "strip, so cyclic rendering would wrap unsupported rows to the "
            "opposite map boundary"
        )

    return _original_generate_map_only_negative_control_report(
        output_dir=output_dir,
        config=config,
        belt_map_path=belt_map_path,
        phase_estimates_path=phase_estimates_path,
        metrics_path=metrics_path,
        report_path=report_path,
        detections_path=detections_path,
        detections_per_frame_path=detections_per_frame_path,
        tracks_path=tracks_path,
        velocities_path=velocities_path,
        track_scores_path=track_scores_path,
    )


setattr(period_safe_generate_map_only_negative_control_report, _PATCHED_ATTR, True)
setattr(
    period_safe_generate_map_only_negative_control_report,
    _ORIGINAL_ATTR,
    _original_generate_map_only_negative_control_report,
)
_map_only.generate_map_only_negative_control_report = (
    period_safe_generate_map_only_negative_control_report
)
