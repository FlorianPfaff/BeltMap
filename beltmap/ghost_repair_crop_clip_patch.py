from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from beltmap import ghost_repair as _ghost_repair

_ORIGINAL_ATTR = "_beltmap_ghost_repair_original_build_defect_maps"
_PATCHED_ATTR = "_beltmap_ghost_repair_crop_clipped"


def _unwrap_patched_callable(func: Any) -> Any:
    return getattr(func, _ORIGINAL_ATTR, func)


_original_build_ghost_defect_maps = _unwrap_patched_callable(
    _ghost_repair.build_ghost_defect_maps
)


def _positive_int(value: Any) -> int | None:
    parsed = _ghost_repair.finite_int(value)
    if parsed is None or parsed <= 0:
        return None
    return parsed


def _visible_crop_height_from_metrics(metrics: Mapping[str, Any], *, fallback: int) -> int:
    """Return the visible crop height used by map-only detections.

    GhostRepair projects map-only detection boxes back into belt-map coordinates.
    The detection boxes are crop-local, so a margin around a bottom-edge box must
    be clipped to the visible crop height before projection.  Falling back to the
    belt-map height preserves the historical behavior for legacy metrics that do
    not record a crop height.
    """

    containers: list[Mapping[str, Any]] = [metrics]
    for key in ("detection_config", "map_only_config", "config"):
        value = metrics.get(key)
        if isinstance(value, Mapping):
            containers.append(value)

    for container in containers:
        for key in ("crop_height_px", "crop_height", "image_height", "height"):
            parsed = _positive_int(container.get(key))
            if parsed is not None:
                return parsed
    return int(fallback)


def _mark_bbox_in_visible_crop_coordinates(
    counts: np.ndarray,
    *,
    phase_px: float,
    bbox_top: float,
    bbox_left: float,
    bbox_bottom: float,
    bbox_right: float,
    margin_px: int,
    crop_height_px: int,
) -> None:
    map_height, width = counts.shape
    crop_height = max(0, int(crop_height_px))
    y0 = max(0, int(math.floor(bbox_top)) - margin_px)
    y1 = min(crop_height, int(math.ceil(bbox_bottom)) + margin_px)
    x0 = max(0, int(math.floor(bbox_left)) - margin_px)
    x1 = min(width, int(math.ceil(bbox_right)) + margin_px)
    if y1 <= y0 or x1 <= x0:
        return
    for crop_y in range(y0, y1):
        map_y_float = (phase_px + crop_y) % map_height
        row0 = int(math.floor(map_y_float)) % map_height
        row1 = (row0 + 1) % map_height
        counts[row0, x0:x1] += 1
        counts[row1, x0:x1] += 1


def crop_clipped_build_ghost_defect_maps(
    *,
    belt_map_shape: tuple[int, int],
    tracks_by_id: Mapping[int, list[dict[str, str]]],
    selected_track_ids: set[int],
    phase_by_frame: Mapping[float, float],
    metrics: Mapping[str, Any],
    margin_px: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Build GhostRepair masks without projecting outside visible crop rows."""

    counts = np.zeros(belt_map_shape, dtype=np.uint16)
    track_rows: list[dict[str, Any]] = []
    crop_height_px = _visible_crop_height_from_metrics(metrics, fallback=belt_map_shape[0])
    for track_id in sorted(selected_track_ids):
        rows = tracks_by_id.get(track_id, [])
        if not rows:
            continue
        centers_y: list[float] = []
        centers_x: list[float] = []
        peak_signals: list[float] = []
        for row in rows:
            frame_index = _ghost_repair.finite_float(row.get("frame_index"))
            bbox_top = _ghost_repair.finite_float(row.get("bbox_top"))
            bbox_left = _ghost_repair.finite_float(row.get("bbox_left"))
            bbox_bottom = _ghost_repair.finite_float(row.get("bbox_bottom"))
            bbox_right = _ghost_repair.finite_float(row.get("bbox_right"))
            if None in {frame_index, bbox_top, bbox_left, bbox_bottom, bbox_right}:
                continue
            assert frame_index is not None
            assert bbox_top is not None and bbox_left is not None
            assert bbox_bottom is not None and bbox_right is not None
            phase_px = _ghost_repair.phase_for_frame(
                frame_index,
                phase_by_frame=phase_by_frame,
                metrics=metrics,
                map_height=belt_map_shape[0],
            )
            _mark_bbox_in_visible_crop_coordinates(
                counts,
                phase_px=phase_px,
                bbox_top=bbox_top,
                bbox_left=bbox_left,
                bbox_bottom=bbox_bottom,
                bbox_right=bbox_right,
                margin_px=margin_px,
                crop_height_px=crop_height_px,
            )
            cy = (phase_px + 0.5 * (bbox_top + bbox_bottom)) % belt_map_shape[0]
            cx = 0.5 * (bbox_left + bbox_right)
            centers_y.append(cy)
            centers_x.append(cx)
            peak = _ghost_repair.finite_float(row.get("peak_signal"))
            if peak is not None:
                peak_signals.append(peak)
        if centers_y:
            reference = centers_y[0]
            unwrapped_y = np.asarray(
                [reference + _ghost_repair.cyclic_delta(value, reference, belt_map_shape[0]) for value in centers_y],
                dtype=np.float64,
            )
            centers_x_arr = np.asarray(centers_x, dtype=np.float64)
            compactness = float(np.sqrt(np.mean((unwrapped_y - np.mean(unwrapped_y)) ** 2)))
            track_rows.append(
                {
                    "track_id": track_id,
                    "n_detections": len(rows),
                    "map_y_min": float(np.min(unwrapped_y)),
                    "map_y_max": float(np.max(unwrapped_y)),
                    "map_x_min": float(np.min(centers_x_arr)),
                    "map_x_max": float(np.max(centers_x_arr)),
                    "max_signal": "" if not peak_signals else float(np.max(peak_signals)),
                    "belt_y_rms_px": compactness,
                    "belt_x_std_px": float(np.std(centers_x_arr)),
                    "causal_ghost_score": "",
                }
            )
    mask = counts > 0
    probability = counts.astype(np.float32)
    max_count = int(np.max(counts)) if counts.size else 0
    if max_count > 0:
        probability /= float(max_count)
    return mask, counts, probability, track_rows


setattr(crop_clipped_build_ghost_defect_maps, _PATCHED_ATTR, True)
setattr(crop_clipped_build_ghost_defect_maps, _ORIGINAL_ATTR, _original_build_ghost_defect_maps)
_ghost_repair.build_ghost_defect_maps = crop_clipped_build_ghost_defect_maps
