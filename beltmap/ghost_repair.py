from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw

from .map_only_negative_control import (
    MapOnlyNegativeControlConfig,
    generate_map_only_negative_control_report,
)


REBUILD_MASKED_REUSE_ENV_VARS = {
    "REUSE_BELT_MAP_PATH",
    "REUSE_PHASE_ESTIMATES_PATH",
    "REUSE_STATIC_NOISE_PATH",
    "REUSE_STATIC_BACKGROUND_PATH",
    "REUSE_RECURRENT_ARTIFACT_MAP_PATH",
    "REUSE_MAP_SUPPORT_PATH",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "null", "n/a"}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def finite_int(value: Any) -> int | None:
    parsed = finite_float(value)
    return None if parsed is None else int(round(parsed))


def first_finite_float(*values: Any) -> float | None:
    for value in values:
        parsed = finite_float(value)
        if parsed is not None:
            return parsed
    return None


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def rebuild_masked_driver_environment(
    *,
    resolved_config_path: Path,
    output_dir: Path,
    mask_path: Path,
) -> dict[str, str]:
    """Return a beltmap-apply environment for a raw-frame rebuild with a defect mask."""

    resolved = load_json_object(resolved_config_path)
    raw_env = resolved.get("driver_environment")
    if not isinstance(raw_env, dict):
        raise ValueError(f"{resolved_config_path} does not contain driver_environment")
    env_updates = {str(key): str(value) for key, value in raw_env.items()}
    for env_var in REBUILD_MASKED_REUSE_ENV_VARS:
        env_updates.pop(env_var, None)
    env_updates["BELTMAP_OUTPUT_DIR"] = str(output_dir)
    env_updates["MAP_EXCLUSION_MASK_PATH"] = str(mask_path)
    env_updates["BELTMAP_STOP_AFTER_BELT_MAP"] = "1"
    return env_updates


def run_rebuild_masked_apply(
    *,
    resolved_config_path: Path,
    output_dir: Path,
    mask_path: Path,
) -> Path:
    """Run beltmap-apply to rebuild belt_map.npy while excluding defect pixels."""

    from beltmap.cli.apply import run_driver

    env_updates = rebuild_masked_driver_environment(
        resolved_config_path=resolved_config_path,
        output_dir=output_dir,
        mask_path=mask_path,
    )
    report = {
        "source": "beltmap-ghost-repair rebuild_masked",
        "resolved_config_path": str(resolved_config_path),
        "driver_environment": dict(sorted(env_updates.items())),
    }
    old_environ = dict(os.environ)
    try:
        run_driver(env_updates, report)
    finally:
        os.environ.clear()
        os.environ.update(old_environ)
        from beltmap import _driver_runtime as rt

        rt.refresh_runtime_paths()
    return output_dir / "belt_map.npy"


def load_phase_by_frame(path: Path) -> dict[float, float]:
    phases: dict[float, float] = {}
    for row in read_csv_rows(path):
        frame = finite_float(row.get("frame_index"))
        phase = finite_float(row.get("phase_px"))
        if frame is None or phase is None:
            continue
        phases[float(frame)] = float(phase)
    return phases


def phase_for_frame(
    frame_index: float,
    *,
    phase_by_frame: Mapping[float, float],
    metrics: Mapping[str, Any],
    map_height: int,
) -> float:
    if frame_index in phase_by_frame:
        return phase_by_frame[frame_index]
    rounded = float(round(frame_index))
    if rounded in phase_by_frame:
        return phase_by_frame[rounded]
    phase_source = metrics.get("phase_source", {}) if isinstance(metrics.get("phase_source"), dict) else {}
    config = metrics.get("detection_config", {}) if isinstance(metrics.get("detection_config"), dict) else {}
    period = first_finite_float(
        phase_source.get("period_px"),
        config.get("period_px"),
        float(map_height),
    )
    velocity = first_finite_float(
        phase_source.get("belt_velocity_px_per_frame"),
        config.get("belt_velocity_px_per_frame"),
    )
    if velocity is None:
        raise ValueError(
            f"no phase estimate for frame {frame_index}; provide phase_estimates.csv or map-only metrics with belt velocity"
        )
    assert period is not None
    return float((-velocity * frame_index) % period)


def track_rows_by_id(rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        track_id = finite_int(row.get("track_id"))
        if track_id is None:
            continue
        grouped.setdefault(track_id, []).append(row)
    return grouped


def selected_ghost_track_ids(
    *,
    tracks_by_id: Mapping[int, list[dict[str, str]]],
    track_scores: list[dict[str, str]],
    velocities: list[dict[str, str]],
    long_track_length: int,
) -> set[int]:
    selected: set[int] = {
        track_id for track_id, rows in tracks_by_id.items() if len(rows) >= long_track_length
    }
    score_track_ids: set[int] = set()
    for row in track_scores:
        track_id = finite_int(row.get("track_id"))
        if track_id is not None:
            score_track_ids.add(track_id)
            if truthy(row.get("accepted")):
                selected.add(track_id)
    if not selected and not score_track_ids:
        # Legacy outputs may have velocities but no track-score acceptance table.
        # In a map-only benchmark every velocity row is false by construction,
        # so use them as the best available accepted-track proxy.
        for row in velocities:
            track_id = finite_int(row.get("track_id"))
            if track_id is not None:
                selected.add(track_id)
    if not selected and tracks_by_id and not score_track_ids and not velocities:
        # Last-resort compatibility for very old map-only outputs with no long
        # or acceptance metadata.
        selected.update(tracks_by_id)
    return selected


def cyclic_delta(value: float, reference: float, period: float) -> float:
    return float((value - reference + 0.5 * period) % period - 0.5 * period)


def mark_bbox_in_belt_coordinates(
    counts: np.ndarray,
    *,
    phase_px: float,
    bbox_top: float,
    bbox_left: float,
    bbox_bottom: float,
    bbox_right: float,
    margin_px: int,
) -> None:
    height, width = counts.shape
    y0 = max(0, int(math.floor(bbox_top)) - margin_px)
    y1 = int(math.ceil(bbox_bottom)) + margin_px
    x0 = max(0, int(math.floor(bbox_left)) - margin_px)
    x1 = min(width, int(math.ceil(bbox_right)) + margin_px)
    if y1 <= y0 or x1 <= x0:
        return
    for crop_y in range(y0, y1):
        map_y_float = (phase_px + crop_y) % height
        row0 = int(math.floor(map_y_float)) % height
        row1 = (row0 + 1) % height
        counts[row0, x0:x1] += 1
        counts[row1, x0:x1] += 1


def build_ghost_defect_maps(
    *,
    belt_map_shape: tuple[int, int],
    tracks_by_id: Mapping[int, list[dict[str, str]]],
    selected_track_ids: set[int],
    phase_by_frame: Mapping[float, float],
    metrics: Mapping[str, Any],
    margin_px: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    counts = np.zeros(belt_map_shape, dtype=np.uint16)
    track_rows: list[dict[str, Any]] = []
    selected_detection_count = 0
    for track_id in sorted(selected_track_ids):
        rows = tracks_by_id.get(track_id, [])
        if not rows:
            continue
        centers_y: list[float] = []
        centers_x: list[float] = []
        peak_signals: list[float] = []
        map_boxes: list[tuple[float, float, float, float]] = []
        for row in rows:
            frame_index = finite_float(row.get("frame_index"))
            bbox_top = finite_float(row.get("bbox_top"))
            bbox_left = finite_float(row.get("bbox_left"))
            bbox_bottom = finite_float(row.get("bbox_bottom"))
            bbox_right = finite_float(row.get("bbox_right"))
            if None in {frame_index, bbox_top, bbox_left, bbox_bottom, bbox_right}:
                continue
            assert frame_index is not None
            assert bbox_top is not None and bbox_left is not None
            assert bbox_bottom is not None and bbox_right is not None
            phase_px = phase_for_frame(
                frame_index,
                phase_by_frame=phase_by_frame,
                metrics=metrics,
                map_height=belt_map_shape[0],
            )
            mark_bbox_in_belt_coordinates(
                counts,
                phase_px=phase_px,
                bbox_top=bbox_top,
                bbox_left=bbox_left,
                bbox_bottom=bbox_bottom,
                bbox_right=bbox_right,
                margin_px=margin_px,
            )
            selected_detection_count += 1
            cy = (phase_px + 0.5 * (bbox_top + bbox_bottom)) % belt_map_shape[0]
            cx = 0.5 * (bbox_left + bbox_right)
            centers_y.append(cy)
            centers_x.append(cx)
            peak = finite_float(row.get("peak_signal"))
            if peak is not None:
                peak_signals.append(peak)
            map_boxes.append(
                (
                    (phase_px + bbox_top) % belt_map_shape[0],
                    bbox_left,
                    (phase_px + bbox_bottom) % belt_map_shape[0],
                    bbox_right,
                )
            )
        if centers_y:
            reference = centers_y[0]
            unwrapped_y = np.asarray(
                [reference + cyclic_delta(value, reference, belt_map_shape[0]) for value in centers_y],
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


def local_inpaint_belt_map(
    belt_map: np.ndarray,
    mask: np.ndarray,
    *,
    radius_px: int = 16,
) -> np.ndarray:
    if belt_map.shape != mask.shape:
        raise ValueError("belt_map and mask must have the same shape")
    repaired = np.asarray(belt_map, dtype=np.float32).copy()
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return repaired
    global_fill = float(np.nanmedian(repaired))
    height, width = repaired.shape
    for y, x in zip(ys, xs, strict=True):
        y_idx = np.mod(np.arange(y - radius_px, y + radius_px + 1), height)
        x0 = max(0, x - radius_px)
        x1 = min(width, x + radius_px + 1)
        window = repaired[np.ix_(y_idx, np.arange(x0, x1))]
        invalid = mask[np.ix_(y_idx, np.arange(x0, x1))]
        values = window[(~invalid) & np.isfinite(window)]
        repaired[y, x] = float(np.median(values)) if values.size else global_fill
    return repaired


def config_from_map_only_metrics(metrics: Mapping[str, Any]) -> MapOnlyNegativeControlConfig:
    raw = metrics.get("detection_config", {})
    if not isinstance(raw, dict):
        return MapOnlyNegativeControlConfig()
    allowed = {field.name for field in fields(MapOnlyNegativeControlConfig)}
    kwargs = {key: raw[key] for key in raw if key in allowed}
    return MapOnlyNegativeControlConfig(**kwargs)


def map_only_metric_row(label: str, metrics: Mapping[str, Any], belt_map_path: Path) -> dict[str, Any]:
    detections = metrics.get("detections", {}) if isinstance(metrics.get("detections"), dict) else {}
    tracks = metrics.get("tracks", {}) if isinstance(metrics.get("tracks"), dict) else {}
    velocities = metrics.get("velocities", {}) if isinstance(metrics.get("velocities"), dict) else {}
    false_det = finite_int(detections.get("false_detections")) or 0
    false_long = finite_int(tracks.get("false_long_tracks")) or 0
    false_accepted = finite_int(velocities.get("false_accepted_tracks")) or 0
    proxy_penalty = 0.05 * false_det + false_long + false_accepted
    return {
        "map_variant": label,
        "belt_map_path": str(belt_map_path),
        "map_only_false_detections": false_det,
        "map_only_false_tracks": finite_int(tracks.get("false_tracks")) or 0,
        "map_only_false_long_tracks": false_long,
        "map_only_false_accepted_tracks": false_accepted,
        "map_only_proxy_ghost_penalty": proxy_penalty,
        "full100_labeled_metrics_status": "not_rerun",
    }


def normalize_for_image(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255).astype(np.uint8)


def defect_crop_slices(mask: np.ndarray, *, margin_px: int = 80, max_height: int = 900) -> tuple[slice, slice]:
    ys, xs = np.nonzero(mask)
    height, width = mask.shape
    if ys.size == 0:
        center = height // 2
        y0 = max(0, center - max_height // 2)
        y1 = min(height, y0 + max_height)
        return slice(y0, y1), slice(0, width)
    y0 = max(0, int(np.min(ys)) - margin_px)
    y1 = min(height, int(np.max(ys)) + margin_px + 1)
    if y1 - y0 > max_height:
        center = int(np.median(ys))
        y0 = max(0, center - max_height // 2)
        y1 = min(height, y0 + max_height)
    x0 = max(0, int(np.min(xs)) - margin_px)
    x1 = min(width, int(np.max(xs)) + margin_px + 1)
    return slice(y0, y1), slice(x0, x1)


def write_defect_overlay(path: Path, belt_map: np.ndarray, mask: np.ndarray) -> None:
    y_slice, x_slice = defect_crop_slices(mask)
    gray = normalize_for_image(belt_map[y_slice, x_slice])
    rgb = np.dstack([gray, gray, gray])
    mask_crop = mask[y_slice, x_slice]
    rgb[mask_crop] = np.asarray([255, 0, 80], dtype=np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), f"defect crop y={y_slice.start}:{y_slice.stop}, x={x_slice.start}:{x_slice.stop}", fill=(255, 255, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def write_before_after(path: Path, original: np.ndarray, repaired: np.ndarray, mask: np.ndarray) -> None:
    y_slice, x_slice = defect_crop_slices(mask)
    panels = []
    for title, values in [("original", original), ("mask", original), ("local_inpaint", repaired)]:
        gray = normalize_for_image(values[y_slice, x_slice])
        rgb = np.dstack([gray, gray, gray])
        if title == "mask":
            rgb[mask[y_slice, x_slice]] = np.asarray([255, 0, 80], dtype=np.uint8)
        image = Image.fromarray(rgb, mode="RGB")
        draw = ImageDraw.Draw(image)
        draw.text((8, 8), title, fill=(255, 255, 0))
        panels.append(image)
    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    out = Image.new("RGB", (width, height), "white")
    x = 0
    for panel in panels:
        out.paste(panel, (x, 0))
        x += panel.width
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def run_map_only_for_map(
    *,
    label: str,
    output_dir: Path,
    base_output_dir: Path,
    belt_map_path: Path,
    phase_estimates_path: Path | None,
    config: MapOnlyNegativeControlConfig,
) -> dict[str, Any]:
    run_dir = output_dir / f"map_only_{label}"
    result = generate_map_only_negative_control_report(
        output_dir=base_output_dir,
        config=config,
        belt_map_path=belt_map_path,
        phase_estimates_path=phase_estimates_path,
        metrics_path=run_dir / "metrics.json",
        report_path=run_dir / "report.md",
        detections_path=run_dir / "detections.csv",
        detections_per_frame_path=run_dir / "detections_per_frame.csv",
        tracks_path=run_dir / "tracks.csv",
        velocities_path=run_dir / "velocities.csv",
        track_scores_path=run_dir / "track_scores.csv",
    )
    return result.metrics


def write_report(
    path: Path,
    *,
    summary_rows: list[dict[str, Any]],
    selected_track_ids: set[int],
    defect_pixels: int,
    rebuild_status: str,
) -> None:
    lines = [
        "# GhostRepair Prototype",
        "",
        "GhostRepair uses map-only negative-control tracks to localize particle-like learned-map artifacts in belt coordinates, then repairs those pixels without overwriting the original map.",
        "",
        f"- Selected ghost tracks: {', '.join(str(v) for v in sorted(selected_track_ids)) or 'none'}",
        f"- Defect-mask pixels: {defect_pixels}",
        f"- Rebuild-masked status: {rebuild_status}",
        "",
        "## Map-only Check",
        "",
        "| map | false detections | false long | false accepted | proxy ghost penalty |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        proxy = finite_float(row.get("map_only_proxy_ghost_penalty"))
        proxy_text = "" if proxy is None else f"{proxy:.3f}"
        lines.append(
            f"| {row['map_variant']} | {row['map_only_false_detections']} | "
            f"{row['map_only_false_long_tracks']} | {row['map_only_false_accepted_tracks']} | "
            f"{proxy_text} |"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- `local_inpaint` is an intervention on the learned map only; it does not prove real-particle metrics improve.",
            "- Full100 labeled metrics must be rerun before claiming improved real detection.",
            "- `rebuild_masked` uses the same defect mask through `MAP_EXCLUSION_MASK_PATH` during raw-frame belt-map accumulation; it is not claimed as executed for this dataset unless the summary says so.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_defect_report(
    path: Path,
    *,
    track_rows: list[dict[str, Any]],
    defect_pixels: int,
    max_count: int,
    overlay_path: Path,
) -> None:
    def fmt_optional_float(value: Any) -> str:
        parsed = finite_float(value)
        return "" if parsed is None else f"{parsed:.3f}"

    lines = [
        "# Ghost Defect Localization",
        "",
        "This diagnostic projects map-only ghost-track detections back into belt-map coordinates.",
        "",
        f"- Defect pixels: {defect_pixels}",
        f"- Maximum defect count per pixel: {max_count}",
        f"- Overlay: `{overlay_path}`",
        "",
        "## Tracks",
        "",
        "| track_id | detections | map y min | map y max | map x min | map x max | max signal | belt-y RMS px | belt-x std px |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in track_rows:
        lines.append(
            f"| {row['track_id']} | {row['n_detections']} | "
            f"{float(row['map_y_min']):.3f} | {float(row['map_y_max']):.3f} | "
            f"{float(row['map_x_min']):.3f} | {float(row['map_x_max']):.3f} | "
            f"{fmt_optional_float(row.get('max_signal'))} | "
            f"{fmt_optional_float(row.get('belt_y_rms_px'))} | "
            f"{fmt_optional_float(row.get('belt_x_std_px'))} |"
        )
    lines.extend(
        [
            "",
            "The compact belt-coordinate footprint is the repair target. Every input detection/track is false by construction because it came from a map-only negative control.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
