from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw

from .tracking import (
    ParticleDetection,
    ParticleTrack,
    ParticleTrackingConfig,
    track_particle_detections,
)


@dataclass(frozen=True)
class VisualQcArtifacts:
    """Additional image artifacts written by visual quality-control checks."""

    plots: dict[str, Path]
    images: dict[str, list[Path]]


@dataclass(frozen=True)
class DetectionRecord:
    """Detection fields needed for visual overlay generation."""

    frame_index: int
    x: float
    y: float
    bbox_top: float
    bbox_left: float
    bbox_bottom: float
    bbox_right: float


TRACK_COLORS = [
    (220, 20, 60),
    (65, 105, 225),
    (34, 139, 34),
    (255, 140, 0),
    (138, 43, 226),
    (0, 139, 139),
    (178, 34, 34),
    (199, 21, 133),
]


def finite_float(value: Any) -> float | None:
    """Parse a finite float value, returning ``None`` for missing values."""

    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def finite_int(value: Any) -> int | None:
    """Parse an integer value, returning ``None`` for missing values."""

    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file, returning an empty list when it is absent."""

    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def draw_empty_plot(path: Path, title: str, message: str) -> None:
    """Write a placeholder diagnostic image."""

    width, height = 900, 420
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((72, 16), title, fill="black")
    draw.line([(72, height - 64), (width - 28, height - 64)], fill="black")
    draw.line([(72, 48), (72, height - 64)], fill="black")
    draw.text((88, 80), message, fill="black")
    image.save(path)


def draw_histogram(
    path: Path,
    *,
    title: str,
    values: Iterable[float],
    x_label: str,
) -> None:
    """Write a simple PIL histogram for scalar diagnostics."""

    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        draw_empty_plot(path, title, "No finite values available")
        return

    width, height = 900, 420
    left, right, top, bottom = 72, 28, 48, 64
    plot_width = width - left - right
    plot_height = height - top - bottom
    bins = max(5, min(30, int(np.sqrt(arr.size)) + 1))
    counts, edges = np.histogram(arr, bins=bins)
    max_count = int(np.max(counts)) if counts.size else 0

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((left, 16), title, fill="black")
    baseline = top + plot_height
    draw.line([(left, baseline), (left + plot_width, baseline)], fill="black")
    draw.line([(left, top), (left, baseline)], fill="black")

    if max_count > 0:
        bar_width = plot_width / len(counts)
        for index, count in enumerate(counts):
            x0 = left + index * bar_width
            x1 = left + (index + 1) * bar_width - 1
            y0 = baseline - (int(count) / max_count) * plot_height
            draw.rectangle((x0, y0, x1, baseline), fill="lightgray", outline="black")

    draw.text((left, height - 36), x_label, fill="black")
    draw.text((8, top), "count", fill="black")
    draw.text((left, baseline + 8), f"{float(edges[0]):.4g}", fill="black")
    draw.text((left + plot_width - 56, baseline + 8), f"{float(edges[-1]):.4g}", fill="black")
    draw.text((18, top - 8), str(max_count), fill="black")
    image.save(path)


def parse_frame_index_from_preview(path: Path) -> int | None:
    """Parse the frame index from ``residual_frame_000123.png``."""

    match = re.search(r"(\d+)", path.stem)
    return None if match is None else int(match.group(1))


def find_preview_paths(output_dir: Path) -> dict[int, Path]:
    """Return saved residual preview images keyed by processed frame index."""

    result: dict[int, Path] = {}
    for path in sorted(output_dir.glob("residual_frame_*.png")):
        frame_index = parse_frame_index_from_preview(path)
        if frame_index is not None:
            result[frame_index] = path
    return result


def residual_preview_values(preview_paths: dict[int, Path]) -> list[float]:
    """Collect display intensities from residual preview PNGs."""

    values: list[float] = []
    for path in preview_paths.values():
        image = Image.open(path).convert("L")
        arr = np.asarray(image, dtype=np.float64)
        values.extend(arr.ravel().tolist())
    return values


def belt_region_shape(metadata: dict[str, Any]) -> tuple[int, int] | None:
    """Return ``(height, width)`` for the processed belt crop."""

    region = metadata.get("belt_region")
    if not isinstance(region, dict):
        return None
    height = finite_int(region.get("height"))
    width = finite_int(region.get("width"))
    if height is None or width is None or height <= 0 or width <= 0:
        return None
    return height, width


def estimate_belt_map_coverage(
    phase_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> np.ndarray | None:
    """Estimate nominal belt-map row coverage from phase estimates.

    This uses the recovered phase trajectory and crop height. It is a QC proxy
    for observation coverage, not the exact accumulation mask used by the driver.
    """

    height = finite_int(metadata.get("belt_map_height_px"))
    shape = belt_region_shape(metadata)
    if height is None or height <= 0 or shape is None:
        return None

    crop_height, _crop_width = shape
    row_counts = np.zeros(height, dtype=np.float64)
    for row in phase_rows:
        phase = finite_float(row.get("phase_px"))
        if phase is None:
            continue
        positions = np.arange(crop_height, dtype=np.float64) + phase
        y0 = np.floor(positions).astype(int)
        frac = positions - y0
        y1 = y0 + 1
        y0 %= height
        y1 %= height
        np.add.at(row_counts, y0, 1.0 - frac)
        np.add.at(row_counts, y1, frac)

    return row_counts


def draw_coverage_image(path: Path, coverage: np.ndarray | None) -> None:
    """Write the nominal belt-map coverage image."""

    if coverage is None or coverage.size == 0 or not np.isfinite(coverage).any():
        draw_empty_plot(path, "Belt-map coverage", "Coverage unavailable")
        return

    arr = np.asarray(coverage, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim != 2:
        draw_empty_plot(path, "Belt-map coverage", "Coverage must be 1-D or 2-D")
        return
    finite = arr[np.isfinite(arr)]
    low = float(np.min(finite))
    high = float(np.max(finite))
    if high <= low:
        scaled = np.zeros_like(arr, dtype=np.uint8)
    else:
        scaled = np.clip((arr - low) / (high - low), 0.0, 1.0)
        scaled = np.round(255.0 * scaled).astype(np.uint8)

    image = Image.fromarray(scaled, mode="L")
    if image.width == 1:
        display_width = min(1024, max(64, image.height // 4))
        image = image.resize((display_width, image.height), Image.Resampling.NEAREST)
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    covered_rows = int(np.count_nonzero(np.nanmax(arr, axis=1) > 0))
    draw.text((8, 8), "nominal belt-map coverage", fill=(255, 0, 0))
    draw.text((8, image.height - 18), f"covered rows: {covered_rows}", fill=(255, 0, 0))
    image.save(path)


def parse_detection_records(rows: Iterable[dict[str, Any]]) -> list[DetectionRecord]:
    """Parse detection CSV rows for overlay rendering."""

    records: list[DetectionRecord] = []
    for row in rows:
        frame_index = finite_int(row.get("frame_index"))
        x = finite_float(row.get("x"))
        y = finite_float(row.get("y"))
        top = finite_float(row.get("bbox_top"))
        left = finite_float(row.get("bbox_left"))
        bottom = finite_float(row.get("bbox_bottom"))
        right = finite_float(row.get("bbox_right"))
        if None in (frame_index, x, y, top, left, bottom, right):
            continue
        assert frame_index is not None
        assert x is not None and y is not None
        assert top is not None and left is not None
        assert bottom is not None and right is not None
        records.append(
            DetectionRecord(
                frame_index=frame_index,
                x=x,
                y=y,
                bbox_top=top,
                bbox_left=left,
                bbox_bottom=bottom,
                bbox_right=right,
            )
        )
    return records


def group_detections_by_frame(
    records: Iterable[DetectionRecord],
) -> dict[int, list[DetectionRecord]]:
    """Group detections by processed frame index."""

    grouped: dict[int, list[DetectionRecord]] = {}
    for record in records:
        grouped.setdefault(record.frame_index, []).append(record)
    return grouped


def draw_detection_overlays(
    output_dir: Path,
    *,
    preview_paths: dict[int, Path],
    detections_by_frame: dict[int, list[DetectionRecord]],
) -> list[Path]:
    """Overlay detection boxes and centroids on residual preview frames."""

    created: list[Path] = []
    for frame_index, preview_path in preview_paths.items():
        image = Image.open(preview_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        for detection in detections_by_frame.get(frame_index, []):
            box = (
                int(round(detection.bbox_left)),
                int(round(detection.bbox_top)),
                int(round(detection.bbox_right)),
                int(round(detection.bbox_bottom)),
            )
            draw.rectangle(box, outline=(0, 255, 0), width=2)
            cx = int(round(detection.x))
            cy = int(round(detection.y))
            draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill=(255, 0, 0))
        draw.text((8, 8), f"detections frame={frame_index}", fill=(255, 255, 0))
        out_path = output_dir / f"detections_overlay_sample_{frame_index:06d}.png"
        image.save(out_path)
        created.append(out_path)
    return created


def _particle_detection_from_record(record: DetectionRecord, *, label: int) -> ParticleDetection:
    width = max(0.0, record.bbox_right - record.bbox_left)
    height = max(0.0, record.bbox_bottom - record.bbox_top)
    area = max(
        1,
        int(round(width * height)),
    )
    return ParticleDetection(
        frame_index=float(record.frame_index),
        label=label,
        y=record.y,
        x=record.x,
        area_px=area,
        bbox_top=int(round(record.bbox_top)),
        bbox_left=int(round(record.bbox_left)),
        bbox_bottom=int(round(record.bbox_bottom)),
        bbox_right=int(round(record.bbox_right)),
    )


def reconstruct_tracks(
    detections_by_frame: dict[int, list[DetectionRecord]],
    *,
    max_match_distance_px: float,
    max_frame_gap: float = 2.0,
) -> list[ParticleTrack]:
    """Reconstruct overlay tracks with the same PyRecEst-backed tracker as the driver."""

    frame_indices = sorted(detections_by_frame)
    detections = [
        [
            _particle_detection_from_record(record, label=index + 1)
            for index, record in enumerate(detections_by_frame[frame_index])
        ]
        for frame_index in frame_indices
    ]
    return track_particle_detections(
        detections,
        config=ParticleTrackingConfig(
            max_match_distance_px=max_match_distance_px,
            max_frame_gap=max_frame_gap,
        ),
        frame_indices=frame_indices,
    )


def draw_track_overlays(
    output_dir: Path,
    *,
    preview_paths: dict[int, Path],
    detections_by_frame: dict[int, list[DetectionRecord]],
    tracks: list[ParticleTrack],
) -> list[Path]:
    """Overlay reconstructed track polylines on residual preview frames."""

    created: list[Path] = []
    for frame_index, preview_path in preview_paths.items():
        image = Image.open(preview_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        for track in tracks:
            visible = [
                (detection.x, detection.y)
                for detection in track.detections
                if detection.frame_index <= frame_index
            ]
            if len(visible) < 2:
                continue
            color = TRACK_COLORS[track.track_id % len(TRACK_COLORS)]
            polyline = [(int(round(x)), int(round(y))) for x, y in visible]
            draw.line(polyline, fill=color, width=2)
            end_x, end_y = polyline[-1]
            draw.ellipse((end_x - 3, end_y - 3, end_x + 3, end_y + 3), fill=color)
            draw.text((end_x + 4, end_y - 4), str(track.track_id), fill=color)

        for detection in detections_by_frame.get(frame_index, []):
            box = (
                int(round(detection.bbox_left)),
                int(round(detection.bbox_top)),
                int(round(detection.bbox_right)),
                int(round(detection.bbox_bottom)),
            )
            draw.rectangle(box, outline=(255, 255, 255), width=1)

        draw.text((8, 8), f"tracks frame={frame_index}", fill=(255, 255, 0))
        out_path = output_dir / f"tracks_overlay_sample_{frame_index:06d}.png"
        image.save(out_path)
        created.append(out_path)
    return created


def make_thumbnail(path: Path, *, size: tuple[int, int]) -> Image.Image:
    """Load an image and return a thumbnail on a white RGB canvas."""

    image = Image.open(path).convert("RGB")
    image.thumbnail(size)
    canvas = Image.new("RGB", size, "white")
    offset = ((size[0] - image.width) // 2, (size[1] - image.height) // 2)
    canvas.paste(image, offset)
    return canvas


def draw_overlay_contact_sheet(
    output_dir: Path,
    *,
    detection_overlays: list[Path],
    track_overlays: list[Path],
    max_rows: int = 6,
) -> Path:
    """Tile detection and track overlays into one quick-look contact sheet."""

    out_path = output_dir / "overlay_contact_sheet.png"
    detection_by_frame = {
        frame: path
        for path in detection_overlays
        if (frame := parse_frame_index_from_preview(path)) is not None
    }
    track_by_frame = {
        frame: path
        for path in track_overlays
        if (frame := parse_frame_index_from_preview(path)) is not None
    }
    frame_indices = sorted(set(detection_by_frame) | set(track_by_frame))[:max_rows]
    if not frame_indices:
        draw_empty_plot(out_path, "Overlay contact sheet", "No overlay samples available")
        return out_path

    thumb_size = (520, 360)
    margin = 24
    header_h = 54
    label_h = 24
    row_h = label_h + thumb_size[1] + margin
    width = 2 * thumb_size[0] + 3 * margin
    height = header_h + len(frame_indices) * row_h + margin
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 18), "Overlay contact sheet", fill="black")
    draw.text((margin, header_h - 20), "Detections", fill="black")
    draw.text((2 * margin + thumb_size[0], header_h - 20), "Tracks", fill="black")

    for row_index, frame_index in enumerate(frame_indices):
        y = header_h + row_index * row_h
        draw.text((margin, y), f"frame {frame_index}", fill="black")
        det_path = detection_by_frame.get(frame_index)
        track_path = track_by_frame.get(frame_index)
        if det_path is not None:
            sheet.paste(make_thumbnail(det_path, size=thumb_size), (margin, y + label_h))
        if track_path is not None:
            sheet.paste(
                make_thumbnail(track_path, size=thumb_size),
                (2 * margin + thumb_size[0], y + label_h),
            )
        draw.rectangle(
            (
                margin,
                y + label_h,
                margin + thumb_size[0] - 1,
                y + label_h + thumb_size[1] - 1,
            ),
            outline="lightgray",
        )
        draw.rectangle(
            (
                2 * margin + thumb_size[0],
                y + label_h,
                2 * margin + 2 * thumb_size[0] - 1,
                y + label_h + thumb_size[1] - 1,
            ),
            outline="lightgray",
        )

    sheet.save(out_path)
    return out_path


def generate_visual_qc(output_dir: Path, data: dict[str, Any]) -> VisualQcArtifacts:
    """Generate residual, coverage, detection-overlay, and track-overlay QC."""

    output_dir.mkdir(parents=True, exist_ok=True)
    preview_paths = find_preview_paths(output_dir)
    detection_records = parse_detection_records(data.get("detections", []))
    detections_by_frame = group_detections_by_frame(detection_records)
    metadata = data.get("metadata", {})
    phase_rows = data.get("phase_rows", [])

    residual_histogram = output_dir / "residual_histogram.png"
    draw_histogram(
        residual_histogram,
        title="Residual preview histogram",
        values=residual_preview_values(preview_paths),
        x_label="preview intensity",
    )

    coverage_path = output_dir / "belt_map_coverage.png"
    coverage = estimate_belt_map_coverage(phase_rows, metadata)
    draw_coverage_image(coverage_path, coverage)

    detection_overlays = draw_detection_overlays(
        output_dir,
        preview_paths=preview_paths,
        detections_by_frame=detections_by_frame,
    )
    belt_velocity = finite_float(metadata.get("belt_velocity_px_per_frame")) or 0.0
    tracks = reconstruct_tracks(
        detections_by_frame,
        max_match_distance_px=max(8.0, 1.5 * abs(belt_velocity) + 5.0),
    )
    track_overlays = draw_track_overlays(
        output_dir,
        preview_paths=preview_paths,
        detections_by_frame=detections_by_frame,
        tracks=tracks,
    )
    contact_sheet = draw_overlay_contact_sheet(
        output_dir,
        detection_overlays=detection_overlays,
        track_overlays=track_overlays,
    )

    return VisualQcArtifacts(
        plots={
            "residual_histogram": residual_histogram,
            "belt_map_coverage": coverage_path,
            "overlay_contact_sheet": contact_sheet,
        },
        images={
            "detections_overlay": detection_overlays,
            "tracks_overlay": track_overlays,
        },
    )
