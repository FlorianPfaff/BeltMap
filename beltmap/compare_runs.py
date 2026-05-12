from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw

from .visual_qc import (
    DetectionRecord,
    find_preview_paths,
    group_detections_by_frame,
    parse_detection_records,
)


SUMMARY_FIELDS = [
    "label",
    "output_dir",
    "complete",
    "n_images",
    "detection_threshold",
    "n_detections",
    "n_tracks",
    "n_velocity_estimates",
    "n_filtered_velocity_estimates",
    "detections_per_frame_mean",
    "detections_per_frame_median",
    "detections_per_frame_max",
    "detection_area_median_px",
    "small_component_share_area_le_8",
    "velocity_ratio_median",
    "velocity_ratio_q25",
    "velocity_ratio_q75",
    "velocity_ratio_share_0_to_1",
    "filtered_velocity_ratio_median",
    "filtered_velocity_ratio_share_0_to_1",
    "velocity_track_length_median",
    "long_velocity_tracks_ge_5",
    "long_velocity_tracks_ge_10",
    "elapsed_s",
]


PLOT_COLORS = [
    (31, 119, 180),
    (214, 39, 40),
    (44, 160, 44),
    (255, 127, 14),
    (148, 103, 189),
    (23, 190, 207),
]


@dataclass(frozen=True)
class RunSpec:
    """Named BeltMap output directory to include in a comparison."""

    label: str
    output_dir: Path


@dataclass(frozen=True)
class RunData:
    """Loaded BeltMap output data for one run."""

    spec: RunSpec
    metadata: dict[str, Any]
    detections: list[dict[str, str]]
    detections_per_frame: list[dict[str, str]]
    velocities: list[dict[str, str]]
    filtered_velocities: list[dict[str, str]]
    filtered_tracks: list[dict[str, str]]
    preview_paths: dict[int, Path]
    detections_by_frame: dict[int, list[DetectionRecord]]
    filtered_detections_by_frame: dict[int, list[DetectionRecord]]


@dataclass(frozen=True)
class ComparisonArtifacts:
    """Files written by the multi-run comparison report."""

    report: Path
    summary_csv: Path
    plots: dict[str, Path]
    images: dict[str, Path]


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object, returning an empty object when the file is absent."""

    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV rows, returning an empty list when the file is absent."""

    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_float(value: Any) -> float | None:
    """Parse a finite float value, returning ``None`` for blanks."""

    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def finite_values(rows: Iterable[dict[str, Any]], field: str) -> list[float]:
    """Collect finite values from a CSV-like row sequence."""

    values: list[float] = []
    for row in rows:
        value = finite_float(row.get(field))
        if value is not None:
            values.append(value)
    return values


def describe(values: Iterable[float]) -> dict[str, float | int | None]:
    """Return compact scalar statistics for finite values."""

    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "q25": None,
            "q75": None,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
    }


def format_value(value: Any, *, digits: int = 4) -> str:
    """Format report table values without noisy trailing zeros."""

    if value is None:
        return "n/a"
    if isinstance(value, float):
        if not np.isfinite(value):
            return "n/a"
        if abs(value) >= 1000 or (0 < abs(value) < 1e-3):
            return f"{value:.{digits}g}"
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def parse_run_spec(value: str) -> RunSpec:
    """Parse ``LABEL=PATH`` or ``PATH`` into a run specification."""

    if "=" in value:
        label, path_text = value.split("=", 1)
        label = label.strip()
        path = Path(path_text.strip())
    else:
        path = Path(value.strip())
        label = path.name
    if not label:
        raise ValueError("run label must not be empty")
    if not str(path):
        raise ValueError("run output directory must not be empty")
    return RunSpec(label=label, output_dir=path)


def load_run_data(spec: RunSpec) -> RunData:
    """Load standard outputs from one BeltMap run directory."""

    detections = read_csv_rows(spec.output_dir / "detections.csv")
    records = parse_detection_records(detections)
    filtered_tracks = read_csv_rows(spec.output_dir / "filtered_tracks.csv")
    filtered_records = parse_detection_records(filtered_tracks)
    return RunData(
        spec=spec,
        metadata=read_json(spec.output_dir / "metadata.json"),
        detections=detections,
        detections_per_frame=read_csv_rows(spec.output_dir / "detections_per_frame.csv"),
        velocities=read_csv_rows(spec.output_dir / "velocities.csv"),
        filtered_velocities=read_csv_rows(spec.output_dir / "filtered_velocities.csv"),
        filtered_tracks=filtered_tracks,
        preview_paths=find_preview_paths(spec.output_dir),
        detections_by_frame=group_detections_by_frame(records),
        filtered_detections_by_frame=group_detections_by_frame(filtered_records),
    )


def safe_share(numerator: int, denominator: int) -> float | None:
    """Return a share, or ``None`` when the denominator is zero."""

    return None if denominator == 0 else numerator / denominator


def metadata_or_count(data: RunData, key: str, rows: list[Any]) -> int | None:
    """Use metadata count when present, otherwise fall back to loaded rows."""

    value = finite_float(data.metadata.get(key))
    return int(value) if value is not None else len(rows)


def summarize_run(data: RunData) -> dict[str, Any]:
    """Compute one row of comparison metrics for a run."""

    detection_counts = finite_values(data.detections_per_frame, "n_detections")
    detection_areas = finite_values(data.detections, "area_px")
    velocity_ratios = finite_values(data.velocities, "velocity_ratio_y")
    filtered_velocity_ratios = finite_values(data.filtered_velocities, "velocity_ratio_y")
    velocity_track_lengths = finite_values(data.velocities, "n_detections")
    detection_stats = describe(detection_counts)
    area_stats = describe(detection_areas)
    ratio_stats = describe(velocity_ratios)
    filtered_ratio_stats = describe(filtered_velocity_ratios)
    length_stats = describe(velocity_track_lengths)
    small_components = sum(1 for value in detection_areas if value <= 8.0)
    plausible_ratios = sum(1 for value in velocity_ratios if 0.0 <= value <= 1.0)
    plausible_filtered_ratios = sum(
        1 for value in filtered_velocity_ratios if 0.0 <= value <= 1.0
    )
    long_ge_5 = sum(1 for value in velocity_track_lengths if value >= 5.0)
    long_ge_10 = sum(1 for value in velocity_track_lengths if value >= 10.0)
    detection_threshold = finite_float(data.metadata.get("detection_threshold"))
    if detection_threshold is None:
        config = read_json(data.spec.output_dir / "config_resolved.json")
        options = config.get("options", {}) if isinstance(config, dict) else {}
        threshold_option = options.get("detection_threshold", {})
        if isinstance(threshold_option, dict):
            detection_threshold = finite_float(threshold_option.get("value"))

    return {
        "label": data.spec.label,
        "output_dir": str(data.spec.output_dir),
        "complete": (data.spec.output_dir / "metadata.json").is_file(),
        "n_images": data.metadata.get("n_images") or len(data.detections_per_frame) or None,
        "detection_threshold": detection_threshold,
        "n_detections": metadata_or_count(data, "n_detections", data.detections),
        "n_tracks": data.metadata.get("n_tracks"),
        "n_velocity_estimates": metadata_or_count(data, "n_velocity_estimates", data.velocities),
        "n_filtered_velocity_estimates": metadata_or_count(
            data,
            "n_filtered_velocity_estimates",
            data.filtered_velocities,
        ),
        "detections_per_frame_mean": detection_stats["mean"],
        "detections_per_frame_median": detection_stats["median"],
        "detections_per_frame_max": detection_stats["max"],
        "detection_area_median_px": area_stats["median"],
        "small_component_share_area_le_8": safe_share(small_components, len(detection_areas)),
        "velocity_ratio_median": ratio_stats["median"],
        "velocity_ratio_q25": ratio_stats["q25"],
        "velocity_ratio_q75": ratio_stats["q75"],
        "velocity_ratio_share_0_to_1": safe_share(plausible_ratios, len(velocity_ratios)),
        "filtered_velocity_ratio_median": filtered_ratio_stats["median"],
        "filtered_velocity_ratio_share_0_to_1": safe_share(
            plausible_filtered_ratios,
            len(filtered_velocity_ratios),
        ),
        "velocity_track_length_median": length_stats["median"],
        "long_velocity_tracks_ge_5": long_ge_5,
        "long_velocity_tracks_ge_10": long_ge_10,
        "elapsed_s": data.metadata.get("elapsed_s"),
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the comparison summary table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in SUMMARY_FIELDS} for row in rows)


def scale_series(
    series: list[tuple[list[float], list[float]]],
    *,
    width: int,
    height: int,
    left: int,
    right: int,
    top: int,
    bottom: int,
) -> list[list[tuple[int, int]]]:
    """Scale multiple x/y series to the same plot coordinates."""

    all_x = np.asarray([x for xs, _ in series for x in xs], dtype=np.float64)
    all_y = np.asarray([y for _, ys in series for y in ys], dtype=np.float64)
    if all_x.size == 0 or all_y.size == 0:
        return []
    x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
    y_min, y_max = float(np.min(all_y)), float(np.max(all_y))
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5
    plot_width = width - left - right
    plot_height = height - top - bottom
    result: list[list[tuple[int, int]]] = []
    for xs, ys in series:
        x_arr = np.asarray(xs, dtype=np.float64)
        y_arr = np.asarray(ys, dtype=np.float64)
        scaled_x = left + (x_arr - x_min) / (x_max - x_min) * plot_width
        scaled_y = top + plot_height - (y_arr - y_min) / (y_max - y_min) * plot_height
        result.append([(int(round(x)), int(round(y))) for x, y in zip(scaled_x, scaled_y)])
    return result


def draw_multiline_plot(
    path: Path,
    *,
    title: str,
    labeled_series: list[tuple[str, list[float], list[float]]],
    x_label: str,
    y_label: str,
) -> None:
    """Draw a compact multi-run line plot with PIL."""

    width, height = 1000, 460
    left, right, top, bottom = 76, 160, 48, 64
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    plot_width = width - left - right
    plot_height = height - top - bottom
    baseline = top + plot_height
    draw.text((left, 16), title, fill="black")
    draw.line([(left, baseline), (left + plot_width, baseline)], fill="black")
    draw.line([(left, top), (left, baseline)], fill="black")
    draw.text((left, height - 36), x_label, fill="black")
    draw.text((8, top), y_label, fill="black")

    nonempty = [(label, xs, ys) for label, xs, ys in labeled_series if xs and ys]
    scaled = scale_series(
        [(xs, ys) for _, xs, ys in nonempty],
        width=width,
        height=height,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
    )
    if not scaled:
        draw.text((left + 16, top + 32), "No finite values available", fill="black")
        image.save(path)
        return

    for index, ((label, _xs, _ys), points) in enumerate(zip(nonempty, scaled)):
        color = PLOT_COLORS[index % len(PLOT_COLORS)]
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        else:
            draw.line(points, fill=color, width=2)
        legend_y = top + 18 * index
        draw.rectangle((left + plot_width + 18, legend_y + 4, left + plot_width + 30, legend_y + 14), fill=color)
        draw.text((left + plot_width + 36, legend_y), label, fill="black")

    image.save(path)


def draw_velocity_ratio_histogram(path: Path, runs: list[RunData]) -> None:
    """Draw overlaid line histograms for velocity-ratio distributions."""

    values_by_run = [(run.spec.label, finite_values(run.velocities, "velocity_ratio_y")) for run in runs]
    all_values = np.asarray([value for _, values in values_by_run for value in values], dtype=np.float64)
    if all_values.size == 0:
        draw_multiline_plot(
            path,
            title="Velocity-ratio distributions",
            labeled_series=[],
            x_label="velocity_ratio_y",
            y_label="count",
        )
        return

    finite = all_values[np.isfinite(all_values)]
    low, high = np.percentile(finite, [1, 99])
    if low == high:
        low -= 0.5
        high += 0.5
    bins = np.linspace(float(low), float(high), 31)
    labeled_series: list[tuple[str, list[float], list[float]]] = []
    centers = ((bins[:-1] + bins[1:]) / 2.0).tolist()
    for label, values in values_by_run:
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        counts, _edges = np.histogram(arr, bins=bins)
        labeled_series.append((label, centers, counts.astype(float).tolist()))

    draw_multiline_plot(
        path,
        title="Velocity-ratio distributions",
        labeled_series=labeled_series,
        x_label="velocity_ratio_y",
        y_label="count",
    )


def detection_count_for_frame(data: RunData, frame_index: int, *, filtered: bool = False) -> int:
    """Return the loaded detection count for a frame."""

    detections = data.filtered_detections_by_frame if filtered else data.detections_by_frame
    return len(detections.get(frame_index, []))


def draw_detection_overlay_tile(
    data: RunData,
    frame_index: int,
    *,
    width: int,
    header_height: int,
    filtered: bool = False,
) -> Image.Image:
    """Create a resized residual-preview tile with detection boxes."""

    detections_by_frame = (
        data.filtered_detections_by_frame
        if filtered
        else data.detections_by_frame
    )
    preview = data.preview_paths.get(frame_index)
    if preview is None:
        tile = Image.new("RGB", (width, header_height + 220), (248, 248, 248))
        draw = ImageDraw.Draw(tile)
        prefix = "filtered " if filtered else ""
        draw.text((10, 10), f"{data.spec.label} {prefix}frame {frame_index}", fill="black")
        draw.text((10, header_height + 82), "preview missing", fill=(90, 90, 90))
        return tile

    image = Image.open(preview).convert("RGB")
    draw = ImageDraw.Draw(image)
    for detection in detections_by_frame.get(frame_index, []):
        box = (
            int(round(detection.bbox_left)),
            int(round(detection.bbox_top)),
            int(round(detection.bbox_right)),
            int(round(detection.bbox_bottom)),
        )
        draw.rectangle(box, outline=(0, 255, 0), width=2)

    scale = width / image.width
    body_height = max(1, int(round(image.height * scale)))
    image = image.resize((width, body_height), Image.Resampling.BILINEAR)
    tile = Image.new("RGB", (width, header_height + body_height), "white")
    tile.paste(image, (0, header_height))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, width - 1, header_height - 1), fill=(245, 245, 245), outline=(180, 180, 180))
    prefix = "filtered | " if filtered else ""
    label = (
        f"{data.spec.label} | {prefix}frame {frame_index} | "
        f"n={detection_count_for_frame(data, frame_index, filtered=filtered)}"
    )
    draw.text((8, 8), label, fill="black")
    return tile


def draw_detection_contact_sheet(
    path: Path,
    runs: list[RunData],
    *,
    frames: list[int],
    tile_width: int = 420,
    filtered: bool = False,
) -> None:
    """Write a side-by-side residual-preview contact sheet with detection boxes."""

    if not frames:
        frames = sorted({frame for run in runs for frame in run.preview_paths})[:5]
    if not frames:
        frames = [0]
    header_height = 30
    margin = 10
    tiles: list[list[Image.Image]] = []
    for frame_index in frames:
        tiles.append(
            [
                draw_detection_overlay_tile(
                    run,
                    frame_index,
                    width=tile_width,
                    header_height=header_height,
                    filtered=filtered,
                )
                for run in runs
            ]
        )
    row_heights = [max(tile.height for tile in row) for row in tiles]
    sheet_width = margin + len(runs) * (tile_width + margin)
    sheet_height = margin + sum(row_heights) + len(row_heights) * margin
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    y = margin
    for row, row_height in zip(tiles, row_heights):
        x = margin
        for tile in row:
            sheet.paste(tile, (x, y))
            x += tile_width + margin
        y += row_height + margin
    sheet.save(path)


def write_plots(report_dir: Path, runs: list[RunData]) -> tuple[dict[str, Path], dict[str, Path]]:
    """Write comparison plots and contact sheets."""

    plots = {
        "detections_per_frame": report_dir / "detections_per_frame_comparison.png",
        "velocity_ratio_histogram": report_dir / "velocity_ratio_histogram_comparison.png",
    }
    images = {
        "detection_contact_sheet": report_dir / "detection_contact_sheet.png",
        "filtered_detection_contact_sheet": report_dir / "filtered_detection_contact_sheet.png",
    }
    series = []
    for run in runs:
        xs = finite_values(run.detections_per_frame, "frame_index")
        if not xs:
            xs = [float(index) for index in range(len(run.detections_per_frame))]
        ys = finite_values(run.detections_per_frame, "n_detections")
        series.append((run.spec.label, xs, ys))
    draw_multiline_plot(
        plots["detections_per_frame"],
        title="Detections per frame",
        labeled_series=series,
        x_label="frame_index",
        y_label="n_detections",
    )
    draw_velocity_ratio_histogram(plots["velocity_ratio_histogram"], runs)
    return plots, images


def markdown_link(path: Path, *, relative_to: Path) -> str:
    """Return a Markdown-friendly relative path where possible."""

    try:
        target = path.relative_to(relative_to)
    except ValueError:
        target = path
    return str(target).replace("\\", "/")


def build_markdown_report(
    report_path: Path,
    *,
    rows: list[dict[str, Any]],
    plots: dict[str, Path],
    images: dict[str, Path],
) -> str:
    """Build the comparison Markdown report."""

    report_dir = report_path.parent
    table_fields = [
        ("label", "run"),
        ("complete", "complete"),
        ("detection_threshold", "threshold"),
        ("n_detections", "detections"),
        ("n_tracks", "tracks"),
        ("n_velocity_estimates", "velocity rows"),
        ("n_filtered_velocity_estimates", "filtered rows"),
        ("detections_per_frame_median", "median/frame"),
        ("detections_per_frame_max", "max/frame"),
        ("detection_area_median_px", "median area"),
        ("small_component_share_area_le_8", "small <=8px share"),
        ("velocity_ratio_median", "ratio median"),
        ("velocity_ratio_share_0_to_1", "ratio 0..1 share"),
        ("filtered_velocity_ratio_share_0_to_1", "filtered 0..1 share"),
        ("long_velocity_tracks_ge_10", "tracks >=10"),
    ]
    lines = [
        "# BeltMap run comparison",
        "",
        "This report compares detection-only or full BeltMap output directories.",
        "Use it as a decision aid, not as a ground-truth benchmark.",
        "",
        "## Summary",
        "",
        "| " + " | ".join(label for _field, label in table_fields) + " |",
        "| " + " | ".join("---" for _ in table_fields) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(format_value(row.get(field)) for field, _label in table_fields)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Visual comparison",
            "",
            f"![Detection contact sheet]({markdown_link(images['detection_contact_sheet'], relative_to=report_dir)})",
            "",
            "## Filtered-track visual comparison",
            "",
            f"![Filtered detection contact sheet]({markdown_link(images['filtered_detection_contact_sheet'], relative_to=report_dir)})",
            "",
            "## Detection counts",
            "",
            f"![Detections per frame]({markdown_link(plots['detections_per_frame'], relative_to=report_dir)})",
            "",
            "## Velocity ratios",
            "",
            f"![Velocity-ratio histogram]({markdown_link(plots['velocity_ratio_histogram'], relative_to=report_dir)})",
            "",
            "## Decision checklist",
            "",
            "- Prefer the run with fewer belt-scratch false positives in the contact sheet.",
            "- Prefer compact filled particles over hollow or fractured particle components.",
            "- Prefer a plausible velocity-ratio distribution, usually mostly between 0 and 1 for slower particles moving with the belt.",
            "- Prefer enough long tracks; a threshold that creates only tiny tracks is usually too permissive.",
            "- Treat total detection count alone as weak evidence because lower thresholds can simply fragment scratches.",
            "",
        ]
    )
    return "\n".join(lines)


def generate_comparison_report(
    specs: list[RunSpec],
    *,
    report_dir: Path,
    frames: list[int] | None = None,
) -> ComparisonArtifacts:
    """Generate summary CSV, comparison plots, and a Markdown report."""

    if len(specs) < 2:
        raise ValueError("at least two runs are required for comparison")
    report_dir.mkdir(parents=True, exist_ok=True)
    runs = [load_run_data(spec) for spec in specs]
    rows = [summarize_run(run) for run in runs]
    summary_csv = report_dir / "summary.csv"
    write_summary_csv(summary_csv, rows)
    plots, images = write_plots(report_dir, runs)
    draw_detection_contact_sheet(
        images["detection_contact_sheet"],
        runs,
        frames=[] if frames is None else frames,
    )
    draw_detection_contact_sheet(
        images["filtered_detection_contact_sheet"],
        runs,
        frames=[] if frames is None else frames,
        filtered=True,
    )
    report = report_dir / "comparison_report.md"
    report.write_text(
        build_markdown_report(report, rows=rows, plots=plots, images=images),
        encoding="utf-8",
    )
    return ComparisonArtifacts(report=report, summary_csv=summary_csv, plots=plots, images=images)
