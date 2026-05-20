from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw


PLOT_FILENAMES = {
    "phase_corrections": "phase_corrections.png",
    "phase_correction_timeseries": "phase_correction_timeseries.png",
    "registration_score": "registration_score.png",
    "detections_per_frame": "detections_per_frame.png",
    "velocity_ratio_histogram": "velocity_ratio_histogram.png",
    "track_length_histogram": "track_length_histogram.png",
}


@dataclass(frozen=True)
class PlotGeometry:
    width: int = 900
    height: int = 420
    left: int = 72
    right: int = 28
    top: int = 48
    bottom: int = 64

    @property
    def plot_width(self) -> int:
        return self.width - self.left - self.right

    @property
    def plot_height(self) -> int:
        return self.height - self.top - self.bottom


@dataclass(frozen=True)
class ValidationArtifacts:
    report: Path
    summary: Path
    plots: dict[str, Path]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beltmap-validate",
        description="Create a Markdown validation report and diagnostic PNG plots for a BeltMap output directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="BeltMap output directory to validate. Default: outputs",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        help="Markdown report path. Default: OUTPUT_DIR/validation_report.md",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write only the Markdown report and skip PNG plot generation.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print the generated artifact paths as JSON.",
    )
    return parser


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_progress_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rows.append(json.loads(stripped))
        except json.JSONDecodeError:
            continue
    return rows


def finite_float(value: Any) -> float | None:
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
    values: list[float] = []
    for row in rows:
        value = finite_float(row.get(field))
        if value is not None:
            values.append(value)
    return values


def paired_values(
    rows: Iterable[dict[str, Any]],
    *,
    x_field: str,
    y_field: str,
) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for index, row in enumerate(rows):
        y_value = finite_float(row.get(y_field))
        if y_value is None:
            continue
        x_value = finite_float(row.get(x_field))
        xs.append(float(index) if x_value is None else x_value)
        ys.append(y_value)
    return xs, ys


def describe(values: Iterable[float]) -> dict[str, float | int | None]:
    arr = np.asarray(list(values), dtype=np.float64)
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


def safe_share(count: int, total: int) -> float | None:
    return None if total <= 0 else float(count / total)


def format_value(value: Any, *, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if abs(value) >= 1000 or (0 < abs(value) < 1e-3):
            return f"{value:.{digits}g}"
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def load_run_data(output_dir: Path) -> dict[str, Any]:
    return {
        "metadata": read_json(output_dir / "metadata.json"),
        "config": read_json(output_dir / "config_resolved.json"),
        "progress": read_progress_jsonl(output_dir / "progress.jsonl"),
        "phase_rows": read_csv_rows(output_dir / "phase_estimates.csv"),
        "detections": read_csv_rows(output_dir / "detections.csv"),
        "detections_per_frame": read_csv_rows(output_dir / "detections_per_frame.csv"),
        "velocities": read_csv_rows(output_dir / "velocities.csv"),
    }


def make_canvas(title: str, geometry: PlotGeometry) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (geometry.width, geometry.height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((geometry.left, 16), title, fill="black")
    draw.line(
        [
            (geometry.left, geometry.top + geometry.plot_height),
            (geometry.left + geometry.plot_width, geometry.top + geometry.plot_height),
        ],
        fill="black",
    )
    draw.line(
        [(geometry.left, geometry.top), (geometry.left, geometry.top + geometry.plot_height)],
        fill="black",
    )
    return image, draw


def draw_empty_plot(path: Path, title: str, message: str) -> None:
    geometry = PlotGeometry()
    image, draw = make_canvas(title, geometry)
    draw.text((geometry.left + 16, geometry.top + 32), message, fill="black")
    image.save(path)


def scale_points(
    xs: Iterable[float],
    ys: Iterable[float],
    geometry: PlotGeometry,
) -> list[tuple[int, int]]:
    x_arr = np.asarray(list(xs), dtype=np.float64)
    y_arr = np.asarray(list(ys), dtype=np.float64)
    x_min = float(np.min(x_arr))
    x_max = float(np.max(x_arr))
    y_min = float(np.min(y_arr))
    y_max = float(np.max(y_arr))
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5
    x_scaled = geometry.left + (x_arr - x_min) / (x_max - x_min) * geometry.plot_width
    y_scaled = (
        geometry.top
        + geometry.plot_height
        - (y_arr - y_min) / (y_max - y_min) * geometry.plot_height
    )
    return [(int(round(x)), int(round(y))) for x, y in zip(x_scaled, y_scaled)]


def draw_line_plot(
    path: Path,
    *,
    title: str,
    xs: list[float],
    ys: list[float],
    x_label: str,
    y_label: str,
) -> None:
    if not xs or not ys:
        draw_empty_plot(path, title, "No finite values available")
        return

    geometry = PlotGeometry()
    image, draw = make_canvas(title, geometry)
    points = scale_points(xs, ys, geometry)
    if len(points) == 1:
        x, y = points[0]
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="black")
    else:
        draw.line(points, fill="black", width=2)
    draw.text((geometry.left, geometry.height - 36), x_label, fill="black")
    draw.text((8, geometry.top), y_label, fill="black")
    draw.text((geometry.left, geometry.top + geometry.plot_height + 8), format_value(min(xs)), fill="black")
    draw.text(
        (geometry.left + geometry.plot_width - 56, geometry.top + geometry.plot_height + 8),
        format_value(max(xs)),
        fill="black",
    )
    draw.text((18, geometry.top + geometry.plot_height - 8), format_value(min(ys)), fill="black")
    draw.text((18, geometry.top - 8), format_value(max(ys)), fill="black")
    image.save(path)


def draw_histogram(
    path: Path,
    *,
    title: str,
    values: list[float],
    x_label: str,
) -> None:
    if not values:
        draw_empty_plot(path, title, "No finite values available")
        return

    arr = np.asarray(values, dtype=np.float64)
    bins = max(5, min(30, int(np.sqrt(arr.size)) + 1))
    counts, edges = np.histogram(arr, bins=bins)
    max_count = int(np.max(counts)) if counts.size else 0
    geometry = PlotGeometry()
    image, draw = make_canvas(title, geometry)
    if max_count <= 0:
        image.save(path)
        return

    bar_width = geometry.plot_width / len(counts)
    baseline = geometry.top + geometry.plot_height
    for index, count in enumerate(counts):
        left = geometry.left + index * bar_width
        right = geometry.left + (index + 1) * bar_width - 1
        top = baseline - (int(count) / max_count) * geometry.plot_height
        draw.rectangle((left, top, right, baseline), fill="lightgray", outline="black")

    draw.text((geometry.left, geometry.height - 36), x_label, fill="black")
    draw.text((8, geometry.top), "count", fill="black")
    draw.text((geometry.left, baseline + 8), format_value(float(edges[0])), fill="black")
    draw.text(
        (geometry.left + geometry.plot_width - 56, baseline + 8),
        format_value(float(edges[-1])),
        fill="black",
    )
    draw.text((18, geometry.top - 8), str(max_count), fill="black")
    image.save(path)


def write_plots(output_dir: Path, data: dict[str, Any]) -> dict[str, Path]:
    paths = {name: output_dir / filename for name, filename in PLOT_FILENAMES.items()}
    phase_rows = data["phase_rows"]
    detection_rows = data["detections_per_frame"]
    velocity_rows = data["velocities"]

    draw_histogram(
        paths["phase_corrections"],
        title="Phase corrections",
        values=finite_values(phase_rows, "correction_px"),
        x_label="correction_px",
    )
    correction_xs, correction_ys = paired_values(
        phase_rows,
        x_field="frame_index",
        y_field="correction_px",
    )
    draw_line_plot(
        paths["phase_correction_timeseries"],
        title="Phase correction over time",
        xs=correction_xs,
        ys=correction_ys,
        x_label="frame_index",
        y_label="correction_px",
    )
    score_xs, score_ys = paired_values(
        phase_rows,
        x_field="frame_index",
        y_field="score",
    )
    draw_line_plot(
        paths["registration_score"],
        title="Registration score",
        xs=score_xs,
        ys=score_ys,
        x_label="frame_index",
        y_label="score",
    )
    detection_xs, detection_ys = paired_values(
        detection_rows,
        x_field="frame_index",
        y_field="n_detections",
    )
    draw_line_plot(
        paths["detections_per_frame"],
        title="Detections per frame",
        xs=detection_xs,
        ys=detection_ys,
        x_label="frame_index",
        y_label="n_detections",
    )
    draw_histogram(
        paths["velocity_ratio_histogram"],
        title="Velocity ratios",
        values=finite_values(velocity_rows, "velocity_ratio_y"),
        x_label="velocity_ratio_y",
    )
    draw_histogram(
        paths["track_length_histogram"],
        title="Track lengths",
        values=finite_values(velocity_rows, "n_detections"),
        x_label="detections per velocity track",
    )
    return paths


def missing_standard_files(output_dir: Path) -> list[str]:
    expected = [
        "metadata.json",
        "config_resolved.json",
        "progress.jsonl",
        "phase_estimates.csv",
        "detections.csv",
        "detections_per_frame.csv",
        "velocities.csv",
        "belt_map.png",
    ]
    return [name for name in expected if not (output_dir / name).exists()]


def final_belt_map_progress(progress_rows: list[dict[str, Any]]) -> dict[str, Any]:
    coverage_fields = {
        "observed_pixels",
        "total_pixels",
        "masked_pixels",
        "contributed_pixels",
    }
    belt_rows = [
        row
        for row in progress_rows
        if row.get("stage") == "belt_map" and coverage_fields.intersection(row)
    ]
    return belt_rows[-1] if belt_rows else {}


def markdown_link(path: Path, *, relative_to: Path) -> str:
    try:
        target = path.relative_to(relative_to)
    except ValueError:
        target = path
    return str(target).replace("\\", "/")


def build_markdown_report(
    output_dir: Path,
    report_path: Path,
    data: dict[str, Any],
    plots: dict[str, Path],
) -> str:
    metadata = data["metadata"]
    config = data["config"]
    phase_rows = data["phase_rows"]
    detection_rows = data["detections_per_frame"]
    velocity_rows = data["velocities"]
    progress_rows = data["progress"]

    corrections = finite_values(phase_rows, "correction_px")
    scores = finite_values(phase_rows, "score")
    detections = finite_values(detection_rows, "n_detections")
    velocity_ratios = finite_values(velocity_rows, "velocity_ratio_y")
    track_lengths = finite_values(velocity_rows, "n_detections")
    correction_stats = describe(corrections)
    score_stats = describe(scores)
    detection_stats = describe(detections)
    ratio_stats = describe(velocity_ratios)
    track_length_stats = describe(track_lengths)
    long_tracks_ge_5 = sum(1 for value in track_lengths if value >= 5)
    long_tracks_ge_10 = sum(1 for value in track_lengths if value >= 10)
    missing = missing_standard_files(output_dir)
    map_progress = final_belt_map_progress(progress_rows)

    def plot_line(key: str, label: str) -> str:
        path = plots.get(key)
        if path is None:
            return ""
        target = markdown_link(path, relative_to=report_path.parent)
        return f"![{label}]({target})\n"

    lines = [
        "# BeltMap validation report",
        "",
        f"Output directory: `{output_dir}`",
        "",
        "## Run summary",
        "",
        "| Quantity | Value |",
        "| --- | --- |",
        f"| selected frames | {format_value(metadata.get('n_images'))} |",
        f"| frame stride | {format_value(metadata.get('frame_stride'))} |",
        f"| belt velocity px/frame | {format_value(metadata.get('belt_velocity_px_per_frame'))} |",
        f"| belt map height px | {format_value(metadata.get('belt_map_height_px'))} |",
        f"| phase estimates | {format_value(metadata.get('n_phase_estimates'))} |",
        f"| detections | {format_value(metadata.get('n_detections'))} |",
        f"| tracks | {format_value(metadata.get('n_tracks'))} |",
        f"| velocity estimates | {format_value(metadata.get('n_velocity_estimates'))} |",
        "",
    ]
    if config:
        lines.extend(
            [
                "Resolved configuration: `config_resolved.json`",
                "",
            ]
        )
    if missing:
        lines.extend(
            [
                "## Missing standard files",
                "",
                *[f"- `{name}`" for name in missing],
                "",
            ]
        )

    lines.extend(
        [
            "## Phase registration",
            "",
            "| Statistic | correction_px | score |",
            "| --- | ---: | ---: |",
            f"| count | {correction_stats['count']} | {score_stats['count']} |",
            f"| median | {format_value(correction_stats['median'])} | {format_value(score_stats['median'])} |",
            f"| q25 | {format_value(correction_stats['q25'])} | {format_value(score_stats['q25'])} |",
            f"| q75 | {format_value(correction_stats['q75'])} | {format_value(score_stats['q75'])} |",
            f"| min | {format_value(correction_stats['min'])} | {format_value(score_stats['min'])} |",
            f"| max | {format_value(correction_stats['max'])} | {format_value(score_stats['max'])} |",
            "",
            plot_line("phase_corrections", "Phase correction histogram"),
            plot_line("phase_correction_timeseries", "Phase correction over time"),
            plot_line("registration_score", "Registration score over time"),
            "## Detections",
            "",
            "| Statistic | n_detections |",
            "| --- | ---: |",
            f"| count | {detection_stats['count']} |",
            f"| mean | {format_value(detection_stats['mean'])} |",
            f"| median | {format_value(detection_stats['median'])} |",
            f"| max | {format_value(detection_stats['max'])} |",
            "",
            plot_line("detections_per_frame", "Detections per frame"),
            "## Velocities",
            "",
            "| Statistic | velocity_ratio_y |",
            "| --- | ---: |",
            f"| count | {ratio_stats['count']} |",
            f"| median | {format_value(ratio_stats['median'])} |",
            f"| q25 | {format_value(ratio_stats['q25'])} |",
            f"| q75 | {format_value(ratio_stats['q75'])} |",
            f"| min | {format_value(ratio_stats['min'])} |",
            f"| max | {format_value(ratio_stats['max'])} |",
            "",
            plot_line("velocity_ratio_histogram", "Velocity-ratio histogram"),
            "## Track lengths",
            "",
            "| Statistic | n_detections per velocity track |",
            "| --- | ---: |",
            f"| count | {track_length_stats['count']} |",
            f"| median | {format_value(track_length_stats['median'])} |",
            f"| q25 | {format_value(track_length_stats['q25'])} |",
            f"| q75 | {format_value(track_length_stats['q75'])} |",
            f"| max | {format_value(track_length_stats['max'])} |",
            f"| tracks >= 5 detections | {long_tracks_ge_5} |",
            f"| tracks >= 10 detections | {long_tracks_ge_10} |",
            "",
            plot_line("track_length_histogram", "Track-length histogram"),
            "## Belt-map progress",
            "",
            "| Quantity | Value |",
            "| --- | --- |",
            f"| observed pixels | {format_value(map_progress.get('observed_pixels'))} |",
            f"| total pixels | {format_value(map_progress.get('total_pixels'))} |",
            f"| masked pixels | {format_value(map_progress.get('masked_pixels'))} |",
            f"| contributed pixels | {format_value(map_progress.get('contributed_pixels'))} |",
            "",
            "## Conclusion checklist",
            "",
            "- Check that phase corrections do not pile up at the search boundary.",
            "- Check that registration scores do not collapse for long intervals.",
            "- Check `belt_map.png` for particle ghosts or interpolation bands.",
            "- Check detection-count spikes against residual previews.",
            "- Check velocity-ratio outliers against the experiment physics.",
            "- Check that useful configurations produce enough long tracks, not just many tiny detections.",
            "",
        ]
    )
    return "\n".join(line for line in lines if line is not None)


def build_validation_summary(output_dir: Path, data: dict[str, Any]) -> dict[str, Any]:
    metadata = data["metadata"]
    phase_rows = data["phase_rows"]
    detection_rows = data["detections_per_frame"]
    velocity_rows = data["velocities"]
    progress_rows = data["progress"]

    corrections = finite_values(phase_rows, "correction_px")
    scores = finite_values(phase_rows, "score")
    detections = finite_values(detection_rows, "n_detections")
    velocity_ratios = finite_values(velocity_rows, "velocity_ratio_y")
    track_lengths = finite_values(velocity_rows, "n_detections")
    velocity_ratios_0_to_1 = sum(1 for value in velocity_ratios if 0.0 <= value <= 1.0)

    return {
        "output_dir": str(output_dir),
        "missing_standard_files": missing_standard_files(output_dir),
        "run": {
            "n_images": metadata.get("n_images"),
            "frame_stride": metadata.get("frame_stride"),
            "belt_velocity_px_per_frame": metadata.get("belt_velocity_px_per_frame"),
            "belt_map_height_px": metadata.get("belt_map_height_px"),
            "n_phase_estimates": metadata.get("n_phase_estimates"),
            "n_detections": metadata.get("n_detections"),
            "n_tracks": metadata.get("n_tracks"),
            "n_velocity_estimates": metadata.get("n_velocity_estimates"),
            "n_filtered_velocity_estimates": metadata.get("n_filtered_velocity_estimates"),
        },
        "phase_registration": {
            "correction_px": describe(corrections),
            "score": describe(scores),
        },
        "detections": {
            "per_frame": describe(detections),
            "zero_detection_frames": sum(1 for value in detections if value == 0),
        },
        "velocities": {
            "velocity_ratio_y": describe(velocity_ratios),
            "velocity_ratio_0_to_1_count": velocity_ratios_0_to_1,
            "velocity_ratio_0_to_1_share": safe_share(
                velocity_ratios_0_to_1,
                len(velocity_ratios),
            ),
        },
        "track_lengths": {
            "n_detections": describe(track_lengths),
            "tracks_ge_5": sum(1 for value in track_lengths if value >= 5),
            "tracks_ge_10": sum(1 for value in track_lengths if value >= 10),
        },
        "belt_map_progress": final_belt_map_progress(progress_rows),
    }


def generate_validation_report(
    output_dir: Path,
    *,
    report_path: Path | None = None,
    make_plots: bool = True,
) -> ValidationArtifacts:
    if not output_dir.is_dir():
        raise FileNotFoundError(f"BeltMap output directory does not exist: {output_dir}")
    report = report_path or (output_dir / "validation_report.md")
    summary_path = report.with_name("validation_summary.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    data = load_run_data(output_dir)
    plots = write_plots(output_dir, data) if make_plots else {}
    summary = build_validation_summary(output_dir, data)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown = build_markdown_report(output_dir, report, data, plots)
    report.write_text(markdown, encoding="utf-8")
    return ValidationArtifacts(report=report, summary=summary_path, plots=plots)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        artifacts = generate_validation_report(
            args.output_dir,
            report_path=args.report_path,
            make_plots=not args.no_plots,
        )
    except FileNotFoundError as exc:
        parser.error(str(exc))
    if not args.quiet:
        print(
            json.dumps(
                {
                    "report": str(artifacts.report),
                    "summary": str(artifacts.summary),
                    "plots": {key: str(path) for key, path in artifacts.plots.items()},
                },
                indent=2,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
