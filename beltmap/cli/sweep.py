from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import shutil
import subprocess
import sys
from html import escape
from pathlib import Path
from typing import Any

from beltmap.benchmark import generate_benchmark_report, read_json

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


SWEEP_METRIC_FIELDS = [
    "run_index",
    "output_dir",
    "overrides",
    "detection_threshold",
    "detection_low_threshold",
    "detection_precision",
    "detection_recall",
    "detection_f1",
    "false_positives_per_frame",
    "event_precision",
    "event_recall",
    "event_f1",
    "filtered_event_precision",
    "filtered_event_recall",
    "filtered_event_f1",
    "mean_track_length",
    "median_track_length",
    "single_frame_tracks",
    "single_frame_track_fraction",
    "filtered_mean_track_length",
    "filtered_median_track_length",
    "filtered_single_frame_tracks",
    "filtered_single_frame_track_fraction",
    "track_fragmentation",
    "filtered_track_fragmentation",
    "fragmented_truth_events",
    "mean_fragments_per_truth_event",
    "birth_false_positive_rate",
    "missed_event_rate",
    "velocity_y_error_px_per_frame",
    "velocity_y_mean_abs_error_px_per_frame",
    "velocity_y_bias_px_per_frame",
    "velocity_y_error_std_px_per_frame",
    "truth_matched_velocity_y_error_px_per_frame",
    "filtered_velocity_y_error_px_per_frame",
    "filtered_velocity_y_mean_abs_error_px_per_frame",
    "filtered_velocity_y_bias_px_per_frame",
    "filtered_velocity_y_error_std_px_per_frame",
    "filtered_truth_matched_velocity_y_error_px_per_frame",
    "phase_rmse_px",
    "map_rmse_gray",
]


def parse_scalar(value: str) -> Any:
    text = value.strip()
    if text == "":
        raise ValueError("sweep parameter values must not be empty")
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if text.lower() in {"none", "null"}:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        parsed = float(text)
    except ValueError:
        return text
    if not math.isfinite(parsed):
        raise ValueError("numeric sweep parameter values must be finite")
    return parsed


def dotted_key_parts(dotted_key: str) -> list[str]:
    parts = [part.strip() for part in dotted_key.split(".")]
    if not parts or any(part == "" for part in parts):
        raise ValueError("dotted keys must contain non-empty path components")
    return parts


def parse_iou_threshold(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "IoU threshold must be finite and in [0, 1]"
        ) from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("IoU threshold must be finite and in [0, 1]")
    return parsed


def parse_param(value: str) -> tuple[str, list[Any]]:
    if "=" not in value:
        raise ValueError("parameters must be KEY=VALUE1,VALUE2")
    key, raw_values = value.split("=", 1)
    key = key.strip()
    dotted_key_parts(key)
    raw_items = raw_values.split(",")
    if not raw_items:
        raise ValueError("parameters must include at least one value")
    values = [parse_scalar(item) for item in raw_items]
    return key, values


def set_dotted(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key_parts(dotted_key)
    target = data
    for key in keys[:-1]:
        child = target.setdefault(key, {})
        if not isinstance(child, dict):
            raise ValueError(f"cannot set nested key below non-table {key!r}")
        target = child
    target[keys[-1]] = value


def get_dotted(data: dict[str, Any], dotted_key: str) -> Any:
    target: Any = data
    for key in dotted_key_parts(dotted_key):
        if not isinstance(target, dict) or key not in target:
            return None
        target = target[key]
    return target


def toml_value(value: Any) -> str:
    if value is None:
        raise ValueError("TOML output cannot represent None sweep values")
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("TOML numeric values must be finite")
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    return json.dumps(str(value))


def write_toml(data: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    flat_items = [
        (key, value) for key, value in data.items() if not isinstance(value, dict)
    ]
    for key, value in flat_items:
        lines.append(f"{key} = {toml_value(value)}")
    for section, values in data.items():
        if not isinstance(values, dict):
            continue
        lines.append("")
        lines.append(f"[{section}]")
        for key, value in values.items():
            if isinstance(value, dict):
                nested_section = f"{section}.{key}"
                lines.append("")
                lines.append(f"[{nested_section}]")
                for nested_key, nested_value in value.items():
                    lines.append(f"{nested_key} = {toml_value(nested_value)}")
            else:
                lines.append(f"{key} = {toml_value(value)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def finite_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def section_value(metrics: dict[str, Any], section: str, key: str) -> Any:
    data = metrics.get(section)
    if not isinstance(data, dict):
        return None
    return data.get(key)


def nonnegative_metric(metrics: dict[str, Any], section: str, key: str) -> float | None:
    value = finite_number(section_value(metrics, section, key))
    return value if value is not None and value >= 0.0 else None


def nonnegative_count_metric(
    metrics: dict[str, Any], section: str, key: str
) -> int | None:
    value = nonnegative_metric(metrics, section, key)
    if value is None or not value.is_integer():
        return None
    return int(value)


def unit_interval_metric(
    metrics: dict[str, Any], section: str, key: str
) -> float | None:
    value = finite_number(section_value(metrics, section, key))
    return value if value is not None and 0.0 <= value <= 1.0 else None


def finite_metric(metrics: dict[str, Any], section: str, key: str) -> float | None:
    return finite_number(section_value(metrics, section, key))


def false_positives_per_frame(metrics: dict[str, Any]) -> float | None:
    false_positives = nonnegative_metric(metrics, "detections", "false_positives")
    frame_count = finite_number(section_value(metrics, "case", "frames"))
    if frame_count in (None, 0):
        frame_count = finite_number(section_value(metrics, "runtime", "frames"))
    if frame_count in (None, 0):
        frame_count = finite_number(section_value(metrics, "run", "n_images"))
    if false_positives is None or frame_count is None or frame_count <= 0:
        return None
    return float(false_positives / frame_count)


def compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def benchmark_summary_row(
    *,
    run_index: int,
    output_dir: Path,
    config: dict[str, Any],
    overrides: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_index": run_index,
        "output_dir": str(output_dir),
        "overrides": compact_json(overrides),
        "detection_threshold": get_dotted(config, "detection.threshold"),
        "detection_low_threshold": get_dotted(config, "detection.low_threshold"),
        "detection_precision": unit_interval_metric(metrics, "detections", "precision"),
        "detection_recall": unit_interval_metric(metrics, "detections", "recall"),
        "detection_f1": unit_interval_metric(metrics, "detections", "f1"),
        "false_positives_per_frame": false_positives_per_frame(metrics),
        "event_precision": unit_interval_metric(metrics, "events", "precision"),
        "event_recall": unit_interval_metric(metrics, "events", "recall"),
        "event_f1": unit_interval_metric(metrics, "events", "f1"),
        "filtered_event_precision": unit_interval_metric(
            metrics, "filtered_events", "precision"
        ),
        "filtered_event_recall": unit_interval_metric(
            metrics, "filtered_events", "recall"
        ),
        "filtered_event_f1": unit_interval_metric(metrics, "filtered_events", "f1"),
        "mean_track_length": nonnegative_metric(metrics, "tracks", "mean_track_length"),
        "median_track_length": nonnegative_metric(
            metrics, "tracks", "median_track_length"
        ),
        "single_frame_tracks": nonnegative_count_metric(
            metrics, "tracks", "single_frame_tracks"
        ),
        "single_frame_track_fraction": unit_interval_metric(
            metrics,
            "tracks",
            "single_frame_track_fraction",
        ),
        "filtered_mean_track_length": nonnegative_metric(
            metrics,
            "filtered_tracks",
            "mean_track_length",
        ),
        "filtered_median_track_length": nonnegative_metric(
            metrics,
            "filtered_tracks",
            "median_track_length",
        ),
        "filtered_single_frame_tracks": nonnegative_count_metric(
            metrics,
            "filtered_tracks",
            "single_frame_tracks",
        ),
        "filtered_single_frame_track_fraction": unit_interval_metric(
            metrics,
            "filtered_tracks",
            "single_frame_track_fraction",
        ),
        "track_fragmentation": nonnegative_metric(
            metrics, "events", "track_fragmentation"
        ),
        "filtered_track_fragmentation": nonnegative_metric(
            metrics,
            "filtered_events",
            "track_fragmentation",
        ),
        "fragmented_truth_events": nonnegative_count_metric(
            metrics, "events", "fragmented_truth_events"
        ),
        "mean_fragments_per_truth_event": nonnegative_metric(
            metrics,
            "events",
            "mean_fragments_per_truth_event",
        ),
        "birth_false_positive_rate": nonnegative_metric(
            metrics,
            "events",
            "birth_false_positive_rate",
        ),
        "missed_event_rate": unit_interval_metric(
            metrics, "events", "missed_event_rate"
        ),
        "velocity_y_error_px_per_frame": finite_metric(
            metrics,
            "velocity",
            "velocity_y_error_px_per_frame",
        ),
        "velocity_y_mean_abs_error_px_per_frame": nonnegative_metric(
            metrics,
            "velocity",
            "velocity_y_mean_abs_error_px_per_frame",
        ),
        "velocity_y_bias_px_per_frame": finite_metric(
            metrics,
            "velocity",
            "velocity_y_bias_px_per_frame",
        ),
        "velocity_y_error_std_px_per_frame": nonnegative_metric(
            metrics,
            "velocity",
            "velocity_y_error_std_px_per_frame",
        ),
        "truth_matched_velocity_y_error_px_per_frame": finite_metric(
            metrics,
            "velocity",
            "truth_matched_velocity_y_error_px_per_frame",
        ),
        "filtered_velocity_y_error_px_per_frame": finite_metric(
            metrics,
            "filtered_velocity",
            "velocity_y_error_px_per_frame",
        ),
        "filtered_velocity_y_mean_abs_error_px_per_frame": nonnegative_metric(
            metrics,
            "filtered_velocity",
            "velocity_y_mean_abs_error_px_per_frame",
        ),
        "filtered_velocity_y_bias_px_per_frame": finite_metric(
            metrics,
            "filtered_velocity",
            "velocity_y_bias_px_per_frame",
        ),
        "filtered_velocity_y_error_std_px_per_frame": nonnegative_metric(
            metrics,
            "filtered_velocity",
            "velocity_y_error_std_px_per_frame",
        ),
        "filtered_truth_matched_velocity_y_error_px_per_frame": finite_metric(
            metrics,
            "filtered_velocity",
            "truth_matched_velocity_y_error_px_per_frame",
        ),
        "phase_rmse_px": nonnegative_metric(metrics, "phase", "rmse_px"),
        "map_rmse_gray": nonnegative_metric(metrics, "belt_map", "rmse_gray"),
    }


def csv_value(value: Any) -> Any:
    return "" if value is None else value


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SWEEP_METRIC_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: csv_value(row.get(field)) for field in SWEEP_METRIC_FIELDS}
            )


def write_summary_json(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def markdown_value(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def markdown_link(path: Path, *, relative_to: Path) -> str:
    try:
        target = path.relative_to(relative_to)
    except ValueError:
        target = path
    return str(target).replace("\\", "/")


def sweep_froc_points(rows: list[dict[str, Any]]) -> list[tuple[float, float, str]]:
    """Return finite FROC points as FP/frame, recall, threshold-label tuples."""

    points: list[tuple[float, float, str]] = []
    for row in rows:
        false_positives_per_frame = finite_number(row.get("false_positives_per_frame"))
        recall = finite_number(row.get("detection_recall"))
        if false_positives_per_frame is None or recall is None:
            continue
        if false_positives_per_frame < 0.0 or not 0.0 <= recall <= 1.0:
            continue
        threshold = row.get("detection_threshold")
        label = "" if threshold in (None, "") else str(threshold)
        points.append((false_positives_per_frame, recall, label))
    return sorted(points, key=lambda item: (item[0], item[1]))


def write_froc_curve_svg(rows: list[dict[str, Any]], path: Path) -> None:
    """Write a lightweight SVG FROC curve from benchmark-sweep rows."""

    points = sweep_froc_points(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 900, 500
    left, right, top, bottom = 78, 36, 48, 72
    plot_width = width - left - right
    plot_height = height - top - bottom
    baseline = top + plot_height

    if points:
        x_max = max(1.0, max(point[0] for point in points))
    else:
        x_max = 1.0

    def sx(value: float) -> float:
        return left + value / x_max * plot_width

    def sy(value: float) -> float:
        return baseline - max(0.0, min(1.0, value)) * plot_height

    polyline = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y, _label in points)
    point_marks = "\n".join(
        f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="#1f77b4" />'
        for x, y, _label in points
    )
    threshold_labels = ""
    if len(points) <= 12:
        threshold_labels = "\n".join(
            f'<text x="{sx(x) + 6:.1f}" y="{sy(y) - 6:.1f}" font-size="11">t={escape(label)}</text>'
            for x, y, label in points
            if label
        )
    no_data = ""
    if not points:
        no_data = (
            f'<text x="{left + 18}" y="{top + 42}" font-size="16">'
            "No finite detection recall / false-positive values available"
            "</text>"
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white" />
  <text x="{left}" y="24" font-size="18" font-weight="bold">Detection FROC</text>
  <line x1="{left}" y1="{baseline}" x2="{left + plot_width}" y2="{baseline}" stroke="black" />
  <line x1="{left}" y1="{top}" x2="{left}" y2="{baseline}" stroke="black" />
  <text x="{left + plot_width / 2 - 110:.1f}" y="{height - 24}" font-size="14">false positives per frame</text>
  <text x="12" y="{top + 18}" font-size="14">recall</text>
  <text x="{left - 8}" y="{baseline + 18}" text-anchor="end" font-size="11">0</text>
  <text x="{left + plot_width}" y="{baseline + 18}" text-anchor="middle" font-size="11">{x_max:.3g}</text>
  <text x="{left - 8}" y="{top + 4}" text-anchor="end" font-size="11">1</text>
  <polyline fill="none" stroke="#1f77b4" stroke-width="2" points="{polyline}" />
  {point_marks}
  {threshold_labels}
  {no_data}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def write_summary_report(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    froc_plot: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BeltMap benchmark sweep",
        "",
        "Each row is one executed parameter-sweep run scored against synthetic truth.",
        "The table is intended for precision-recall, F1-threshold, FP-recall,",
        "fragmentation-threshold, and velocity-bias-threshold plots.",
        "",
    ]
    if froc_plot is not None:
        lines.extend(
            [
                "## Detection FROC",
                "",
                "This plot uses detection recall on the y-axis and false positives per frame on the x-axis. Each point is one executed sweep run, so this is the full rerun-based FROC view rather than a single operating-point F1 score.",
                "",
                f"![Detection FROC]({markdown_link(froc_plot, relative_to=path.parent)})",
                "",
            ]
        )
    lines.extend(
        [
            "| Run | Threshold | Precision | Recall | F1 | FP/frame | Event F1 | Single-frame tracks | Median track length | Track fragmentation | Velocity bias | Map RMSE |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    markdown_value(row.get("run_index")),
                    markdown_value(row.get("detection_threshold")),
                    markdown_value(row.get("detection_precision")),
                    markdown_value(row.get("detection_recall")),
                    markdown_value(row.get("detection_f1")),
                    markdown_value(row.get("false_positives_per_frame")),
                    markdown_value(row.get("event_f1")),
                    markdown_value(row.get("single_frame_tracks")),
                    markdown_value(row.get("median_track_length")),
                    markdown_value(row.get("track_fragmentation")),
                    markdown_value(row.get("velocity_y_bias_px_per_frame")),
                    markdown_value(row.get("map_rmse_gray")),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beltmap-sweep",
        description="Generate or execute BeltMap parameter sweep configurations.",
    )
    parser.add_argument(
        "--base-config", type=Path, required=True, help="Base TOML config."
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Dotted key and comma-separated values, e.g. detection.threshold=3.5,4.0.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/sweeps"),
        help="Directory for generated run configs and outputs.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run beltmap-apply and beltmap-validate for each generated config.",
    )
    parser.add_argument(
        "--benchmark-truth-path",
        type=Path,
        help="Synthetic truth JSON. When set, benchmark each run and write sweep metrics.",
    )
    parser.add_argument(
        "--benchmark-iou-threshold",
        type=parse_iou_threshold,
        default=0.25,
        help="IoU threshold for synthetic benchmark matching. Default: 0.25.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        help="Benchmark sweep CSV path. Default: OUTPUT_ROOT/sweep_metrics.csv.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Benchmark sweep JSON path. Default: OUTPUT_ROOT/sweep_metrics.json.",
    )
    parser.add_argument(
        "--summary-report",
        type=Path,
        help="Benchmark sweep Markdown report path. Default: OUTPUT_ROOT/sweep_report.md.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        base = tomllib.loads(args.base_config.read_text(encoding="utf-8"))
        params = [parse_param(item) for item in args.param]
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        parser.error(str(exc))
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    keys = [key for key, _values in params]
    value_grid = (
        itertools.product(*(values for _key, values in params)) if params else [()]
    )
    for run_index, values in enumerate(value_grid):
        config = json.loads(json.dumps(base))
        run_dir = args.output_root / f"run_{run_index:03d}"
        set_dotted(config, "paths.output_dir", str(run_dir))
        overrides = dict(zip(keys, values))
        for key, value in overrides.items():
            set_dotted(config, key, value)
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "beltmap.toml"
        try:
            write_toml(config, config_path)
        except ValueError as exc:
            parser.error(str(exc))
        manifest.append(
            {
                "run_index": run_index,
                "config": str(config_path),
                "output_dir": str(run_dir),
                "overrides": overrides,
            }
        )
        if args.execute:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "beltmap.cli.apply",
                    "--config",
                    str(config_path),
                ],
                check=True,
            )
            if shutil.which("beltmap-validate"):
                subprocess.run(
                    ["beltmap-validate", "--output-dir", str(run_dir)], check=True
                )
        if args.benchmark_truth_path is not None:
            artifacts = generate_benchmark_report(
                output_dir=run_dir,
                truth_path=args.benchmark_truth_path,
                iou_threshold=args.benchmark_iou_threshold,
            )
            metrics = read_json(artifacts.metrics)
            summary_rows.append(
                benchmark_summary_row(
                    run_index=run_index,
                    output_dir=run_dir,
                    config=config,
                    overrides=overrides,
                    metrics=metrics,
                )
            )
    manifest_path = args.output_root / "sweep_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if summary_rows:
        summary_csv = args.summary_csv or args.output_root / "sweep_metrics.csv"
        summary_json = args.summary_json or args.output_root / "sweep_metrics.json"
        summary_report = args.summary_report or args.output_root / "sweep_report.md"
        froc_plot = summary_report.with_name("sweep_froc_curve.svg")
        write_summary_csv(summary_rows, summary_csv)
        write_summary_json(summary_rows, summary_json)
        write_froc_curve_svg(summary_rows, froc_plot)
        write_summary_report(summary_rows, summary_report, froc_plot=froc_plot)
    print(manifest_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
