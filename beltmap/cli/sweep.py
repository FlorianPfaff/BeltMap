from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import shutil
import subprocess
import sys
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
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if text.lower() in {"none", "null"}:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def parse_param(value: str) -> tuple[str, list[Any]]:
    if "=" not in value:
        raise ValueError("parameters must be KEY=VALUE1,VALUE2")
    key, raw_values = value.split("=", 1)
    values = [parse_scalar(item) for item in raw_values.split(",")]
    return key.strip(), values


def set_dotted(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    target = data
    for key in keys[:-1]:
        child = target.setdefault(key, {})
        if not isinstance(child, dict):
            raise ValueError(f"cannot set nested key below non-table {key!r}")
        target = child
    target[keys[-1]] = value


def get_dotted(data: dict[str, Any], dotted_key: str) -> Any:
    target: Any = data
    for key in dotted_key.split("."):
        if not isinstance(target, dict) or key not in target:
            return None
        target = target[key]
    return target


def toml_value(value: Any) -> str:
    if value is None:
        return '""'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    return json.dumps(str(value))


def write_toml(data: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    flat_items = [(key, value) for key, value in data.items() if not isinstance(value, dict)]
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


def false_positives_per_frame(metrics: dict[str, Any]) -> float | None:
    false_positives = finite_number(section_value(metrics, "detections", "false_positives"))
    frame_count = finite_number(section_value(metrics, "case", "frames"))
    if frame_count in (None, 0):
        frame_count = finite_number(section_value(metrics, "runtime", "frames"))
    if frame_count in (None, 0):
        frame_count = finite_number(section_value(metrics, "run", "n_images"))
    if false_positives is None or frame_count in (None, 0):
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
        "detection_precision": section_value(metrics, "detections", "precision"),
        "detection_recall": section_value(metrics, "detections", "recall"),
        "detection_f1": section_value(metrics, "detections", "f1"),
        "false_positives_per_frame": false_positives_per_frame(metrics),
        "event_precision": section_value(metrics, "events", "precision"),
        "event_recall": section_value(metrics, "events", "recall"),
        "event_f1": section_value(metrics, "events", "f1"),
        "filtered_event_precision": section_value(metrics, "filtered_events", "precision"),
        "filtered_event_recall": section_value(metrics, "filtered_events", "recall"),
        "filtered_event_f1": section_value(metrics, "filtered_events", "f1"),
        "mean_track_length": section_value(metrics, "tracks", "mean_track_length"),
        "median_track_length": section_value(metrics, "tracks", "median_track_length"),
        "single_frame_tracks": section_value(metrics, "tracks", "single_frame_tracks"),
        "single_frame_track_fraction": section_value(
            metrics,
            "tracks",
            "single_frame_track_fraction",
        ),
        "filtered_mean_track_length": section_value(
            metrics,
            "filtered_tracks",
            "mean_track_length",
        ),
        "filtered_median_track_length": section_value(
            metrics,
            "filtered_tracks",
            "median_track_length",
        ),
        "filtered_single_frame_tracks": section_value(
            metrics,
            "filtered_tracks",
            "single_frame_tracks",
        ),
        "filtered_single_frame_track_fraction": section_value(
            metrics,
            "filtered_tracks",
            "single_frame_track_fraction",
        ),
        "track_fragmentation": section_value(metrics, "events", "track_fragmentation"),
        "filtered_track_fragmentation": section_value(
            metrics,
            "filtered_events",
            "track_fragmentation",
        ),
        "fragmented_truth_events": section_value(metrics, "events", "fragmented_truth_events"),
        "mean_fragments_per_truth_event": section_value(
            metrics,
            "events",
            "mean_fragments_per_truth_event",
        ),
        "birth_false_positive_rate": section_value(
            metrics,
            "events",
            "birth_false_positive_rate",
        ),
        "missed_event_rate": section_value(metrics, "events", "missed_event_rate"),
        "velocity_y_error_px_per_frame": section_value(
            metrics,
            "velocity",
            "velocity_y_error_px_per_frame",
        ),
        "velocity_y_mean_abs_error_px_per_frame": section_value(
            metrics,
            "velocity",
            "velocity_y_mean_abs_error_px_per_frame",
        ),
        "velocity_y_bias_px_per_frame": section_value(
            metrics,
            "velocity",
            "velocity_y_bias_px_per_frame",
        ),
        "velocity_y_error_std_px_per_frame": section_value(
            metrics,
            "velocity",
            "velocity_y_error_std_px_per_frame",
        ),
        "truth_matched_velocity_y_error_px_per_frame": section_value(
            metrics,
            "velocity",
            "truth_matched_velocity_y_error_px_per_frame",
        ),
        "filtered_velocity_y_error_px_per_frame": section_value(
            metrics,
            "filtered_velocity",
            "velocity_y_error_px_per_frame",
        ),
        "filtered_velocity_y_mean_abs_error_px_per_frame": section_value(
            metrics,
            "filtered_velocity",
            "velocity_y_mean_abs_error_px_per_frame",
        ),
        "filtered_velocity_y_bias_px_per_frame": section_value(
            metrics,
            "filtered_velocity",
            "velocity_y_bias_px_per_frame",
        ),
        "filtered_velocity_y_error_std_px_per_frame": section_value(
            metrics,
            "filtered_velocity",
            "velocity_y_error_std_px_per_frame",
        ),
        "filtered_truth_matched_velocity_y_error_px_per_frame": section_value(
            metrics,
            "filtered_velocity",
            "truth_matched_velocity_y_error_px_per_frame",
        ),
        "phase_rmse_px": section_value(metrics, "phase", "rmse_px"),
        "map_rmse_gray": section_value(metrics, "belt_map", "rmse_gray"),
    }


def csv_value(value: Any) -> Any:
    return "" if value is None else value


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SWEEP_METRIC_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in SWEEP_METRIC_FIELDS})


def write_summary_json(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def markdown_value(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def write_summary_report(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BeltMap benchmark sweep",
        "",
        "Each row is one executed parameter-sweep run scored against synthetic truth.",
        "The table is intended for precision-recall, F1-threshold, FP-recall,",
        "fragmentation-threshold, and velocity-bias-threshold plots.",
        "",
        "| Run | Threshold | Precision | Recall | F1 | FP/frame | Event F1 | Single-frame tracks | Median track length | Track fragmentation | Velocity bias | Map RMSE |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
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
    parser.add_argument("--base-config", type=Path, required=True, help="Base TOML config.")
    parser.add_argument("--param", action="append", default=[], help="Dotted key and comma-separated values, e.g. detection.threshold=3.5,4.0.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/sweeps"), help="Directory for generated run configs and outputs.")
    parser.add_argument("--execute", action="store_true", help="Run beltmap-apply and beltmap-validate for each generated config.")
    parser.add_argument("--benchmark-truth-path", type=Path, help="Synthetic truth JSON. When set, benchmark each run and write sweep metrics.")
    parser.add_argument("--benchmark-iou-threshold", type=float, default=0.25, help="IoU threshold for synthetic benchmark matching. Default: 0.25.")
    parser.add_argument("--summary-csv", type=Path, help="Benchmark sweep CSV path. Default: OUTPUT_ROOT/sweep_metrics.csv.")
    parser.add_argument("--summary-json", type=Path, help="Benchmark sweep JSON path. Default: OUTPUT_ROOT/sweep_metrics.json.")
    parser.add_argument("--summary-report", type=Path, help="Benchmark sweep Markdown report path. Default: OUTPUT_ROOT/sweep_report.md.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base = tomllib.loads(args.base_config.read_text(encoding="utf-8"))
    params = [parse_param(item) for item in args.param]
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    keys = [key for key, _values in params]
    value_grid = itertools.product(*(values for _key, values in params)) if params else [()]
    for run_index, values in enumerate(value_grid):
        config = json.loads(json.dumps(base))
        run_dir = args.output_root / f"run_{run_index:03d}"
        set_dotted(config, "paths.output_dir", str(run_dir))
        overrides = dict(zip(keys, values))
        for key, value in overrides.items():
            set_dotted(config, key, value)
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "beltmap.toml"
        write_toml(config, config_path)
        manifest.append({"run_index": run_index, "config": str(config_path), "output_dir": str(run_dir), "overrides": overrides})
        if args.execute:
            subprocess.run([sys.executable, "-m", "beltmap.cli.apply", "--config", str(config_path)], check=True)
            if shutil.which("beltmap-validate"):
                subprocess.run(["beltmap-validate", "--output-dir", str(run_dir)], check=True)
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
        write_summary_csv(summary_rows, args.summary_csv or args.output_root / "sweep_metrics.csv")
        write_summary_json(summary_rows, args.summary_json or args.output_root / "sweep_metrics.json")
        write_summary_report(summary_rows, args.summary_report or args.output_root / "sweep_report.md")
    print(manifest_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
