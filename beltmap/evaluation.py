from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SUMMARY_FIELDS = [
    "run",
    "output_dir",
    "missing_files",
    "n_images",
    "n_phase_estimates",
    "n_detections",
    "n_tracks",
    "n_velocity_estimates",
    "phase_correction_abs_median_px",
    "phase_correction_abs_q95_px",
    "registration_score_median",
    "registration_score_q95",
    "detections_per_frame_mean",
    "detections_per_frame_median",
    "detections_per_frame_max",
    "velocity_ratio_y_median",
    "velocity_ratio_y_iqr",
    "velocity_ratio_y_outlier_fraction",
    "belt_map_observed_fraction",
    "belt_map_masked_fraction",
    "belt_map_contributed_fraction",
]

STANDARD_OUTPUT_FILES = [
    "metadata.json",
    "config_resolved.json",
    "progress.jsonl",
    "phase_estimates.csv",
    "detections.csv",
    "detections_per_frame.csv",
    "velocities.csv",
    "belt_map.png",
]


@dataclass(frozen=True)
class RunSpec:
    name: str
    output_dir: Path


@dataclass(frozen=True)
class EvaluationArtifacts:
    json_path: Path
    csv_path: Path
    markdown_path: Path


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


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
            row = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
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


def finite_nonnegative_values(
    rows: Iterable[dict[str, Any]], field: str
) -> list[float]:
    values = finite_values(rows, field)
    return [value for value in values if value >= 0.0]


def scalar_from_sources(
    metadata: dict[str, Any],
    *keys: str,
    fallback: int | float | None = None,
) -> int | float | None:
    """Return an integer metadata count, falling back when metadata is malformed."""

    for key in keys:
        raw_value = metadata.get(key)
        if isinstance(raw_value, bool):
            continue
        value = finite_float(metadata.get(key))
        if value is None:
            continue
        if float(value).is_integer() and value >= 0:
            return int(value)
    return fallback


def row_count_if_present(path: Path, rows: list[Any]) -> int | None:
    return len(rows) if path.is_file() else None


def percentile(values: Iterable[float], q: float) -> float | None:
    q_value = finite_float(q)
    if q_value is None or not 0.0 <= q_value <= 100.0:
        raise ValueError("percentile q must be finite and in [0, 100]")
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.percentile(arr, q_value))


def mean(values: Iterable[float]) -> float | None:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def fraction(numerator: Any, denominator: Any) -> float | None:
    num = finite_float(numerator)
    den = finite_float(denominator)
    if num is None or den is None or den <= 0 or num < 0 or num > den:
        return None
    return float(num / den)


def final_stage_progress(
    progress_rows: Iterable[dict[str, Any]],
    stage: str,
) -> dict[str, Any]:
    matches = [row for row in progress_rows if row.get("stage") == stage]
    return matches[-1] if matches else {}


def format_markdown_value(value: Any, *, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if not np.isfinite(value):
            return "n/a"
        if abs(value) >= 1000 or (0 < abs(value) < 1e-3):
            return f"{value:.{digits}g}"
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def missing_standard_files(output_dir: Path) -> list[str]:
    return [name for name in STANDARD_OUTPUT_FILES if not (output_dir / name).is_file()]


def summary_value(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return summary_value(value.item())
    return value


def summary_row(summary: dict[str, Any]) -> dict[str, Any]:
    return {field: summary_value(summary.get(field)) for field in SUMMARY_FIELDS}


def summarize_output_dir(run: RunSpec) -> dict[str, Any]:
    """Summarize one BeltMap output directory into comparable scalar metrics."""

    output_dir = run.output_dir
    metadata = read_json(output_dir / "metadata.json")
    progress_rows = read_progress_jsonl(output_dir / "progress.jsonl")
    phase_path = output_dir / "phase_estimates.csv"
    detection_count_path = output_dir / "detections_per_frame.csv"
    detection_path = output_dir / "detections.csv"
    velocity_path = output_dir / "velocities.csv"
    phase_rows = read_csv_rows(phase_path)
    detection_rows = read_csv_rows(detection_count_path)
    detections = read_csv_rows(detection_path)
    velocity_rows = read_csv_rows(velocity_path)

    corrections = finite_values(phase_rows, "correction_px")
    abs_corrections = [abs(value) for value in corrections]
    scores = finite_values(phase_rows, "score")
    detections_per_frame = finite_nonnegative_values(detection_rows, "n_detections")
    velocity_ratios = finite_values(velocity_rows, "velocity_ratio_y")

    belt_progress = final_stage_progress(progress_rows, "belt_map")
    observed_pixels = belt_progress.get("observed_pixels")
    total_pixels = belt_progress.get("total_pixels")
    masked_pixels = belt_progress.get("masked_pixels")
    contributed_pixels = belt_progress.get("contributed_pixels")

    q25_ratio = percentile(velocity_ratios, 25)
    q75_ratio = percentile(velocity_ratios, 75)
    ratio_iqr = (
        None if q25_ratio is None or q75_ratio is None else q75_ratio - q25_ratio
    )
    if velocity_ratios:
        ratio_arr = np.asarray(velocity_ratios, dtype=np.float64)
        ratio_outlier_fraction = float(np.mean((ratio_arr < -0.1) | (ratio_arr > 1.1)))
    else:
        ratio_outlier_fraction = None

    summary: dict[str, Any] = {
        "run": run.name,
        "output_dir": str(output_dir),
        "missing_files": ",".join(missing_standard_files(output_dir)),
        "n_images": scalar_from_sources(
            metadata,
            "n_images",
            fallback=row_count_if_present(detection_count_path, detection_rows),
        ),
        "n_phase_estimates": scalar_from_sources(
            metadata,
            "n_phase_estimates",
            fallback=row_count_if_present(phase_path, phase_rows),
        ),
        "n_detections": scalar_from_sources(
            metadata,
            "n_detections",
            fallback=row_count_if_present(detection_path, detections),
        ),
        "n_tracks": scalar_from_sources(metadata, "n_tracks"),
        "n_velocity_estimates": scalar_from_sources(
            metadata,
            "n_velocity_estimates",
            fallback=row_count_if_present(velocity_path, velocity_rows),
        ),
        "phase_correction_abs_median_px": percentile(abs_corrections, 50),
        "phase_correction_abs_q95_px": percentile(abs_corrections, 95),
        "registration_score_median": percentile(scores, 50),
        "registration_score_q95": percentile(scores, 95),
        "detections_per_frame_mean": mean(detections_per_frame),
        "detections_per_frame_median": percentile(detections_per_frame, 50),
        "detections_per_frame_max": percentile(detections_per_frame, 100),
        "velocity_ratio_y_median": percentile(velocity_ratios, 50),
        "velocity_ratio_y_iqr": ratio_iqr,
        "velocity_ratio_y_outlier_fraction": ratio_outlier_fraction,
        "belt_map_observed_fraction": fraction(observed_pixels, total_pixels),
        "belt_map_masked_fraction": fraction(masked_pixels, total_pixels),
        "belt_map_contributed_fraction": fraction(contributed_pixels, total_pixels),
    }
    return summary


def summarize_runs(runs: Iterable[RunSpec]) -> list[dict[str, Any]]:
    return [summarize_output_dir(run) for run in runs]


def write_json(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "summary_fields": SUMMARY_FIELDS,
        "runs": [summary_row(summary) for summary in summaries],
    }
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary_row(summary))


def build_markdown(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# BeltMap evaluation summary",
        "",
        "This report compares completed `beltmap-apply` output directories. It is intended for ablations such as baseline vs. phase feedback, static background/noise learning, threshold settings, and tracker settings.",
        "",
        "## Run comparison",
        "",
        "| Run | Images | Detections | Tracks | Median abs correction px | Median score | Mean detections/frame | Median velocity ratio | Observed map | Missing files |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for summary in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{summary['run']}`",
                    format_markdown_value(summary.get("n_images")),
                    format_markdown_value(summary.get("n_detections")),
                    format_markdown_value(summary.get("n_tracks")),
                    format_markdown_value(
                        summary.get("phase_correction_abs_median_px")
                    ),
                    format_markdown_value(summary.get("registration_score_median")),
                    format_markdown_value(summary.get("detections_per_frame_mean")),
                    format_markdown_value(summary.get("velocity_ratio_y_median")),
                    format_markdown_value(summary.get("belt_map_observed_fraction")),
                    summary.get("missing_files") or "none",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpreting ablations",
            "",
            "- Lower absolute phase corrections and higher registration scores usually indicate a cleaner phase model and belt map.",
            "- Large detection-count changes should be checked against residual previews before interpreting them as improvements.",
            "- Velocity-ratio outliers outside the plausible physical range are a useful proxy for fragmented or mismatched tracks.",
            "- Low observed-map or contributed-map fractions indicate that map quality may be limited by insufficient belt coverage or excessive masking.",
            "",
        ]
    )
    return "\n".join(lines)


def write_markdown(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_markdown(summaries), encoding="utf-8")


def write_evaluation(
    runs: Iterable[RunSpec],
    *,
    output_dir: Path,
    json_path: Path | None = None,
    csv_path: Path | None = None,
    markdown_path: Path | None = None,
) -> EvaluationArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = summarize_runs(runs)

    resolved_json_path = json_path or (output_dir / "evaluation_summary.json")
    resolved_csv_path = csv_path or (output_dir / "evaluation_summary.csv")
    resolved_markdown_path = markdown_path or (output_dir / "evaluation_summary.md")

    write_json(resolved_json_path, summaries)
    write_csv(resolved_csv_path, summaries)
    write_markdown(resolved_markdown_path, summaries)

    return EvaluationArtifacts(
        json_path=resolved_json_path,
        csv_path=resolved_csv_path,
        markdown_path=resolved_markdown_path,
    )
