from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw

from .benchmark import detection_metrics, source_frame_index
from .bootstrap_ci import (
    BOOTSTRAP_SUMMARY_FIELDS,
    bootstrap_run_summary,
    empty_bootstrap_metrics,
)
from .track_diagnostics import (
    DEFAULT_NEAR_THRESHOLD_MARGIN,
    DEFAULT_SMALL_AREA_THRESHOLD_PX,
    accepted_track_quality_summary,
    detection_quality_summary,
)
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
    "small_detection_share_area_lt_50",
    "near_threshold_peak_share",
    "labeled_detection_available",
    "labeled_scored_frames",
    "labeled_detection_iou_threshold",
    "labeled_truth_boxes",
    "labeled_predicted_boxes",
    "labeled_true_positives",
    "labeled_false_positives",
    "labeled_false_negatives",
    "labeled_false_positives_per_frame",
    "labeled_empty_scored_frames",
    "labeled_empty_frame_false_positives",
    "labeled_empty_frame_fp_per_frame",
    "labeled_precision",
    "labeled_recall",
    "labeled_f1",
    "labeled_froc_available",
    "labeled_froc_score_field",
    "labeled_froc_points",
    "labeled_froc_auc_fp_per_frame_le_1",
    "labeled_froc_recall_at_0_1_fp_per_frame",
    "labeled_froc_recall_at_0_5_fp_per_frame",
    "labeled_froc_recall_at_1_0_fp_per_frame",
    "labeled_mean_matched_iou",
    "labeled_mean_centroid_error_px",
    *BOOTSTRAP_SUMMARY_FIELDS,
    "small_component_share_area_le_8",
    "small_accepted_tracks_lt_50",
    "long_small_accepted_tracks_ge_5",
    "long_small_accepted_tracks_ge_10",
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

REQUIRED_CSV_COLUMNS = {
    "detections.csv": {
        "frame_index",
        "y",
        "x",
        "area_px",
        "bbox_top",
        "bbox_left",
        "bbox_bottom",
        "bbox_right",
    },
    "detections_per_frame.csv": {"frame_index", "n_detections"},
    "velocities.csv": {"n_detections", "velocity_ratio_y"},
    "filtered_velocities.csv": {"n_detections", "velocity_ratio_y"},
    "filtered_tracks.csv": {
        "track_id",
        "frame_index",
        "y",
        "x",
        "bbox_top",
        "bbox_left",
        "bbox_bottom",
        "bbox_right",
    },
}

TRUTH_CONTAINER_KEYS = ("particles", "annotations", "labels", "detections")
TRUTH_FRAME_KEYS = ("frame", "image_index")
TRUTH_FRAME_SET_KEYS = ("scored_frames", "frames", "labeled_frames")
TRUTH_BOX_FIELD_SETS = (
    ("bbox_top", "bbox_left", "bbox_bottom", "bbox_right"),
    ("top", "left", "bottom", "right"),
    ("y_min", "x_min", "y_max", "x_max"),
    ("y1", "x1", "y2", "x2"),
)
TRUTH_EVENT_ID_KEYS = ("event_id", "particle_id", "track_id", "id")

LABELED_METRIC_FIELDS = (
    "precision",
    "recall",
    "f1",
    "mean_matched_iou",
    "mean_centroid_error_px",
)

FROC_SCORE_FIELDS = (
    "peak_signal",
    "mean_signal",
    "score",
    "confidence",
)
FROC_FP_PER_FRAME_LIMITS = (0.1, 0.5, 1.0)
FROC_AUC_FP_PER_FRAME_LIMIT = 1.0


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
    fixed_preview_paths: dict[int, Path]
    raw_preview_paths: dict[int, Path]
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


def find_named_preview_paths(output_dir: Path, prefix: str) -> dict[int, Path]:
    """Return saved preview images keyed by frame index for ``<prefix>_frame_*.png``."""

    result: dict[int, Path] = {}
    marker = f"{prefix}_frame_"
    for path in sorted(output_dir.glob(f"{prefix}_frame_*.png")):
        stem = path.stem
        if not stem.startswith(marker):
            continue
        try:
            frame_index = int(stem[len(marker) :])
        except ValueError:
            continue
        result[frame_index] = path
    return result


def validate_csv_columns(
    path: Path,
    fieldnames: list[str] | None,
    required_columns: set[str] | None,
) -> None:
    if required_columns is None:
        return
    available = set(fieldnames or [])
    missing = sorted(required_columns - available)
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"{path} is missing required column(s): {missing_text}")


def read_csv_rows(
    path: Path,
    *,
    required_columns: set[str] | None = None,
) -> list[dict[str, str]]:
    """Read CSV rows, returning an empty list when the file is absent."""

    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        validate_csv_columns(path, reader.fieldnames, required_columns)
        return list(reader)


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


def finite_int(value: Any) -> int | None:
    """Parse an integer value through finite-float handling."""

    parsed = finite_float(value)
    return None if parsed is None else int(parsed)


def nonempty_value(row: dict[str, Any], key: str) -> Any | None:
    """Return a non-empty row value when present."""

    value = row.get(key)
    return None if value is None or str(value).strip() == "" else value


def finite_values(rows: Iterable[dict[str, Any]], field: str) -> list[float]:
    """Collect finite values from a CSV-like row sequence."""

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
    """Collect aligned x/y values, falling back to row index for missing x."""

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


def format_bootstrap_interval(row: dict[str, Any], field: str) -> str:
    """Format one bootstrap median and confidence interval cell."""

    median = finite_float(row.get(f"{field}_bootstrap_median"))
    low = finite_float(row.get(f"{field}_ci_low"))
    high = finite_float(row.get(f"{field}_ci_high"))
    if median is None or low is None or high is None:
        return "n/a"
    return f"{format_value(median)} [{format_value(low)}, {format_value(high)}]"


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


def row_frame_index(row: dict[str, Any]) -> int | None:
    """Infer a frame index from a label or detection row."""

    frame_index = source_frame_index(row)
    if frame_index is not None:
        return frame_index
    for key in TRUTH_FRAME_KEYS:
        frame_index = finite_int(nonempty_value(row, key))
        if frame_index is not None:
            return frame_index
    return None


def parse_frame_set(value: Any) -> set[int]:
    """Parse a JSON scalar/list of scored frame indices."""

    if value is None:
        return set()
    if isinstance(value, str):
        values: Iterable[Any] = [part.strip() for part in value.split(",")]
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, dict)):
        values = value
    else:
        values = [value]
    frames: set[int] = set()
    for item in values:
        frame_index = finite_int(item)
        if frame_index is not None:
            frames.add(frame_index)
    return frames


def label_rows_from_json(data: Any) -> tuple[list[dict[str, Any]], set[int]]:
    """Extract label rows and optional scored-frame indices from JSON data."""

    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)], set()
    if not isinstance(data, dict):
        raise ValueError("truth JSON must be an object or a list of objects")

    scored_frames: set[int] = set()
    for key in TRUTH_FRAME_SET_KEYS:
        scored_frames.update(parse_frame_set(data.get(key)))

    for key in TRUTH_CONTAINER_KEYS:
        rows = data.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)], scored_frames
    return [data], scored_frames


def truth_particle_from_label_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Convert one real-data label row to the benchmark truth-box schema."""

    frame_index = row_frame_index(row)
    if frame_index is None:
        return None

    for field_set in TRUTH_BOX_FIELD_SETS:
        values = [finite_float(nonempty_value(row, field)) for field in field_set]
        if any(value is None for value in values):
            continue
        top, left, bottom, right = values
        assert top is not None and left is not None
        assert bottom is not None and right is not None
        if bottom <= top or right <= left:
            return None
        particle: dict[str, Any] = {
            "frame_index": frame_index,
            "top": float(top),
            "left": float(left),
            "bottom": float(bottom),
            "right": float(right),
        }
        for key in TRUTH_EVENT_ID_KEYS:
            value = nonempty_value(row, key)
            if value is not None:
                particle[key] = value
                break
        return particle
    return None


def load_labeled_detection_truth(path: Path) -> dict[str, Any]:
    """Load manually labeled detection boxes from CSV or JSON.

    The returned object uses the same ``particles`` list consumed by the
    synthetic benchmark's detection matcher, but labels are deliberately scoped
    to ``scored_frames`` so sparse real-data annotations do not turn every
    unlabeled frame into a false positive.
    """

    if not path.is_file():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".csv":
        rows: list[dict[str, Any]] = list(read_csv_rows(path))
        scored_frames: set[int] = set()
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows, scored_frames = label_rows_from_json(data)
    else:
        raise ValueError("truth labels must be a CSV or JSON file")

    particles: list[dict[str, Any]] = []
    skipped_rows = 0
    for row in rows:
        frame_index = row_frame_index(row)
        if frame_index is not None:
            scored_frames.add(frame_index)
        particle = truth_particle_from_label_row(row)
        if particle is None:
            skipped_rows += 1
            continue
        particles.append(particle)

    if rows and not particles and not scored_frames:
        raise ValueError(
            "truth labels did not contain any usable frame indices or bounding boxes"
        )

    return {
        "particles": particles,
        "scored_frames": sorted(scored_frames),
        "source": str(path),
        "label_rows": len(rows),
        "skipped_label_rows": skipped_rows,
    }


def truth_frame_indices(truth: dict[str, Any]) -> set[int]:
    """Return the set of frames covered by a sparse labeled target."""

    frames = parse_frame_set(truth.get("scored_frames"))
    for particle in truth.get("particles", []):
        if isinstance(particle, dict):
            frame_index = row_frame_index(particle)
            if frame_index is not None:
                frames.add(frame_index)
    return frames


def truth_particle_frame_indices(truth: dict[str, Any]) -> set[int]:
    """Return frames that contain at least one labeled particle box."""

    frames: set[int] = set()
    for particle in truth.get("particles", []):
        if isinstance(particle, dict):
            frame_index = row_frame_index(particle)
            if frame_index is not None:
                frames.add(frame_index)
    return frames


def empty_labeled_frame_indices(truth: dict[str, Any]) -> set[int]:
    """Return scored frames that have no labeled particle boxes.

    Blank CSV rows with only ``frame_index`` therefore become explicit negative
    controls. Detections on these frames are counted as empty-frame false
    positives in addition to the standard matched-box metrics.
    """

    return truth_frame_indices(truth) - truth_particle_frame_indices(truth)


def restrict_detection_rows_to_frames(
    rows: list[dict[str, str]],
    frame_indices: set[int],
) -> list[dict[str, str]]:
    """Keep detections only on frames that were manually scored."""

    if not frame_indices:
        return []
    return [row for row in rows if row_frame_index(row) in frame_indices]


def detection_score_field(
    rows: Iterable[dict[str, Any]],
    *,
    score_fields: tuple[str, ...] = FROC_SCORE_FIELDS,
) -> str | None:
    """Return the first finite per-detection score field available in rows."""

    buffered_rows = list(rows)
    for field in score_fields:
        if any(finite_float(row.get(field)) is not None for row in buffered_rows):
            return field
    return None


def froc_frame_denominator(
    detection_rows: list[dict[str, str]],
    truth: dict[str, Any],
    *,
    scored_frames: set[int] | None = None,
) -> int:
    """Return the evaluated-frame denominator for false positives per frame."""

    if scored_frames:
        return len(scored_frames)
    frames = truth_frame_indices(truth)
    for row in detection_rows:
        frame_index = row_frame_index(row)
        if frame_index is not None:
            frames.add(frame_index)
    return len(frames)


def froc_point_from_metrics(
    metrics: dict[str, Any],
    *,
    score_threshold: float | None,
    evaluated_frames: int,
) -> dict[str, Any]:
    """Convert labeled detection metrics at one score threshold to a FROC point."""

    false_positives = int(metrics.get("false_positives") or 0)
    recall = finite_float(metrics.get("recall"))
    precision = finite_float(metrics.get("precision"))
    return {
        "score_threshold": score_threshold,
        "false_positives_per_frame": (
            false_positives / evaluated_frames if evaluated_frames > 0 else None
        ),
        "recall": recall,
        "precision": precision,
        "true_positives": metrics.get("true_positives"),
        "false_positives": false_positives,
        "false_negatives": metrics.get("false_negatives"),
        "predicted_boxes": metrics.get("predicted_boxes"),
    }


def monotone_froc_points(points: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """Return a monotone recall envelope sorted by false positives per frame."""

    best_by_fp: dict[float, float] = {}
    for point in points:
        false_positives_per_frame = finite_float(point.get("false_positives_per_frame"))
        recall = finite_float(point.get("recall"))
        if false_positives_per_frame is None or recall is None:
            continue
        best_by_fp[false_positives_per_frame] = max(
            recall,
            best_by_fp.get(false_positives_per_frame, 0.0),
        )

    envelope: list[tuple[float, float]] = []
    best_recall = 0.0
    for false_positives_per_frame, recall in sorted(best_by_fp.items()):
        best_recall = max(best_recall, recall)
        envelope.append((false_positives_per_frame, best_recall))
    return envelope


def froc_recall_at_fp_per_frame(
    points: list[dict[str, Any]],
    *,
    max_fp_per_frame: float,
) -> float | None:
    """Return best recall attainable at or below a false-positive budget."""

    envelope = monotone_froc_points(points)
    if not envelope:
        return None
    feasible = [recall for fp_per_frame, recall in envelope if fp_per_frame <= max_fp_per_frame]
    return max(feasible) if feasible else 0.0


def froc_auc_up_to_fp_per_frame(
    points: list[dict[str, Any]],
    *,
    max_fp_per_frame: float,
) -> float | None:
    """Return normalized trapezoidal FROC AUC up to a fixed FP/frame budget."""

    if max_fp_per_frame <= 0:
        return None
    envelope = monotone_froc_points(points)
    if not envelope:
        return None

    clipped: list[tuple[float, float]] = [(0.0, 0.0)]
    for false_positives_per_frame, recall in envelope:
        if false_positives_per_frame < 0:
            continue
        if false_positives_per_frame > max_fp_per_frame:
            break
        if clipped[-1][0] == false_positives_per_frame:
            clipped[-1] = (
                false_positives_per_frame,
                max(clipped[-1][1], recall),
            )
        else:
            clipped.append((false_positives_per_frame, recall))
    if clipped[-1][0] < max_fp_per_frame:
        clipped.append((max_fp_per_frame, clipped[-1][1]))

    area = 0.0
    for (x0, y0), (x1, y1) in zip(clipped, clipped[1:]):
        area += 0.5 * (y0 + y1) * (x1 - x0)
    return float(area / max_fp_per_frame)


def detection_froc_curve(
    detection_rows: list[dict[str, str]],
    truth: dict[str, Any],
    *,
    scored_frames: set[int] | None = None,
    iou_threshold: float = 0.25,
    score_fields: tuple[str, ...] = FROC_SCORE_FIELDS,
) -> dict[str, Any]:
    """Compute a detection FROC curve by sweeping a per-detection score.

    The curve is limited to detections already emitted by a run. It is therefore
    a within-run confidence sweep, not a replacement for rerunning a lower
    detector threshold when sub-threshold candidates were never written.
    """

    evaluated_frames = froc_frame_denominator(
        detection_rows,
        truth,
        scored_frames=scored_frames,
    )
    if evaluated_frames == 0:
        return {
            "available": False,
            "reason": "no evaluated frames for FROC denominator",
            "score_field": None,
            "points": [],
            "point_count": 0,
        }

    score_field = detection_score_field(detection_rows, score_fields=score_fields)
    empty_metrics = detection_metrics([], truth, iou_threshold=iou_threshold)
    truth_boxes = int(empty_metrics.get("truth_boxes") or 0)
    points = [
        froc_point_from_metrics(
            empty_metrics,
            score_threshold=None,
            evaluated_frames=evaluated_frames,
        )
    ]

    skipped_score_rows = 0
    thresholds: list[float] = []
    if score_field is not None:
        for row in detection_rows:
            score = finite_float(row.get(score_field))
            if score is None:
                skipped_score_rows += 1
                continue
            thresholds.append(score)
        for threshold in sorted(set(thresholds), reverse=True):
            kept_rows = [
                row
                for row in detection_rows
                if (score := finite_float(row.get(score_field))) is not None
                and score >= threshold
            ]
            metrics = detection_metrics(kept_rows, truth, iou_threshold=iou_threshold)
            points.append(
                froc_point_from_metrics(
                    metrics,
                    score_threshold=threshold,
                    evaluated_frames=evaluated_frames,
                )
            )
    elif detection_rows:
        metrics = detection_metrics(detection_rows, truth, iou_threshold=iou_threshold)
        points.append(
            froc_point_from_metrics(
                metrics,
                score_threshold=None,
                evaluated_frames=evaluated_frames,
            )
        )

    recall_limits = {
        limit: froc_recall_at_fp_per_frame(points, max_fp_per_frame=limit)
        for limit in FROC_FP_PER_FRAME_LIMITS
    }
    available = truth_boxes > 0 and (score_field is not None or not detection_rows)
    reason = None
    if truth_boxes == 0:
        reason = "no truth boxes on scored frames"
    elif score_field is None and detection_rows:
        reason = "no finite per-detection score field available for threshold sweep"

    return {
        "available": available,
        "reason": reason,
        "score_field": score_field,
        "score_fields_considered": list(score_fields),
        "evaluated_frames": evaluated_frames,
        "truth_boxes": truth_boxes,
        "skipped_score_rows": skipped_score_rows,
        "points": points,
        "point_count": len(points),
        "auc_fp_per_frame_le_1": froc_auc_up_to_fp_per_frame(
            points,
            max_fp_per_frame=FROC_AUC_FP_PER_FRAME_LIMIT,
        ),
        "recall_at_0_1_fp_per_frame": recall_limits[0.1],
        "recall_at_0_5_fp_per_frame": recall_limits[0.5],
        "recall_at_1_0_fp_per_frame": recall_limits[1.0],
    }


def load_run_data(spec: RunSpec) -> RunData:
    """Load standard outputs from one BeltMap run directory."""

    detections = read_csv_rows(
        spec.output_dir / "detections.csv",
        required_columns=REQUIRED_CSV_COLUMNS["detections.csv"],
    )
    records = parse_detection_records(detections)
    filtered_tracks = read_csv_rows(
        spec.output_dir / "filtered_tracks.csv",
        required_columns=REQUIRED_CSV_COLUMNS["filtered_tracks.csv"],
    )
    filtered_records = parse_detection_records(filtered_tracks)
    return RunData(
        spec=spec,
        metadata=read_json(spec.output_dir / "metadata.json"),
        detections=detections,
        detections_per_frame=read_csv_rows(
            spec.output_dir / "detections_per_frame.csv",
            required_columns=REQUIRED_CSV_COLUMNS["detections_per_frame.csv"],
        ),
        velocities=read_csv_rows(
            spec.output_dir / "velocities.csv",
            required_columns=REQUIRED_CSV_COLUMNS["velocities.csv"],
        ),
        filtered_velocities=read_csv_rows(
            spec.output_dir / "filtered_velocities.csv",
            required_columns=REQUIRED_CSV_COLUMNS["filtered_velocities.csv"],
        ),
        filtered_tracks=filtered_tracks,
        preview_paths=find_preview_paths(spec.output_dir),
        fixed_preview_paths=find_named_preview_paths(spec.output_dir, "residual_fixed"),
        raw_preview_paths=find_named_preview_paths(spec.output_dir, "raw"),
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


def empty_labeled_metrics() -> dict[str, Any]:
    """Return blank labeled-target metrics for proxy-only comparisons."""

    return {
        "labeled_detection_available": False,
        "labeled_scored_frames": None,
        "labeled_detection_iou_threshold": None,
        "labeled_truth_boxes": None,
        "labeled_predicted_boxes": None,
        "labeled_true_positives": None,
        "labeled_false_positives": None,
        "labeled_false_negatives": None,
        "labeled_false_positives_per_frame": None,
        "labeled_empty_scored_frames": None,
        "labeled_empty_frame_false_positives": None,
        "labeled_empty_frame_fp_per_frame": None,
        "labeled_precision": None,
        "labeled_recall": None,
        "labeled_f1": None,
        "labeled_froc_available": False,
        "labeled_froc_score_field": None,
        "labeled_froc_points": None,
        "labeled_froc_auc_fp_per_frame_le_1": None,
        "labeled_froc_recall_at_0_1_fp_per_frame": None,
        "labeled_froc_recall_at_0_5_fp_per_frame": None,
        "labeled_froc_recall_at_1_0_fp_per_frame": None,
        "labeled_mean_matched_iou": None,
        "labeled_mean_centroid_error_px": None,
    }


def summarize_run(
    data: RunData,
    *,
    labeled_truth: dict[str, Any] | None = None,
    truth_iou_threshold: float = 0.25,
    bootstrap_samples: int = 0,
    bootstrap_confidence_level: float = 0.95,
    bootstrap_seed: int | None = 0,
    bootstrap_block_length_frames: int = 1,
) -> dict[str, Any]:
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
    detection_threshold = finite_float(data.metadata.get("detection_threshold"))
    if detection_threshold is None:
        config = read_json(data.spec.output_dir / "config_resolved.json")
        options = config.get("options", {}) if isinstance(config, dict) else {}
        threshold_option = options.get("detection_threshold", {})
        if isinstance(threshold_option, dict):
            detection_threshold = finite_float(threshold_option.get("value"))
    detection_quality = detection_quality_summary(
        data.detections,
        small_area_threshold_px=DEFAULT_SMALL_AREA_THRESHOLD_PX,
        detection_threshold=detection_threshold,
        near_threshold_margin=DEFAULT_NEAR_THRESHOLD_MARGIN,
    )
    accepted_track_quality = accepted_track_quality_summary(
        data.filtered_tracks,
        small_area_threshold_px=DEFAULT_SMALL_AREA_THRESHOLD_PX,
    )
    plausible_ratios = sum(1 for value in velocity_ratios if 0.0 <= value <= 1.0)
    plausible_filtered_ratios = sum(
        1 for value in filtered_velocity_ratios if 0.0 <= value <= 1.0
    )
    long_ge_5 = sum(1 for value in velocity_track_lengths if value >= 5.0)
    long_ge_10 = sum(1 for value in velocity_track_lengths if value >= 10.0)

    row = {
        "label": data.spec.label,
        "output_dir": str(data.spec.output_dir),
        "complete": (data.spec.output_dir / "metadata.json").is_file(),
        "n_images": metadata_or_count(data, "n_images", data.detections_per_frame),
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
        "small_detection_share_area_lt_50": detection_quality[
            "small_detection_share_area_lt_threshold"
        ],
        "near_threshold_peak_share": detection_quality["near_threshold_peak_share"],
        "small_component_share_area_le_8": safe_share(small_components, len(detection_areas)),
        "small_accepted_tracks_lt_50": accepted_track_quality["small_accepted_tracks"],
        "long_small_accepted_tracks_ge_5": accepted_track_quality[
            "long_small_accepted_tracks_ge_5"
        ],
        "long_small_accepted_tracks_ge_10": accepted_track_quality[
            "long_small_accepted_tracks_ge_10"
        ],
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
    row.update(empty_labeled_metrics())
    row.update(empty_bootstrap_metrics())
    scored_frames: set[int] | None = None
    if labeled_truth is not None:
        scored_frames = truth_frame_indices(labeled_truth)
        scored_detections = restrict_detection_rows_to_frames(data.detections, scored_frames)
        metrics = detection_metrics(
            scored_detections,
            labeled_truth,
            iou_threshold=truth_iou_threshold,
        )
        false_positives = finite_int(metrics.get("false_positives")) or 0
        empty_frames = empty_labeled_frame_indices(labeled_truth)
        empty_frame_false_positives = len(
            restrict_detection_rows_to_frames(data.detections, empty_frames)
        )
        row.update(
            {
                "labeled_detection_available": metrics.get("available"),
                "labeled_scored_frames": len(scored_frames),
                "labeled_detection_iou_threshold": metrics.get("iou_threshold"),
                "labeled_truth_boxes": metrics.get("truth_boxes"),
                "labeled_predicted_boxes": metrics.get("predicted_boxes"),
                "labeled_true_positives": metrics.get("true_positives"),
                "labeled_false_positives": metrics.get("false_positives"),
                "labeled_false_negatives": metrics.get("false_negatives"),
                "labeled_false_positives_per_frame": safe_share(
                    false_positives,
                    len(scored_frames),
                ),
                "labeled_empty_scored_frames": len(empty_frames),
                "labeled_empty_frame_false_positives": empty_frame_false_positives,
                "labeled_empty_frame_fp_per_frame": safe_share(
                    empty_frame_false_positives,
                    len(empty_frames),
                ),
                **{f"labeled_{field}": metrics.get(field) for field in LABELED_METRIC_FIELDS},
            }
        )
        froc = detection_froc_curve(
            scored_detections,
            labeled_truth,
            scored_frames=scored_frames,
            iou_threshold=truth_iou_threshold,
        )
        row.update(
            {
                "labeled_froc_available": froc.get("available"),
                "labeled_froc_score_field": froc.get("score_field"),
                "labeled_froc_points": froc.get("point_count"),
                "labeled_froc_auc_fp_per_frame_le_1": froc.get("auc_fp_per_frame_le_1"),
                "labeled_froc_recall_at_0_1_fp_per_frame": froc.get("recall_at_0_1_fp_per_frame"),
                "labeled_froc_recall_at_0_5_fp_per_frame": froc.get("recall_at_0_5_fp_per_frame"),
                "labeled_froc_recall_at_1_0_fp_per_frame": froc.get("recall_at_1_0_fp_per_frame"),
            }
        )
    if bootstrap_samples > 0:
        row.update(
            bootstrap_run_summary(
                detections_per_frame=data.detections_per_frame,
                detections=data.detections,
                velocities=data.velocities,
                filtered_velocities=data.filtered_velocities,
                labeled_truth=labeled_truth,
                scored_frames=scored_frames,
                truth_iou_threshold=truth_iou_threshold,
                samples=bootstrap_samples,
                confidence_level=bootstrap_confidence_level,
                seed=bootstrap_seed,
                block_length_frames=bootstrap_block_length_frames,
            )
        )
    return row


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


def draw_labeled_froc_plot(
    path: Path,
    runs: list[RunData],
    *,
    labeled_truth: dict[str, Any],
    truth_iou_threshold: float,
) -> None:
    """Draw labeled detection FROC curves for the comparison report."""

    scored_frames = truth_frame_indices(labeled_truth)
    labeled_series: list[tuple[str, list[float], list[float]]] = []
    for run in runs:
        froc = detection_froc_curve(
            run.detections,
            labeled_truth,
            scored_frames=scored_frames,
            iou_threshold=truth_iou_threshold,
        )
        points = monotone_froc_points(list(froc.get("points") or []))
        labeled_series.append(
            (
                run.spec.label,
                [false_positives_per_frame for false_positives_per_frame, _recall in points],
                [recall for _false_positives_per_frame, recall in points],
            )
        )

    draw_multiline_plot(
        path,
        title="Labeled detection FROC",
        labeled_series=labeled_series,
        x_label="false positives per scored frame",
        y_label="recall",
    )


def detection_count_for_frame(data: RunData, frame_index: int, *, filtered: bool = False) -> int:
    """Return the loaded detection count for a frame."""

    detections = data.filtered_detections_by_frame if filtered else data.detections_by_frame
    return len(detections.get(frame_index, []))


def preview_paths_for_kind(data: RunData, preview_kind: str) -> dict[int, Path]:
    """Return the requested preview map."""

    if preview_kind == "raw":
        return data.raw_preview_paths
    if preview_kind == "residual_fixed":
        return data.fixed_preview_paths
    if preview_kind == "residual":
        return data.preview_paths
    raise ValueError(f"unknown preview kind: {preview_kind}")


def preview_kind_label(preview_kind: str) -> str:
    labels = {
        "raw": "raw",
        "residual_fixed": "fixed residual",
        "residual": "residual",
    }
    return labels.get(preview_kind, preview_kind)


def draw_detection_overlay_tile(
    data: RunData,
    frame_index: int,
    *,
    width: int,
    header_height: int,
    filtered: bool = False,
    preview_kind: str = "residual",
) -> Image.Image:
    """Create a resized residual-preview tile with detection boxes."""

    detections_by_frame = (
        data.filtered_detections_by_frame
        if filtered
        else data.detections_by_frame
    )
    preview = preview_paths_for_kind(data, preview_kind).get(frame_index)
    if preview is None:
        tile = Image.new("RGB", (width, header_height + 220), (248, 248, 248))
        draw = ImageDraw.Draw(tile)
        prefix = "filtered " if filtered else ""
        kind = preview_kind_label(preview_kind)
        draw.text((10, 10), f"{data.spec.label} | {prefix}{kind} frame {frame_index}", fill="black")
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
    kind = preview_kind_label(preview_kind)
    label = (
        f"{data.spec.label} | {prefix}{kind} frame {frame_index} | "
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
    preview_kind: str = "residual",
) -> None:
    """Write a side-by-side residual-preview contact sheet with detection boxes."""

    if not frames:
        frames = sorted(
            {frame for run in runs for frame in preview_paths_for_kind(run, preview_kind)}
        )[:5]
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
                    preview_kind=preview_kind,
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


def write_plots(
    report_dir: Path,
    runs: list[RunData],
    *,
    labeled_truth: dict[str, Any] | None = None,
    truth_iou_threshold: float = 0.25,
) -> tuple[dict[str, Path], dict[str, Path]]:
    """Write comparison plots and contact sheets."""

    plots = {
        "detections_per_frame": report_dir / "detections_per_frame_comparison.png",
        "velocity_ratio_histogram": report_dir / "velocity_ratio_histogram_comparison.png",
    }
    if labeled_truth is not None:
        plots["labeled_detection_froc"] = report_dir / "labeled_detection_froc.png"
    images = {
        "detection_contact_sheet": report_dir / "detection_contact_sheet.png",
        "filtered_detection_contact_sheet": report_dir / "filtered_detection_contact_sheet.png",
    }
    if any(run.fixed_preview_paths for run in runs):
        images.update(
            {
                "fixed_scale_detection_contact_sheet": report_dir / "fixed_scale_detection_contact_sheet.png",
                "fixed_scale_filtered_detection_contact_sheet": report_dir / "fixed_scale_filtered_detection_contact_sheet.png",
            }
        )
    if any(run.raw_preview_paths for run in runs):
        images.update(
            {
                "raw_detection_contact_sheet": report_dir / "raw_detection_contact_sheet.png",
                "raw_filtered_detection_contact_sheet": report_dir / "raw_filtered_detection_contact_sheet.png",
            }
        )
    series = []
    for run in runs:
        xs, ys = paired_values(
            run.detections_per_frame,
            x_field="frame_index",
            y_field="n_detections",
        )
        if not xs:
            ys = finite_values(run.detections_per_frame, "n_detections")
            xs = [float(index) for index in range(len(ys))]
        series.append((run.spec.label, xs, ys))
    draw_multiline_plot(
        plots["detections_per_frame"],
        title="Detections per frame",
        labeled_series=series,
        x_label="frame_index",
        y_label="n_detections",
    )
    draw_velocity_ratio_histogram(plots["velocity_ratio_histogram"], runs)
    if labeled_truth is not None:
        draw_labeled_froc_plot(
            plots["labeled_detection_froc"],
            runs,
            labeled_truth=labeled_truth,
            truth_iou_threshold=truth_iou_threshold,
        )
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
    truth_path: Path | None = None,
    truth_iou_threshold: float = 0.25,
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
        ("small_detection_share_area_lt_50", "small det <50 share"),
        ("small_component_share_area_le_8", "small <=8px share"),
        ("small_accepted_tracks_lt_50", "small accepted tracks"),
        ("velocity_ratio_median", "ratio median"),
        ("velocity_ratio_share_0_to_1", "ratio 0..1 share"),
        ("filtered_velocity_ratio_share_0_to_1", "filtered 0..1 share"),
        ("long_velocity_tracks_ge_10", "tracks >=10"),
    ]
    lines = [
        "# BeltMap run comparison",
        "",
        "This report compares detection-only or full BeltMap output directories.",
    ]
    if truth_path is None:
        lines.append("Use it as a decision aid, not as a ground-truth benchmark.")
    else:
        lines.append(
            "Labeled detection metrics are the primary target on the manually scored frames; proxy metrics remain secondary."
        )
    lines.extend(
        [
        "",
        "## Summary",
        "",
        "| " + " | ".join(label for _field, label in table_fields) + " |",
        "| " + " | ".join("---" for _ in table_fields) + " |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(format_value(row.get(field)) for field, _label in table_fields)
            + " |"
        )
    if truth_path is not None:
        labeled_fields = [
            ("label", "run"),
            ("labeled_detection_available", "available"),
            ("labeled_scored_frames", "scored frames"),
            ("labeled_truth_boxes", "truth boxes"),
            ("labeled_predicted_boxes", "pred boxes"),
            ("labeled_true_positives", "TP"),
            ("labeled_false_positives", "FP"),
            ("labeled_false_negatives", "FN"),
            ("labeled_false_positives_per_frame", "FP/frame"),
            ("labeled_empty_scored_frames", "empty frames"),
            ("labeled_empty_frame_false_positives", "empty FP"),
            ("labeled_empty_frame_fp_per_frame", "empty FP/frame"),
            ("labeled_precision", "precision"),
            ("labeled_recall", "recall"),
            ("labeled_f1", "F1"),
            ("labeled_froc_score_field", "FROC score"),
            ("labeled_froc_auc_fp_per_frame_le_1", "FROC AUC<=1 FP/frame"),
            ("labeled_froc_recall_at_0_1_fp_per_frame", "recall @0.1 FP/frame"),
            ("labeled_froc_recall_at_0_5_fp_per_frame", "recall @0.5 FP/frame"),
            ("labeled_froc_recall_at_1_0_fp_per_frame", "recall @1.0 FP/frame"),
            ("labeled_mean_matched_iou", "mean IoU"),
            ("labeled_mean_centroid_error_px", "centroid px"),
        ]
        lines.extend(
            [
                "",
                "## Labeled real-data target",
                "",
                f"Truth labels: `{truth_path}`; detection matching IoU threshold: {format_value(truth_iou_threshold)}.",
                "Detections outside the scored frame set are ignored so sparse labels do not penalize unlabeled frames.",
                "Use labeled F1 and fixed-budget FROC recall as the primary labeled target metrics.",
                "",
                "| " + " | ".join(label for _field, label in labeled_fields) + " |",
                "| " + " | ".join("---" for _ in labeled_fields) + " |",
            ]
        )
        for row in rows:
            lines.append(
                "| "
                + " | ".join(format_value(row.get(field)) for field, _label in labeled_fields)
                + " |"
            )
    bootstrap_rows = [row for row in rows if finite_int(row.get("bootstrap_samples"))]
    if bootstrap_rows:
        first_bootstrap = bootstrap_rows[0]
        proxy_ci_fields = [
            ("label", "run"),
            ("detections_per_frame_mean", "det/frame mean"),
            ("detections_per_frame_median", "det/frame median"),
            ("velocity_ratio_median", "ratio median"),
            ("velocity_ratio_share_0_to_1", "ratio 0..1 share"),
            ("filtered_velocity_ratio_median", "filtered ratio median"),
            ("filtered_velocity_ratio_share_0_to_1", "filtered 0..1 share"),
            ("long_velocity_tracks_ge_5", "tracks >=5"),
            ("long_velocity_tracks_ge_10", "tracks >=10"),
        ]
        lines.extend(
            [
                "",
                "## Bootstrap confidence intervals",
                "",
                "Values are bootstrap median [low, high] intervals. Frame-count and labeled metrics use circular contiguous frame blocks; velocity and track-length summaries resample velocity rows.",
                f"Samples: {format_value(first_bootstrap.get('bootstrap_samples'))}; confidence level: {format_value(first_bootstrap.get('bootstrap_confidence_level'))}; frame block length: {format_value(first_bootstrap.get('bootstrap_block_length_frames'))}.",
                "",
                "| " + " | ".join(label for _field, label in proxy_ci_fields) + " |",
                "| " + " | ".join("---" for _ in proxy_ci_fields) + " |",
            ]
        )
        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    format_value(row.get(field))
                    if field == "label"
                    else format_bootstrap_interval(row, field)
                    for field, _label in proxy_ci_fields
                )
                + " |"
            )
        if truth_path is not None:
            labeled_ci_fields = [
                ("label", "run"),
                ("labeled_precision", "precision"),
                ("labeled_recall", "recall"),
                ("labeled_f1", "F1"),
                ("labeled_false_positives", "FP"),
                ("labeled_false_negatives", "FN"),
                ("labeled_mean_matched_iou", "mean IoU"),
                ("labeled_mean_centroid_error_px", "centroid px"),
            ]
            lines.extend(
                [
                    "",
                    "### Labeled-frame bootstrap intervals",
                    "",
                    "| " + " | ".join(label for _field, label in labeled_ci_fields) + " |",
                    "| " + " | ".join("---" for _ in labeled_ci_fields) + " |",
                ]
            )
            for row in rows:
                lines.append(
                    "| "
                    + " | ".join(
                        format_value(row.get(field))
                        if field == "label"
                        else format_bootstrap_interval(row, field)
                        for field, _label in labeled_ci_fields
                    )
                    + " |"
                )
    if "labeled_detection_froc" in plots:
        lines.extend(
            [
                "",
                "### Labeled detection FROC",
                "",
                "The FROC curve sweeps the loaded detections by their per-component score, using `peak_signal` when available. It measures recall against false positives per scored frame, so empty scored frames contribute to the false-positive budget. Because detections below the run's original detector threshold were never written, use this plot as a within-run confidence sweep; full threshold sweeps still require rerunning the detector.",
                "",
                f"![Labeled detection FROC]({markdown_link(plots['labeled_detection_froc'], relative_to=report_dir)})",
                "",
            ]
        )
    lines.extend(["", "## Visual comparison", ""])
    if "raw_detection_contact_sheet" in images:
        lines.extend(
            [
                "Raw crops use one shared display scale across the sampled frames, so frame-to-frame brightness is comparable.",
                "",
                f"![Raw detection contact sheet]({markdown_link(images['raw_detection_contact_sheet'], relative_to=report_dir)})",
                "",
            ]
        )
    if "fixed_scale_detection_contact_sheet" in images:
        lines.extend(
            [
                "Fixed residual previews use a fixed normalized-residual display range instead of per-frame percentile stretching.",
                "",
                f"![Fixed-scale detection contact sheet]({markdown_link(images['fixed_scale_detection_contact_sheet'], relative_to=report_dir)})",
                "",
            ]
        )
    lines.extend(
        [
            "Autoscaled residual previews are kept for compatibility with older runs.",
            "",
            f"![Detection contact sheet]({markdown_link(images['detection_contact_sheet'], relative_to=report_dir)})",
            "",
            "## Filtered-track visual comparison",
            "",
        ]
    )
    if "raw_filtered_detection_contact_sheet" in images:
        lines.extend(
            [
                f"![Raw filtered detection contact sheet]({markdown_link(images['raw_filtered_detection_contact_sheet'], relative_to=report_dir)})",
                "",
            ]
        )
    if "fixed_scale_filtered_detection_contact_sheet" in images:
        lines.extend(
            [
                f"![Fixed-scale filtered detection contact sheet]({markdown_link(images['fixed_scale_filtered_detection_contact_sheet'], relative_to=report_dir)})",
                "",
            ]
        )
    lines.extend(
        [
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
        ]
    )
    if truth_path is not None:
        lines.append(
            "- Prefer higher labeled recall at a fixed false-positive-per-frame budget or higher labeled FROC AUC; empty-frame FP counts are the fastest ghost-artifact warning."
        )
    lines.extend(
        [
            "- Use the contact sheet to inspect which false positives are belt scratches or map ghosts.",
            "- Prefer compact filled particles over hollow or fractured particle components when labeled metrics are tied.",
            "- Use velocity-ratio plausibility and long-track counts as secondary checks, not as substitutes for labels.",
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
    truth_path: Path | None = None,
    truth_iou_threshold: float = 0.25,
    bootstrap_samples: int = 0,
    bootstrap_confidence_level: float = 0.95,
    bootstrap_seed: int | None = 0,
    bootstrap_block_length_frames: int = 1,
) -> ComparisonArtifacts:
    """Generate summary CSV, comparison plots, and a Markdown report."""

    if len(specs) < 2:
        raise ValueError("at least two runs are required for comparison")
    if bootstrap_samples < 0:
        raise ValueError("bootstrap samples must be non-negative")
    if not 0.0 < bootstrap_confidence_level < 1.0:
        raise ValueError("bootstrap confidence level must be between 0 and 1")
    if bootstrap_block_length_frames < 1:
        raise ValueError("bootstrap block length must be at least 1 frame")
    report_dir.mkdir(parents=True, exist_ok=True)
    labeled_truth = (
        None if truth_path is None else load_labeled_detection_truth(truth_path)
    )
    runs = [load_run_data(spec) for spec in specs]
    rows = []
    for run_index, run in enumerate(runs):
        bootstrap_seed_for_run = None if bootstrap_seed is None else bootstrap_seed + run_index
        rows.append(
            summarize_run(
                run,
                labeled_truth=labeled_truth,
                truth_iou_threshold=truth_iou_threshold,
                bootstrap_samples=bootstrap_samples,
                bootstrap_confidence_level=bootstrap_confidence_level,
                bootstrap_seed=bootstrap_seed_for_run,
                bootstrap_block_length_frames=bootstrap_block_length_frames,
            )
        )
    summary_csv = report_dir / "summary.csv"
    write_summary_csv(summary_csv, rows)
    plots, images = write_plots(
        report_dir,
        runs,
        labeled_truth=labeled_truth,
        truth_iou_threshold=truth_iou_threshold,
    )
    contact_sheet_specs = [
        ("detection_contact_sheet", "residual", False),
        ("filtered_detection_contact_sheet", "residual", True),
        ("fixed_scale_detection_contact_sheet", "residual_fixed", False),
        ("fixed_scale_filtered_detection_contact_sheet", "residual_fixed", True),
        ("raw_detection_contact_sheet", "raw", False),
        ("raw_filtered_detection_contact_sheet", "raw", True),
    ]
    for key, preview_kind, filtered in contact_sheet_specs:
        if key not in images:
            continue
        draw_detection_contact_sheet(
            images[key],
            runs,
            frames=[] if frames is None else frames,
            filtered=filtered,
            preview_kind=preview_kind,
        )
    report = report_dir / "comparison_report.md"
    report.write_text(
        build_markdown_report(
            report,
            rows=rows,
            plots=plots,
            images=images,
            truth_path=truth_path,
            truth_iou_threshold=truth_iou_threshold,
        ),
        encoding="utf-8",
    )
    return ComparisonArtifacts(report=report, summary_csv=summary_csv, plots=plots, images=images)
