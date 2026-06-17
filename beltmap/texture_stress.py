from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from .benchmark import detection_metrics
from .compare_runs import (
    RunData,
    RunSpec,
    draw_multiline_plot,
    finite_float,
    finite_int,
    format_value,
    load_labeled_detection_truth,
    load_run_data,
    markdown_link,
    read_csv_rows,
    restrict_detection_rows_to_frames,
    row_frame_index,
    truth_frame_indices,
)


FRAME_FIELDS = [
    "frame_index",
    "subset",
    "stress_score",
    "raw_std",
    "raw_gradient_mad",
    "residual_mad",
    "residual_p95_abs",
    "registration_loss",
    "registration_score",
    "reference_detections_per_frame",
]

SUMMARY_FIELDS = [
    "run",
    "output_dir",
    "subset",
    "stress_rank",
    "stress_min",
    "stress_median",
    "stress_max",
    "n_stress_frames",
    "n_detections",
    "detections_per_frame_mean",
    "detections_per_frame_median",
    "detections_per_frame_max",
    "filtered_track_points",
    "filtered_tracks",
    "velocity_rows",
    "velocity_ratio_median",
    "labeled_detection_available",
    "labeled_scored_frames",
    "labeled_truth_boxes",
    "labeled_predicted_boxes",
    "labeled_true_positives",
    "labeled_false_positives",
    "labeled_false_negatives",
    "labeled_precision",
    "labeled_recall",
    "labeled_f1",
]

STRESS_FEATURES = [
    "raw_std",
    "raw_gradient_mad",
    "residual_mad",
    "residual_p95_abs",
    "registration_loss",
    "reference_detections_per_frame",
]


@dataclass(frozen=True)
class TextureStressArtifacts:
    """Files written by the texture-stress subset report."""

    report: Path
    frames_csv: Path
    summary_csv: Path
    plots: dict[str, Path]


@dataclass(frozen=True)
class StressFrame:
    """Per-frame difficulty score derived from existing BeltMap diagnostics."""

    frame_index: int
    subset: str
    stress_score: float | None
    raw_std: float | None = None
    raw_gradient_mad: float | None = None
    residual_mad: float | None = None
    residual_p95_abs: float | None = None
    registration_loss: float | None = None
    registration_score: float | None = None
    reference_detections_per_frame: float | None = None

    def as_row(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "subset": self.subset,
            "stress_score": self.stress_score,
            "raw_std": self.raw_std,
            "raw_gradient_mad": self.raw_gradient_mad,
            "residual_mad": self.residual_mad,
            "residual_p95_abs": self.residual_p95_abs,
            "registration_loss": self.registration_loss,
            "registration_score": self.registration_score,
            "reference_detections_per_frame": self.reference_detections_per_frame,
        }


def frame_indexed_float(rows: Iterable[dict[str, Any]], field: str) -> dict[int, float]:
    """Return finite ``field`` values keyed by frame index."""

    result: dict[int, float] = {}
    for row in rows:
        frame_index = row_frame_index(row)
        value = finite_float(row.get(field))
        if frame_index is None or value is None:
            continue
        result[frame_index] = value
    return result


def robust_mad(values: np.ndarray) -> float:
    """Return the median absolute deviation of a finite array."""

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    median = float(np.median(finite))
    return float(np.median(np.abs(finite - median)))


def robust_z_by_frame(values_by_frame: dict[int, float]) -> dict[int, float]:
    """Convert feature values to robust z-scores, clipping extreme leverage."""

    if len(values_by_frame) < 2:
        return {}
    frames = sorted(values_by_frame)
    values = np.asarray([values_by_frame[frame] for frame in frames], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return {}
    median = float(np.median(values))
    scale = 1.4826 * robust_mad(values)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(values))
    if not np.isfinite(scale) or scale <= 1e-12:
        return {}
    return {
        frame: float(np.clip((value - median) / scale, -5.0, 5.0))
        for frame, value in values_by_frame.items()
        if np.isfinite(value)
    }


def image_texture_metrics(path: Path) -> dict[str, float | None]:
    """Compute display-scale texture metrics from a saved preview image."""

    image = Image.open(path).convert("L")
    arr = np.asarray(image, dtype=np.float64)
    if arr.size == 0:
        return {"std": None, "gradient_mad": None, "mad": None, "p95_abs": None}
    median = float(np.median(arr))
    centered = arr - median
    diffs: list[np.ndarray] = []
    if arr.shape[0] > 1:
        diffs.append(np.diff(arr, axis=0).ravel())
    if arr.shape[1] > 1:
        diffs.append(np.diff(arr, axis=1).ravel())
    gradient_mad = None
    if diffs:
        gradient_mad = robust_mad(np.concatenate(diffs))
    return {
        "std": float(np.std(arr)),
        "gradient_mad": gradient_mad,
        "mad": robust_mad(centered.ravel()),
        "p95_abs": float(np.percentile(np.abs(centered), 95)),
    }


def collect_reference_features(reference: RunData) -> dict[int, dict[str, float | None]]:
    """Collect stress features from one reference run without rerunning BeltMap."""

    counts = frame_indexed_float(reference.detections_per_frame, "n_detections")
    phase_rows = read_csv_rows(reference.spec.output_dir / "phase_estimates.csv")
    losses = frame_indexed_float(phase_rows, "loss")
    scores = frame_indexed_float(phase_rows, "score")

    frames: set[int] = set(counts) | set(losses) | set(scores)
    frames.update(reference.raw_preview_paths)
    frames.update(reference.fixed_preview_paths)
    frames.update(reference.preview_paths)

    features: dict[int, dict[str, float | None]] = {
        frame: {
            "raw_std": None,
            "raw_gradient_mad": None,
            "residual_mad": None,
            "residual_p95_abs": None,
            "registration_loss": losses.get(frame),
            "registration_score": scores.get(frame),
            "reference_detections_per_frame": counts.get(frame),
        }
        for frame in sorted(frames)
    }

    for frame, path in reference.raw_preview_paths.items():
        metrics = image_texture_metrics(path)
        features.setdefault(frame, {})["raw_std"] = metrics["std"]
        features.setdefault(frame, {})["raw_gradient_mad"] = metrics["gradient_mad"]

    residual_paths = dict(reference.preview_paths)
    residual_paths.update(reference.fixed_preview_paths)
    for frame, path in residual_paths.items():
        metrics = image_texture_metrics(path)
        features.setdefault(frame, {})["residual_mad"] = metrics["mad"]
        features.setdefault(frame, {})["residual_p95_abs"] = metrics["p95_abs"]

    return features


def score_stress_frames(
    features: dict[int, dict[str, float | None]],
    *,
    quartiles: int = 4,
) -> list[StressFrame]:
    """Assign robust composite texture-stress scores and quantile subsets."""

    if quartiles < 2:
        raise ValueError("texture-stress analysis requires at least two subsets")
    feature_z: dict[str, dict[int, float]] = {}
    for feature in STRESS_FEATURES:
        values = {
            frame: float(value)
            for frame, row in features.items()
            if (value := row.get(feature)) is not None and np.isfinite(float(value))
        }
        z_values = robust_z_by_frame(values)
        if z_values:
            feature_z[feature] = z_values

    scores: dict[int, float | None] = {}
    for frame in sorted(features):
        values = [z_by_frame[frame] for z_by_frame in feature_z.values() if frame in z_by_frame]
        scores[frame] = None if not values else float(np.mean(values))

    finite_frames = [frame for frame, score in scores.items() if score is not None]
    finite_frames.sort(key=lambda frame: (float(scores[frame]), frame))
    subset_by_frame: dict[int, str] = {}
    total = len(finite_frames)
    for rank, frame in enumerate(finite_frames):
        subset_index = min(quartiles - 1, int(rank * quartiles / total)) if total else 0
        subset_by_frame[frame] = f"Q{subset_index + 1}"

    result: list[StressFrame] = []
    for frame in sorted(features):
        row = features[frame]
        result.append(
            StressFrame(
                frame_index=frame,
                subset=subset_by_frame.get(frame, "unscored"),
                stress_score=scores[frame],
                raw_std=row.get("raw_std"),
                raw_gradient_mad=row.get("raw_gradient_mad"),
                residual_mad=row.get("residual_mad"),
                residual_p95_abs=row.get("residual_p95_abs"),
                registration_loss=row.get("registration_loss"),
                registration_score=row.get("registration_score"),
                reference_detections_per_frame=row.get("reference_detections_per_frame"),
            )
        )
    return result


def subset_rank(label: str) -> int:
    if label.startswith("Q"):
        parsed = finite_int(label[1:])
        if parsed is not None:
            return parsed
    return 9999


def mean_or_none(values: Iterable[float]) -> float | None:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return None if arr.size == 0 else float(np.mean(arr))


def median_or_none(values: Iterable[float]) -> float | None:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return None if arr.size == 0 else float(np.median(arr))


def count_detections_in_frames(rows: Iterable[dict[str, Any]], frames: set[int]) -> int:
    return sum(1 for row in rows if row_frame_index(row) in frames)


def velocity_rows_in_frames(rows: Iterable[dict[str, Any]], frames: set[int]) -> list[dict[str, Any]]:
    """Assign velocity rows to stress subsets by track midpoint/start/end when available."""

    selected: list[dict[str, Any]] = []
    for row in rows:
        start = finite_int(row.get("frame_start"))
        end = finite_int(row.get("frame_end"))
        if start is not None and end is not None:
            midpoint = int(round((start + end) / 2.0))
            if midpoint in frames or start in frames or end in frames:
                selected.append(row)
            continue
        if start is not None and start in frames:
            selected.append(row)
    return selected


def subset_labeled_truth(labeled_truth: dict[str, Any], frames: set[int]) -> dict[str, Any]:
    scored_frames = truth_frame_indices(labeled_truth) & frames
    particles = [
        particle
        for particle in labeled_truth.get("particles", [])
        if isinstance(particle, dict) and row_frame_index(particle) in frames
    ]
    return {
        **labeled_truth,
        "particles": particles,
        "scored_frames": sorted(scored_frames),
    }


def empty_labeled_subset_metrics() -> dict[str, Any]:
    return {
        "labeled_detection_available": False,
        "labeled_scored_frames": None,
        "labeled_truth_boxes": None,
        "labeled_predicted_boxes": None,
        "labeled_true_positives": None,
        "labeled_false_positives": None,
        "labeled_false_negatives": None,
        "labeled_precision": None,
        "labeled_recall": None,
        "labeled_f1": None,
    }


def summarize_run_subset(
    run: RunData,
    *,
    subset: str,
    frames: set[int],
    stress_scores: list[float],
    labeled_truth: dict[str, Any] | None = None,
    truth_iou_threshold: float = 0.25,
) -> dict[str, Any]:
    counts = [
        float(value)
        for row in run.detections_per_frame
        if row_frame_index(row) in frames and (value := finite_float(row.get("n_detections"))) is not None
    ]
    detections = count_detections_in_frames(run.detections, frames)
    filtered_track_rows = [row for row in run.filtered_tracks if row_frame_index(row) in frames]
    filtered_track_ids = {
        str(row.get("track_id")) for row in filtered_track_rows if str(row.get("track_id", "")).strip()
    }
    velocity_rows = velocity_rows_in_frames(run.velocities, frames)
    velocity_ratios = [
        value
        for row in velocity_rows
        if (value := finite_float(row.get("velocity_ratio_y"))) is not None
    ]
    row = {
        "run": run.spec.label,
        "output_dir": str(run.spec.output_dir),
        "subset": subset,
        "stress_rank": subset_rank(subset),
        "stress_min": min(stress_scores) if stress_scores else None,
        "stress_median": median_or_none(stress_scores),
        "stress_max": max(stress_scores) if stress_scores else None,
        "n_stress_frames": len(frames),
        "n_detections": detections,
        "detections_per_frame_mean": mean_or_none(counts),
        "detections_per_frame_median": median_or_none(counts),
        "detections_per_frame_max": max(counts) if counts else None,
        "filtered_track_points": len(filtered_track_rows),
        "filtered_tracks": len(filtered_track_ids),
        "velocity_rows": len(velocity_rows),
        "velocity_ratio_median": median_or_none(velocity_ratios),
    }
    row.update(empty_labeled_subset_metrics())
    if labeled_truth is not None:
        scoped_truth = subset_labeled_truth(labeled_truth, frames)
        scored_frames = truth_frame_indices(scoped_truth)
        scored_detections = restrict_detection_rows_to_frames(run.detections, scored_frames)
        metrics = detection_metrics(
            scored_detections,
            scoped_truth,
            iou_threshold=truth_iou_threshold,
        )
        row.update(
            {
                "labeled_detection_available": metrics.get("available"),
                "labeled_scored_frames": len(scored_frames),
                "labeled_truth_boxes": metrics.get("truth_boxes"),
                "labeled_predicted_boxes": metrics.get("predicted_boxes"),
                "labeled_true_positives": metrics.get("true_positives"),
                "labeled_false_positives": metrics.get("false_positives"),
                "labeled_false_negatives": metrics.get("false_negatives"),
                "labeled_precision": metrics.get("precision"),
                "labeled_recall": metrics.get("recall"),
                "labeled_f1": metrics.get("f1"),
            }
        )
    return row


def grouped_stress_frames(stress_frames: Iterable[StressFrame]) -> dict[str, list[StressFrame]]:
    groups: dict[str, list[StressFrame]] = {}
    for frame in stress_frames:
        if frame.stress_score is None or frame.subset == "unscored":
            continue
        groups.setdefault(frame.subset, []).append(frame)
    return dict(sorted(groups.items(), key=lambda item: subset_rank(item[0])))


def summarize_texture_stress(
    runs: list[RunData],
    stress_frames: list[StressFrame],
    *,
    labeled_truth: dict[str, Any] | None = None,
    truth_iou_threshold: float = 0.25,
) -> list[dict[str, Any]]:
    groups = grouped_stress_frames(stress_frames)
    rows: list[dict[str, Any]] = []
    for run in runs:
        for subset, frames_in_subset in groups.items():
            frames = {frame.frame_index for frame in frames_in_subset}
            scores = [float(frame.stress_score) for frame in frames_in_subset if frame.stress_score is not None]
            rows.append(
                summarize_run_subset(
                    run,
                    subset=subset,
                    frames=frames,
                    stress_scores=scores,
                    labeled_truth=labeled_truth,
                    truth_iou_threshold=truth_iou_threshold,
                )
            )
    return rows


def write_csv_dicts(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def write_texture_stress_plots(report_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Path]:
    plots = {"detections_by_texture_stress": report_dir / "detections_by_texture_stress.png"}
    series: list[tuple[str, list[float], list[float]]] = []
    labels = sorted({str(row.get("run")) for row in rows})
    for label in labels:
        run_rows = [row for row in rows if row.get("run") == label]
        def stress_rank_sort_key(row: dict[str, Any]) -> float:
            rank = finite_float(row.get("stress_rank"))
            return math.inf if rank is None else float(rank)

        run_rows.sort(key=stress_rank_sort_key)
        xs = [float(row["stress_rank"]) for row in run_rows if finite_float(row.get("stress_rank")) is not None]
        ys = [
            float(value)
            for row in run_rows
            if (value := finite_float(row.get("detections_per_frame_mean"))) is not None
        ]
        if len(xs) == len(ys) and xs and ys:
            series.append((label, xs, ys))
    draw_multiline_plot(
        plots["detections_by_texture_stress"],
        title="Mean detections by texture-stress subset",
        labeled_series=series,
        x_label="texture-stress subset rank (Q1 low, Q4 high)",
        y_label="mean detections/frame",
    )
    return plots


def feature_coverage_rows(stress_frames: list[StressFrame]) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for field in STRESS_FEATURES + ["registration_score"]:
        count = sum(1 for frame in stress_frames if getattr(frame, field) is not None)
        rows.append((field, count))
    return rows


def build_texture_stress_markdown(
    report_path: Path,
    *,
    reference: RunData,
    stress_frames: list[StressFrame],
    rows: list[dict[str, Any]],
    plots: dict[str, Path],
    truth_path: Path | None = None,
    truth_iou_threshold: float = 0.25,
) -> str:
    report_dir = report_path.parent
    table_fields = [
        ("run", "run"),
        ("subset", "subset"),
        ("n_stress_frames", "frames"),
        ("stress_median", "stress median"),
        ("n_detections", "detections"),
        ("detections_per_frame_mean", "mean/frame"),
        ("detections_per_frame_max", "max/frame"),
        ("filtered_tracks", "filtered tracks"),
        ("velocity_ratio_median", "ratio median"),
    ]
    labeled_fields = [
        ("run", "run"),
        ("subset", "subset"),
        ("labeled_scored_frames", "scored frames"),
        ("labeled_truth_boxes", "truth boxes"),
        ("labeled_predicted_boxes", "pred boxes"),
        ("labeled_true_positives", "TP"),
        ("labeled_false_positives", "FP"),
        ("labeled_false_negatives", "FN"),
        ("labeled_precision", "precision"),
        ("labeled_recall", "recall"),
        ("labeled_f1", "F1"),
    ]
    lines = [
        "# Texture-stress subset analysis",
        "",
        "This report stratifies frames by a reference-run texture/residual stress score and then reports run metrics inside each subset.",
        "Q1 is the lowest-stress subset; the highest-numbered subset is the most artifact-prone stress subset.",
        "",
        f"Reference run: `{reference.spec.label}` (`{reference.spec.output_dir}`).",
        "",
        "## Stress feature coverage",
        "",
        "| feature | finite frames |",
        "| --- | ---: |",
    ]
    for feature, count in feature_coverage_rows(stress_frames):
        lines.append(f"| {feature} | {count} |")
    lines.extend(
        [
            "",
            "The composite stress score is the mean of robust z-scored available features, using saved raw/residual previews, phase-estimate loss, and the reference run's detection density when present.",
            "Because preview PNGs are display-scale diagnostics, use the subsets to compare methods under matched frame difficulty, not as calibrated physical texture measurements.",
            "",
            "## Proxy metrics by stress subset",
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
        lines.extend(
            [
                "",
                "## Labeled metrics by stress subset",
                "",
                f"Truth labels: `{truth_path}`; detection matching IoU threshold: {format_value(truth_iou_threshold)}.",
                "Sparse labels remain frame-scoped: detections outside scored frames are ignored inside each subset.",
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
    lines.extend(
        [
            "",
            "## Plot",
            "",
            f"![Detections by texture stress]({markdown_link(plots['detections_by_texture_stress'], relative_to=report_dir)})",
            "",
            "## Interpretation checklist",
            "",
            "- Look for methods whose false positives or detection density rise sharply from Q1 to the highest-stress subset.",
            "- Treat a raw-image baseline that only wins in Q1 as less concerning than one that remains strong in the highest-stress subset.",
            "- Use labeled precision/recall when labels are available; otherwise use this report as a frame-matched diagnostic, not a ground-truth benchmark.",
            "",
        ]
    )
    return "\n".join(lines)


def select_reference_run(runs: list[RunData], reference_label: str | None) -> RunData:
    if not runs:
        raise ValueError("at least one run is required")
    if reference_label is None:
        return runs[0]
    for run in runs:
        if run.spec.label == reference_label or str(run.spec.output_dir) == reference_label:
            return run
    labels = ", ".join(run.spec.label for run in runs)
    raise ValueError(f"reference run {reference_label!r} not found; available labels: {labels}")


def generate_texture_stress_report(
    specs: list[RunSpec],
    *,
    report_dir: Path,
    reference_label: str | None = None,
    quartiles: int = 4,
    truth_path: Path | None = None,
    truth_iou_threshold: float = 0.25,
) -> TextureStressArtifacts:
    """Generate a frame-difficulty-stratified report for BeltMap output runs."""

    if not specs:
        raise ValueError("at least one run is required")
    if quartiles < 2:
        raise ValueError("quartiles must be at least 2")
    report_dir.mkdir(parents=True, exist_ok=True)
    runs = [load_run_data(spec) for spec in specs]
    reference = select_reference_run(runs, reference_label)
    features = collect_reference_features(reference)
    if not features:
        raise ValueError(
            f"reference run {reference.spec.label!r} has no usable stress inputs; "
            "expected detections_per_frame.csv, phase_estimates.csv, or saved preview PNGs"
        )
    stress_frames = score_stress_frames(features, quartiles=quartiles)
    if not any(frame.stress_score is not None for frame in stress_frames):
        raise ValueError(
            f"reference run {reference.spec.label!r} did not contain enough varying stress features"
        )
    labeled_truth = None if truth_path is None else load_labeled_detection_truth(truth_path)
    rows = summarize_texture_stress(
        runs,
        stress_frames,
        labeled_truth=labeled_truth,
        truth_iou_threshold=truth_iou_threshold,
    )
    frames_csv = report_dir / "texture_stress_frames.csv"
    summary_csv = report_dir / "texture_stress_summary.csv"
    write_csv_dicts(frames_csv, [frame.as_row() for frame in stress_frames], FRAME_FIELDS)
    write_csv_dicts(summary_csv, rows, SUMMARY_FIELDS)
    plots = write_texture_stress_plots(report_dir, rows)
    report = report_dir / "texture_stress_report.md"
    report.write_text(
        build_texture_stress_markdown(
            report,
            reference=reference,
            stress_frames=stress_frames,
            rows=rows,
            plots=plots,
            truth_path=truth_path,
            truth_iou_threshold=truth_iou_threshold,
        ),
        encoding="utf-8",
    )
    return TextureStressArtifacts(
        report=report,
        frames_csv=frames_csv,
        summary_csv=summary_csv,
        plots=plots,
    )
