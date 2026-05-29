from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

EVENT_ID_KEYS = ("event_id", "particle_id", "track_id", "id")


@dataclass(frozen=True)
class BenchmarkArtifacts:
    """Files written by a ground-truth benchmark run."""

    metrics: Path
    report: Path


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from ``path``."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV rows, returning an empty list when the file is absent."""

    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_float(value: Any) -> float | None:
    """Return a finite float or ``None`` for missing/non-finite values."""

    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def finite_int(value: Any) -> int | None:
    """Return an integer or ``None`` when parsing fails."""

    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or not parsed.is_integer():
        return None
    return int(parsed)


def circular_signed_error_px(estimate: float, truth: float, period: float) -> float:
    """Return the signed circular error ``estimate - truth`` in pixels."""

    if period <= 0:
        raise ValueError("period must be positive")
    return float((estimate - truth + 0.5 * period) % period - 0.5 * period)


def source_frame_index(row: dict[str, Any]) -> int | None:
    """Infer the original source-frame index for a driver CSV row.

    The image driver writes processed-frame indices after striding/truncation,
    while the synthetic metadata uses original frame indices. For generated
    frames named ``frame_000.png`` this function therefore recovers ``0`` from
    the ``image`` column and falls back to ``frame_index`` if no number is found.
    """

    image = str(row.get("image", ""))
    if image:
        matches = re.findall(r"\d+", Path(image).stem)
        if matches:
            return int(matches[-1])
    return finite_int(row.get("frame_index"))


def summary_errors(values: Iterable[float], *, unit: str) -> dict[str, float | int | None]:
    """Summarize signed errors with absolute-error and RMSE statistics."""

    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            f"mean_error_{unit}": None,
            f"mean_abs_error_{unit}": None,
            f"median_abs_error_{unit}": None,
            f"p95_abs_error_{unit}": None,
            f"max_abs_error_{unit}": None,
            f"rmse_{unit}": None,
        }

    abs_arr = np.abs(arr)
    return {
        "count": int(arr.size),
        f"mean_error_{unit}": float(np.mean(arr)),
        f"mean_abs_error_{unit}": float(np.mean(abs_arr)),
        f"median_abs_error_{unit}": float(np.median(abs_arr)),
        f"p95_abs_error_{unit}": float(np.percentile(abs_arr, 95)),
        f"max_abs_error_{unit}": float(np.max(abs_arr)),
        f"rmse_{unit}": float(np.sqrt(np.mean(np.square(arr)))),
    }


def true_phase_px(truth: dict[str, Any], frame_index: int, period_px: float) -> float:
    """Return synthetic ground-truth phase for ``frame_index``."""

    explicit = truth.get("true_phase_px_by_frame")
    if isinstance(explicit, list) and 0 <= frame_index < len(explicit):
        value = finite_float(explicit[frame_index])
        if value is not None:
            return value % period_px

    belt_shift = finite_float(
        truth.get(
            "true_belt_velocity_y_px_per_frame",
            truth.get("belt_shift_px_per_frame"),
        )
    )
    if belt_shift is None:
        raise ValueError("Synthetic truth lacks belt velocity/shift information")
    return float((-belt_shift * frame_index) % period_px)


def phase_metrics(phase_rows: list[dict[str, str]], truth: dict[str, Any]) -> dict[str, Any]:
    """Compute circular phase-estimation errors against synthetic truth."""

    period = finite_float(truth.get("belt_period_px", truth.get("height")))
    if period is None or period <= 0:
        return {
            "available": False,
            "reason": "Synthetic truth lacks a positive belt_period_px",
            **summary_errors([], unit="px"),
        }

    errors: list[float] = []
    skipped_rows = 0
    for row in phase_rows:
        estimate = finite_float(row.get("phase_px"))
        frame_index = source_frame_index(row)
        if estimate is None or frame_index is None:
            skipped_rows += 1
            continue
        target = true_phase_px(truth, frame_index, period)
        errors.append(circular_signed_error_px(estimate, target, period))

    return {
        "available": bool(errors),
        "period_px": period,
        "skipped_rows": skipped_rows,
        **summary_errors(errors, unit="px"),
    }


def resolve_truth_path(truth_path: Path, value: Any, fallback_name: str) -> Path:
    """Resolve a truth artifact path relative to the synthetic metadata file."""

    if value:
        candidate = Path(str(value))
        return candidate if candidate.is_absolute() else truth_path.parent / candidate
    return truth_path.parent / fallback_name


def map_metrics(output_dir: Path, truth_path: Path, truth: dict[str, Any]) -> dict[str, Any]:
    """Compare reconstructed and true belt maps with cyclic row-shift invariance."""

    reconstructed_path = output_dir / "belt_map.npy"
    true_map_path = resolve_truth_path(
        truth_path,
        truth.get("true_belt_map_npy"),
        "true_belt_map.npy",
    )
    if not reconstructed_path.is_file():
        return {"available": False, "reason": f"Missing {reconstructed_path}"}
    if not true_map_path.is_file():
        return {"available": False, "reason": f"Missing {true_map_path}"}

    reconstructed = np.asarray(np.load(reconstructed_path), dtype=np.float64)
    target = np.asarray(np.load(true_map_path), dtype=np.float64)
    if reconstructed.shape != target.shape:
        return {
            "available": False,
            "reason": "Shape mismatch",
            "reconstructed_shape": list(reconstructed.shape),
            "truth_shape": list(target.shape),
        }
    if reconstructed.ndim != 2:
        return {
            "available": False,
            "reason": "Expected 2-D belt maps",
            "reconstructed_shape": list(reconstructed.shape),
        }

    best_shift = 0
    best_rmse = float("inf")
    best_mae = float("inf")
    for shift in range(target.shape[0]):
        shifted = np.roll(reconstructed, shift=shift, axis=0)
        error = shifted - target
        rmse = float(np.sqrt(np.mean(np.square(error))))
        if rmse < best_rmse:
            best_shift = shift
            best_rmse = rmse
            best_mae = float(np.mean(np.abs(error)))

    return {
        "available": True,
        "truth_map": str(true_map_path),
        "reconstructed_map": str(reconstructed_path),
        "shape": list(reconstructed.shape),
        "best_cyclic_shift_px": int(best_shift),
        "rmse_gray": best_rmse,
        "mean_abs_error_gray": best_mae,
    }


def bbox_from_truth(row: dict[str, Any]) -> dict[str, float]:
    """Convert a synthetic truth particle entry to a box."""

    return {
        "top": float(row["top"]),
        "left": float(row["left"]),
        "bottom": float(row["bottom"]),
        "right": float(row["right"]),
    }


def bbox_from_detection(row: dict[str, Any]) -> dict[str, float] | None:
    """Convert one detection CSV row to a box with optional centroid fields."""

    required = ("bbox_top", "bbox_left", "bbox_bottom", "bbox_right")
    values = [finite_float(row.get(name)) for name in required]
    if any(value is None for value in values):
        return None
    top, left, bottom, right = values
    assert top is not None and left is not None
    assert bottom is not None and right is not None
    box = {"top": top, "left": left, "bottom": bottom, "right": right}
    y = finite_float(row.get("y"))
    x = finite_float(row.get("x"))
    if y is not None and x is not None:
        box["centroid_y"] = y
        box["centroid_x"] = x
    return box


def bbox_iou(a: dict[str, float], b: dict[str, float]) -> float:
    """Compute intersection-over-union for half-open boxes."""

    top = max(a["top"], b["top"])
    left = max(a["left"], b["left"])
    bottom = min(a["bottom"], b["bottom"])
    right = min(a["right"], b["right"])
    intersection = max(0.0, bottom - top) * max(0.0, right - left)
    area_a = max(0.0, a["bottom"] - a["top"]) * max(0.0, a["right"] - a["left"])
    area_b = max(0.0, b["bottom"] - b["top"]) * max(0.0, b["right"] - b["left"])
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else float(intersection / union)


def truth_center(box: dict[str, float]) -> tuple[float, float]:
    """Return the pixel-center coordinate of a half-open integer box."""

    return (
        0.5 * (box["top"] + box["bottom"] - 1.0),
        0.5 * (box["left"] + box["right"] - 1.0),
    )


def predicted_center(box: dict[str, float]) -> tuple[float, float]:
    """Return the detected centroid, falling back to box center if absent."""

    y = box.get("centroid_y")
    x = box.get("centroid_x")
    if y is not None and x is not None:
        return y, x
    return truth_center(box)


def center_distance_px(a: dict[str, float], b: dict[str, float]) -> float:
    """Return Euclidean distance between two box centroids."""

    ay, ax = predicted_center(a)
    by, bx = predicted_center(b)
    return float(math.hypot(ay - by, ax - bx))


def box_diagonal_px(box: dict[str, float]) -> float:
    """Return the diagonal length of a half-open box."""

    return float(
        math.hypot(
            max(0.0, box["bottom"] - box["top"]),
            max(0.0, box["right"] - box["left"]),
        )
    )


def event_id_from_row(row: dict[str, Any]) -> str | None:
    """Return an explicit event/particle identifier when present."""

    for key in EVENT_ID_KEYS:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def truth_particle_rows(truth: dict[str, Any]) -> list[dict[str, Any]]:
    """Return per-frame truth particle rows from supported metadata layouts."""

    rows: list[dict[str, Any]] = []
    for particle in truth.get("particles", []):
        if isinstance(particle, dict):
            rows.append(particle)

    frames = truth.get("frames", [])
    if isinstance(frames, list):
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            frame_index = finite_int(frame.get("frame_index"))
            for box in frame.get("boxes", []):
                if not isinstance(box, dict):
                    continue
                row = dict(box)
                if frame_index is not None and finite_int(row.get("frame_index")) is None:
                    row["frame_index"] = frame_index
                rows.append(row)
    return rows


def truth_event_boxes(truth: dict[str, Any]) -> list[dict[str, Any]]:
    """Return synthetic truth boxes with frame and optional event identifiers."""

    boxes: list[dict[str, Any]] = []
    for particle in truth_particle_rows(truth):
        frame_index = finite_int(particle.get("frame_index"))
        if frame_index is None:
            continue
        try:
            box: dict[str, Any] = bbox_from_truth(particle)
        except KeyError:
            continue
        box["frame_index"] = frame_index
        event_id = event_id_from_row(particle)
        if event_id is not None:
            box["event_id"] = event_id
        boxes.append(box)
    return boxes


def predicted_event_boxes(detection_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Return prediction boxes with frame and optional event identifiers."""

    boxes: list[dict[str, Any]] = []
    for row in detection_rows:
        frame_index = source_frame_index(row)
        box = bbox_from_detection(row)
        if frame_index is None or box is None:
            continue
        event_box: dict[str, Any] = dict(box)
        event_box["frame_index"] = frame_index
        event_id = event_id_from_row(row)
        if event_id is not None:
            event_box["event_id"] = event_id
        boxes.append(event_box)
    return boxes


def event_center_link_threshold_px(boxes: list[dict[str, Any]]) -> float:
    """Infer a conservative center-distance threshold for auto-linking boxes."""

    diagonals = [box_diagonal_px(box) for box in boxes]
    if not diagonals:
        return 0.0
    return float(max(1.5, 1.5 * np.median(np.asarray(diagonals, dtype=np.float64))))


def box_link_score(
    previous: dict[str, Any],
    current: dict[str, Any],
    *,
    iou_threshold: float,
    center_threshold_px: float,
) -> float | None:
    """Return a link score for adjacent event boxes, or ``None`` if not linked."""

    iou = bbox_iou(previous, current)
    distance = center_distance_px(previous, current)
    if iou >= iou_threshold or distance <= center_threshold_px:
        distance_score = max(0.0, 1.0 - distance / max(center_threshold_px, 1e-9))
        return float(max(iou, distance_score))
    return None


def make_event(event_id: str, boxes: list[dict[str, Any]]) -> dict[str, Any]:
    """Create an event dictionary from per-frame boxes."""

    event = {
        "event_id": event_id,
        "boxes": sorted(boxes, key=lambda item: int(item["frame_index"])),
    }
    update_event_summary(event)
    return event


def update_event_summary(event: dict[str, Any]) -> None:
    """Update cached event summary fields in place."""

    frames = [int(box["frame_index"]) for box in event["boxes"]]
    event["frame_start"] = min(frames)
    event["frame_end"] = max(frames)
    event["n_frames"] = len(set(frames))
    event["duration_frames"] = event["frame_end"] - event["frame_start"] + 1


def build_events_from_boxes(
    boxes: list[dict[str, Any]],
    *,
    prefix: str,
    iou_threshold: float,
    max_frame_gap: int = 1,
) -> list[dict[str, Any]]:
    """Group per-frame boxes into temporally linked events.

    Explicit identifiers such as ``event_id``, ``particle_id``, or ``track_id``
    are honored when present. Boxes without explicit identifiers are linked with
    greedy nearest-neighbor matching between adjacent frames.
    """

    if max_frame_gap < 1:
        raise ValueError("max_frame_gap must be at least 1")

    explicit: dict[str, list[dict[str, Any]]] = {}
    unassigned: list[dict[str, Any]] = []
    for box in boxes:
        event_id = box.get("event_id")
        if event_id is None:
            unassigned.append(box)
        else:
            explicit.setdefault(str(event_id), []).append(box)

    events: list[dict[str, Any]] = []
    for event_id, event_boxes in sorted(explicit.items()):
        events.append(make_event(f"{prefix}:{event_id}", event_boxes))

    center_threshold_px = event_center_link_threshold_px(unassigned)
    active_event_indices: list[int] = []
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for box in unassigned:
        frame_index = finite_int(box.get("frame_index"))
        if frame_index is not None:
            by_frame.setdefault(frame_index, []).append(box)

    for frame_index in sorted(by_frame):
        current = sorted(by_frame[frame_index], key=lambda item: predicted_center(item))
        active_event_indices = [
            event_index
            for event_index in active_event_indices
            if frame_index - int(events[event_index]["frame_end"]) <= max_frame_gap
        ]
        candidates: list[tuple[float, int, int]] = []
        for event_index in active_event_indices:
            previous = events[event_index]["boxes"][-1]
            dt = frame_index - int(previous["frame_index"])
            if dt <= 0 or dt > max_frame_gap:
                continue
            for box_index, box in enumerate(current):
                score = box_link_score(
                    previous,
                    box,
                    iou_threshold=iou_threshold,
                    center_threshold_px=center_threshold_px,
                )
                if score is not None:
                    candidates.append((score, event_index, box_index))

        assigned_events: set[int] = set()
        assigned_boxes: set[int] = set()
        for _score, event_index, box_index in sorted(candidates, reverse=True):
            if event_index in assigned_events or box_index in assigned_boxes:
                continue
            events[event_index]["boxes"].append(current[box_index])
            update_event_summary(events[event_index])
            assigned_events.add(event_index)
            assigned_boxes.add(box_index)

        for box_index, box in enumerate(current):
            if box_index in assigned_boxes:
                continue
            events.append(make_event(f"{prefix}:auto-{len(events)}", [box]))
            active_event_indices.append(len(events) - 1)

    return events


def event_boxes_by_frame(event: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    """Group one event's boxes by frame."""

    grouped: dict[int, list[dict[str, Any]]] = {}
    for box in event["boxes"]:
        grouped.setdefault(int(box["frame_index"]), []).append(box)
    return grouped


def compare_events(
    predicted: dict[str, Any],
    target: dict[str, Any],
    *,
    iou_threshold: float,
) -> dict[str, Any]:
    """Compare one predicted event with one truth event."""

    pred_by_frame = event_boxes_by_frame(predicted)
    truth_by_frame = event_boxes_by_frame(target)
    common_frames = sorted(set(pred_by_frame) & set(truth_by_frame))
    matched_frames = 0
    matched_ious: list[float] = []
    first_matched_frame: int | None = None

    for frame_index in common_frames:
        best_iou = max(
            bbox_iou(pred, truth)
            for pred in pred_by_frame[frame_index]
            for truth in truth_by_frame[frame_index]
        )
        if best_iou >= iou_threshold:
            matched_frames += 1
            matched_ious.append(best_iou)
            if first_matched_frame is None:
                first_matched_frame = frame_index

    union_frames = set(pred_by_frame) | set(truth_by_frame)
    truth_frames = max(1, int(target["n_frames"]))
    pred_frames = max(1, int(predicted["n_frames"]))
    return {
        "pred_event_id": predicted["event_id"],
        "truth_event_id": target["event_id"],
        "matched_frames": matched_frames,
        "union_frames": len(union_frames),
        "truth_frames": truth_frames,
        "predicted_frames": pred_frames,
        "temporal_iou": matched_frames / len(union_frames) if union_frames else 0.0,
        "truth_frame_coverage": matched_frames / truth_frames,
        "predicted_frame_precision": matched_frames / pred_frames,
        "mean_frame_iou": float(np.mean(matched_ious)) if matched_ious else None,
        "first_matched_frame": first_matched_frame,
        "latency_frames": None if first_matched_frame is None else first_matched_frame - int(target["frame_start"]),
        "duration_error_frames": int(predicted["duration_frames"]) - int(target["duration_frames"]),
    }


def event_metrics(
    prediction_rows: list[dict[str, str]],
    truth: dict[str, Any],
    *,
    iou_threshold: float = 0.25,
    prediction_source: str = "detections.csv",
) -> dict[str, Any]:
    """Compute event-level precision/recall against synthetic particle events."""

    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0, 1]")

    truth_events = build_events_from_boxes(
        truth_event_boxes(truth),
        prefix="truth",
        iou_threshold=iou_threshold,
    )
    predicted_events = build_events_from_boxes(
        predicted_event_boxes(prediction_rows),
        prefix="pred",
        iou_threshold=iou_threshold,
    )

    candidates: list[tuple[float, int, int, dict[str, Any]]] = []
    for pred_index, predicted in enumerate(predicted_events):
        for truth_index, target in enumerate(truth_events):
            comparison = compare_events(predicted, target, iou_threshold=iou_threshold)
            if comparison["matched_frames"] > 0:
                candidates.append((float(comparison["temporal_iou"]), pred_index, truth_index, comparison))

    matched_predictions: set[int] = set()
    matched_truths: set[int] = set()
    matches: list[dict[str, Any]] = []
    for _score, pred_index, truth_index, comparison in sorted(candidates, reverse=True):
        if pred_index in matched_predictions or truth_index in matched_truths:
            continue
        matched_predictions.add(pred_index)
        matched_truths.add(truth_index)
        matches.append(comparison)

    true_positives = len(matches)
    false_positives = len(predicted_events) - true_positives
    false_negatives = len(truth_events) - true_positives
    precision = true_positives / len(predicted_events) if predicted_events else None
    recall = true_positives / len(truth_events) if truth_events else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    def mean_field(name: str) -> float | None:
        values = [finite_float(match.get(name)) for match in matches]
        finite = [value for value in values if value is not None]
        return None if not finite else float(np.mean(finite))

    return {
        "available": bool(truth_events or predicted_events),
        "iou_threshold": iou_threshold,
        "prediction_source": prediction_source,
        "prediction_rows": len(prediction_rows),
        "truth_events": len(truth_events),
        "predicted_events": len(predicted_events),
        "matched_events": true_positives,
        "false_positive_events": false_positives,
        "false_negative_events": false_negatives,
        "precision": None if precision is None else float(precision),
        "recall": None if recall is None else float(recall),
        "f1": None if f1 is None else float(f1),
        "mean_temporal_iou": mean_field("temporal_iou"),
        "mean_truth_frame_coverage": mean_field("truth_frame_coverage"),
        "mean_predicted_frame_precision": mean_field("predicted_frame_precision"),
        "mean_frame_iou": mean_field("mean_frame_iou"),
        "mean_latency_frames": mean_field("latency_frames"),
        "mean_duration_error_frames": mean_field("duration_error_frames"),
        "matches": matches,
    }


def group_truth_boxes(truth: dict[str, Any]) -> dict[int, list[dict[str, float]]]:
    """Group synthetic particle boxes by source frame."""

    grouped: dict[int, list[dict[str, float]]] = {}
    for particle in truth_particle_rows(truth):
        frame_index = finite_int(particle.get("frame_index"))
        if frame_index is None:
            continue
        try:
            grouped.setdefault(frame_index, []).append(bbox_from_truth(particle))
        except KeyError:
            continue
    return grouped


def group_detection_boxes(detection_rows: list[dict[str, str]]) -> dict[int, list[dict[str, float]]]:
    """Group predicted detection boxes by source frame."""

    grouped: dict[int, list[dict[str, float]]] = {}
    for row in detection_rows:
        frame_index = source_frame_index(row)
        box = bbox_from_detection(row)
        if frame_index is None or box is None:
            continue
        grouped.setdefault(frame_index, []).append(box)
    return grouped


def detection_metrics(
    detection_rows: list[dict[str, str]],
    truth: dict[str, Any],
    *,
    iou_threshold: float = 0.25,
) -> dict[str, Any]:
    """Compute greedy IoU detection precision/recall against synthetic boxes."""

    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0, 1]")

    truth_by_frame = group_truth_boxes(truth)
    pred_by_frame = group_detection_boxes(detection_rows)
    frame_indices = sorted(set(truth_by_frame) | set(pred_by_frame))

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    centroid_errors: list[float] = []
    matched_ious: list[float] = []

    for frame_index in frame_indices:
        truths = truth_by_frame.get(frame_index, [])
        preds = pred_by_frame.get(frame_index, [])
        candidates: list[tuple[float, int, int]] = []
        for pred_index, pred in enumerate(preds):
            for truth_index, target in enumerate(truths):
                candidates.append((bbox_iou(pred, target), pred_index, truth_index))

        matched_preds: set[int] = set()
        matched_truths: set[int] = set()
        for iou, pred_index, truth_index in sorted(candidates, reverse=True):
            if iou < iou_threshold:
                break
            if pred_index in matched_preds or truth_index in matched_truths:
                continue
            matched_preds.add(pred_index)
            matched_truths.add(truth_index)
            true_positives += 1
            matched_ious.append(iou)
            pred_y, pred_x = predicted_center(preds[pred_index])
            truth_y, truth_x = truth_center(truths[truth_index])
            centroid_errors.append(float(math.hypot(pred_y - truth_y, pred_x - truth_x)))

        false_positives += len(preds) - len(matched_preds)
        false_negatives += len(truths) - len(matched_truths)

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else None
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    centroid_stats = summary_errors(centroid_errors, unit="px")
    iou_values = np.asarray(matched_ious, dtype=np.float64)

    return {
        "available": bool(frame_indices),
        "iou_threshold": iou_threshold,
        "truth_boxes": sum(len(items) for items in truth_by_frame.values()),
        "predicted_boxes": sum(len(items) for items in pred_by_frame.values()),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": None if precision is None else float(precision),
        "recall": None if recall is None else float(recall),
        "f1": None if f1 is None else float(f1),
        "mean_matched_iou": None if iou_values.size == 0 else float(np.mean(iou_values)),
        "mean_centroid_error_px": centroid_stats["mean_abs_error_px"],
        "median_centroid_error_px": centroid_stats["median_abs_error_px"],
        "max_centroid_error_px": centroid_stats["max_abs_error_px"],
    }


def choose_velocity_row(velocity_rows: list[dict[str, str]], true_ratio: float | None) -> dict[str, str] | None:
    """Choose the representative velocity row for the simple synthetic benchmark."""

    if not velocity_rows:
        return None

    def key(row: dict[str, str]) -> tuple[int, float]:
        detections = finite_int(row.get("n_detections")) or 0
        ratio = finite_float(row.get("velocity_ratio_y"))
        ratio_penalty = abs(ratio - true_ratio) if ratio is not None and true_ratio is not None else float("inf")
        return detections, -ratio_penalty

    return max(velocity_rows, key=key)


def velocity_metrics(velocity_rows: list[dict[str, str]], truth: dict[str, Any]) -> dict[str, Any]:
    """Compare estimated track velocity against synthetic particle truth."""

    true_velocity = finite_float(
        truth.get(
            "true_particle_velocity_y_px_per_frame",
            truth.get("particle_shift_y_px_per_frame"),
        )
    )
    true_belt_velocity = finite_float(
        truth.get(
            "true_belt_velocity_y_px_per_frame",
            truth.get("belt_shift_px_per_frame"),
        )
    )
    true_ratio = finite_float(truth.get("true_velocity_ratio_y"))
    if true_ratio is None and true_velocity is not None and true_belt_velocity not in (None, 0):
        true_ratio = true_velocity / true_belt_velocity

    representative = choose_velocity_row(velocity_rows, true_ratio)
    if representative is None:
        return {
            "available": False,
            "reason": "No velocity rows found",
            "truth_velocity_y_px_per_frame": true_velocity,
            "truth_velocity_ratio_y": true_ratio,
            "velocity_rows": 0,
        }

    estimated_velocity = finite_float(representative.get("velocity_y_px_per_frame"))
    estimated_ratio = finite_float(representative.get("velocity_ratio_y"))
    return {
        "available": estimated_velocity is not None or estimated_ratio is not None,
        "velocity_rows": len(velocity_rows),
        "representative_track_id": finite_int(representative.get("track_id")),
        "representative_track_detections": finite_int(representative.get("n_detections")),
        "truth_velocity_y_px_per_frame": true_velocity,
        "estimated_velocity_y_px_per_frame": estimated_velocity,
        "velocity_y_error_px_per_frame": None if true_velocity is None or estimated_velocity is None else float(estimated_velocity - true_velocity),
        "truth_velocity_ratio_y": true_ratio,
        "estimated_velocity_ratio_y": estimated_ratio,
        "velocity_ratio_error": None if true_ratio is None or estimated_ratio is None else float(estimated_ratio - true_ratio),
    }


def read_progress_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL progress rows, ignoring malformed lines."""

    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def runtime_metrics(output_dir: Path) -> dict[str, Any]:
    """Extract runtime and memory metrics from standard BeltMap outputs."""

    metadata_path = output_dir / "metadata.json"
    metadata = read_json(metadata_path) if metadata_path.is_file() else {}
    progress_rows = read_progress_jsonl(output_dir / "progress.jsonl")
    elapsed = finite_float(metadata.get("elapsed_s"))
    frames = finite_int(metadata.get("n_images"))
    rss_values = [finite_float(row.get("rss_mb")) for row in progress_rows]
    rss_values = [value for value in rss_values if value is not None]
    fps = None if elapsed in (None, 0) or frames is None else frames / elapsed
    return {
        "available": bool(metadata or progress_rows),
        "elapsed_s": elapsed,
        "frames": frames,
        "frames_per_second": None if fps is None else float(fps),
        "peak_rss_mb": None if not rss_values else float(max(rss_values)),
    }


def compute_benchmark_metrics(
    *,
    output_dir: Path,
    truth_path: Path,
    iou_threshold: float = 0.25,
) -> dict[str, Any]:
    """Compute synthetic ground-truth benchmark metrics for a BeltMap run."""

    truth = read_json(truth_path)
    phase_rows = read_csv_rows(output_dir / "phase_estimates.csv")
    detection_rows = read_csv_rows(output_dir / "detections.csv")
    track_rows = read_csv_rows(output_dir / "tracks.csv")
    velocity_rows = read_csv_rows(output_dir / "velocities.csv")
    metadata_path = output_dir / "metadata.json"
    metadata = read_json(metadata_path) if metadata_path.is_file() else {}
    truth_rows = truth_particle_rows(truth)
    truth_frame_count = (
        len(truth.get("frames"))
        if isinstance(truth.get("frames"), list)
        else finite_int(truth.get("frames"))
    )

    return {
        "benchmark": {
            "type": "synthetic_ground_truth",
            "truth": str(truth_path),
            "output_dir": str(output_dir),
        },
        "case": {
            "frames": truth_frame_count,
            "truth_boxes": len(truth_rows),
            "height": truth.get("height"),
            "width": truth.get("width"),
            "belt_period_px": truth.get("belt_period_px"),
            "belt_shift_px_per_frame": truth.get("belt_shift_px_per_frame"),
            "particle_shift_y_px_per_frame": truth.get("particle_shift_y_px_per_frame"),
            "particle_size_px": truth.get("particle_size_px"),
        },
        "run": {
            "n_images": metadata.get("n_images"),
            "belt_velocity_px_per_frame": metadata.get("belt_velocity_px_per_frame"),
            "belt_map_height_px": metadata.get("belt_map_height_px"),
            "n_phase_estimates": metadata.get("n_phase_estimates"),
            "n_detections": metadata.get("n_detections"),
            "n_tracks": metadata.get("n_tracks"),
            "n_velocity_estimates": metadata.get("n_velocity_estimates"),
        },
        "phase": phase_metrics(phase_rows, truth),
        "belt_map": map_metrics(output_dir, truth_path, truth),
        "detections": detection_metrics(detection_rows, truth, iou_threshold=iou_threshold),
        "events": event_metrics(
            track_rows or detection_rows,
            truth,
            iou_threshold=iou_threshold,
            prediction_source="tracks.csv" if track_rows else "detections.csv",
        ),
        "velocity": velocity_metrics(velocity_rows, truth),
        "runtime": runtime_metrics(output_dir),
    }


def format_value(value: Any, *, digits: int = 4) -> str:
    """Format values compactly for Markdown tables."""

    if value is None:
        return "n/a"
    if isinstance(value, float):
        if abs(value) >= 1000 or (0 < abs(value) < 1e-3):
            return f"{value:.{digits}g}"
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def markdown_report(metrics: dict[str, Any]) -> str:
    """Render benchmark metrics as a compact Markdown report."""

    phase = metrics["phase"]
    belt_map = metrics["belt_map"]
    detections = metrics["detections"]
    events = metrics["events"]
    velocity = metrics["velocity"]
    runtime = metrics["runtime"]

    lines = [
        "# BeltMap synthetic ground-truth benchmark",
        "",
        f"Truth file: `{metrics['benchmark']['truth']}`",
        f"Output directory: `{metrics['benchmark']['output_dir']}`",
        "",
        "## Case",
        "",
        "| Quantity | Value |",
        "| --- | ---: |",
    ]
    for key, value in metrics["case"].items():
        lines.append(f"| `{key}` | {format_value(value)} |")

    lines.extend(["", "## Run summary", "", "| Quantity | Value |", "| --- | ---: |"])
    for key, value in metrics["run"].items():
        lines.append(f"| `{key}` | {format_value(value)} |")

    lines.extend(
        [
            "",
            "## Ground-truth metrics",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| phase RMSE [px] | {format_value(phase.get('rmse_px'))} |",
            f"| phase median abs. error [px] | {format_value(phase.get('median_abs_error_px'))} |",
            f"| phase p95 abs. error [px] | {format_value(phase.get('p95_abs_error_px'))} |",
            f"| belt-map RMSE [gray] | {format_value(belt_map.get('rmse_gray'))} |",
            f"| belt-map best cyclic shift [px] | {format_value(belt_map.get('best_cyclic_shift_px'))} |",
            f"| detection precision | {format_value(detections.get('precision'))} |",
            f"| detection recall | {format_value(detections.get('recall'))} |",
            f"| detection F1 | {format_value(detections.get('f1'))} |",
            f"| mean centroid error [px] | {format_value(detections.get('mean_centroid_error_px'))} |",
            f"| event precision | {format_value(events.get('precision'))} |",
            f"| event recall | {format_value(events.get('recall'))} |",
            f"| event F1 | {format_value(events.get('f1'))} |",
            f"| event prediction source | {format_value(events.get('prediction_source'))} |",
            f"| matched events | {format_value(events.get('matched_events'))} |",
            f"| truth events | {format_value(events.get('truth_events'))} |",
            f"| predicted events | {format_value(events.get('predicted_events'))} |",
            f"| mean event temporal IoU | {format_value(events.get('mean_temporal_iou'))} |",
            f"| mean event truth-frame coverage | {format_value(events.get('mean_truth_frame_coverage'))} |",
            f"| mean event latency [frames] | {format_value(events.get('mean_latency_frames'))} |",
            f"| velocity y error [px/frame] | {format_value(velocity.get('velocity_y_error_px_per_frame'))} |",
            f"| velocity-ratio error | {format_value(velocity.get('velocity_ratio_error'))} |",
            f"| elapsed [s] | {format_value(runtime.get('elapsed_s'))} |",
            f"| frames/s | {format_value(runtime.get('frames_per_second'))} |",
            f"| peak RSS [MB] | {format_value(runtime.get('peak_rss_mb'))} |",
            "",
            "## Notes",
            "",
            "- Phase errors are circular in belt-period pixels.",
            "- Belt-map RMSE is minimized over cyclic vertical shifts, so a constant phase offset",
            "  in the reconstructed map is not counted as a reconstruction error.",
            "- Detection scores use greedy per-frame IoU matching against synthetic particle boxes.",
            "- Event scores use `tracks.csv` when present, falling back to `detections.csv`",
            "  for older outputs without track rows. Rows without event IDs are linked into",
            "  particle events before greedy event matching.",
            "- Velocity metrics use the representative output track with the most detections.",
            "",
        ]
    )

    unavailable = [
        ("belt map", belt_map),
        ("phase", phase),
        ("detections", detections),
        ("events", events),
        ("velocity", velocity),
        ("runtime", runtime),
    ]
    missing = [
        f"- {name}: {section.get('reason', 'not available')}"
        for name, section in unavailable
        if section.get("available") is False
    ]
    if missing:
        lines.extend(["## Missing or unavailable sections", "", *missing, ""])

    return "\n".join(lines)


def write_benchmark_artifacts(
    metrics: dict[str, Any],
    *,
    metrics_path: Path,
    report_path: Path,
) -> BenchmarkArtifacts:
    """Write benchmark JSON and Markdown artifacts."""

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_path.write_text(markdown_report(metrics), encoding="utf-8")
    return BenchmarkArtifacts(metrics=metrics_path, report=report_path)


def generate_benchmark_report(
    *,
    output_dir: Path,
    truth_path: Path,
    metrics_path: Path | None = None,
    report_path: Path | None = None,
    iou_threshold: float = 0.25,
) -> BenchmarkArtifacts:
    """Compute and write synthetic benchmark artifacts."""

    if not output_dir.is_dir():
        raise FileNotFoundError(f"BeltMap output directory does not exist: {output_dir}")
    metrics = compute_benchmark_metrics(
        output_dir=output_dir,
        truth_path=truth_path,
        iou_threshold=iou_threshold,
    )
    return write_benchmark_artifacts(
        metrics,
        metrics_path=metrics_path or output_dir / "benchmark_metrics.json",
        report_path=report_path or output_dir / "benchmark_report.md",
    )
