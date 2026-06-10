from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .benchmark import bbox_iou, finite_float, finite_int, source_frame_index

TRACKLET_ID_KEYS = (
    "tracklet_id",
    "truth_tracklet_id",
    "particle_id",
    "event_id",
    "track_id",
    "id",
)
PREDICTION_ID_KEYS = ("track_id", "tracklet_id", "particle_id", "event_id", "id")
FRAME_KEYS = ("frame_index", "frame", "image_index")
SCORED_FRAME_KEYS = (
    "scored_frames",
    "labeled_frames",
    "frames",
    "empty_frames",
    "frame_reviews",
)
TRACKLET_CONTAINER_KEYS = ("tracklets", "tracks", "annotations", "particles", "labels")
NESTED_BOX_KEYS = ("boxes", "frames", "detections")
REVIEWED_GROUND_TRUTH_STATUS = "reviewed_ground_truth"
BOX_FIELD_SETS = (
    ("bbox_top", "bbox_left", "bbox_bottom", "bbox_right"),
    ("top", "left", "bottom", "right"),
    ("y_min", "x_min", "y_max", "x_max"),
    ("y1", "x1", "y2", "x2"),
)


@dataclass(frozen=True)
class TrackletBox:
    """One labeled or predicted crop-local box belonging to a tracklet."""

    tracklet_id: str
    frame_index: int
    top: float
    left: float
    bottom: float
    right: float
    centroid_y: float | None = None
    centroid_x: float | None = None


@dataclass(frozen=True)
class TrackletTruth:
    """Sparse real-data tracklet labels and the frames that were manually scored."""

    boxes: list[TrackletBox]
    scored_frames: set[int]
    source: Path
    label_rows: int
    skipped_label_rows: int


@dataclass(frozen=True)
class TrackletPredictions:
    """PyRecEst track rows loaded from ``tracks.csv`` or ``filtered_tracks.csv``."""

    boxes: list[TrackletBox]
    source: Path
    prediction_rows: int
    skipped_prediction_rows: int


@dataclass(frozen=True)
class TrackletMatch:
    """One frame-level truth/prediction match after greedy IoU assignment."""

    frame_index: int
    truth_index: int
    prediction_index: int
    truth_tracklet_id: str
    predicted_track_id: str
    iou: float
    centroid_error_px: float


@dataclass(frozen=True)
class TrackletEvaluationArtifacts:
    """Files written by a sparse real-tracklet evaluation run."""

    metrics: Path
    report: Path
    matches: Path


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV dictionaries from ``path``."""

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def nonempty_value(row: dict[str, Any], key: str) -> Any | None:
    """Return a non-empty field value, treating blanks as missing."""

    value = row.get(key)
    return None if value is None or str(value).strip() == "" else value


def row_frame_index(row: dict[str, Any]) -> int | None:
    """Infer the source frame index used by annotation and BeltMap CSV rows."""

    frame_index = source_frame_index(row)
    if frame_index is not None:
        return frame_index
    for key in FRAME_KEYS:
        frame_index = finite_int(nonempty_value(row, key))
        if frame_index is not None:
            return frame_index
    return None


def parse_frame_set(value: Any) -> set[int]:
    """Parse a scalar, comma-separated string, list, or frame-object list."""

    if value is None:
        return set()
    if isinstance(value, str):
        items: Iterable[Any] = [part.strip() for part in value.split(",")]
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, dict)):
        items = value
    else:
        items = [value]

    frames: set[int] = set()
    for item in items:
        if isinstance(item, dict):
            frame_index = row_frame_index(item)
        else:
            frame_index = finite_int(item)
        if frame_index is not None:
            frames.add(frame_index)
    return frames


def boolean_review_flag(value: Any) -> bool:
    """Parse common JSON review flags without treating ``"false"`` as true."""

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "0", "false", "no", "n"}:
            return False
        if normalized in {"1", "true", "yes", "y"}:
            return True
    return bool(value)


def validate_reviewed_truth_status(data: dict[str, Any]) -> None:
    """Reject tracklet label scaffolds that still declare pending review."""

    status = data.get("status")
    requires_manual_review = data.get("requires_manual_review")
    if status is None and requires_manual_review is None:
        return
    if status != REVIEWED_GROUND_TRUTH_STATUS or boolean_review_flag(
        requires_manual_review
    ):
        raise ValueError(
            "tracklet truth JSON is not reviewed ground truth; set status="
            f"{REVIEWED_GROUND_TRUTH_STATUS!r} and requires_manual_review=false "
            "only after all scored frames have been reviewed"
        )


def tracklet_id_from_row(
    row: dict[str, Any],
    *,
    keys: tuple[str, ...] = TRACKLET_ID_KEYS,
) -> str | None:
    """Return the first explicit non-empty tracklet identifier from ``row``."""

    for key in keys:
        value = nonempty_value(row, key)
        if value is not None:
            return str(value)
    return None


def bbox_tuple_from_row(row: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """Return a half-open bbox tuple from a supported annotation layout."""

    for field_set in BOX_FIELD_SETS:
        values = [finite_float(nonempty_value(row, field)) for field in field_set]
        if any(value is None for value in values):
            continue
        top, left, bottom, right = values
        assert top is not None and left is not None
        assert bottom is not None and right is not None
        if bottom <= top or right <= left:
            return None
        return float(top), float(left), float(bottom), float(right)
    return None


def tracklet_box_from_row(
    row: dict[str, Any],
    *,
    default_tracklet_id: str | None = None,
    id_keys: tuple[str, ...] = TRACKLET_ID_KEYS,
    require_tracklet_id: bool = True,
) -> TrackletBox | None:
    """Convert one annotation or PyRecEst track row to a ``TrackletBox``."""

    bbox = bbox_tuple_from_row(row)
    if bbox is None:
        return None
    frame_index = row_frame_index(row)
    if frame_index is None:
        return None
    tracklet_id = tracklet_id_from_row(row, keys=id_keys) or default_tracklet_id
    if tracklet_id is None:
        if require_tracklet_id:
            raise ValueError(
                "tracklet box rows must include one of: " + ", ".join(id_keys)
            )
        tracklet_id = f"anonymous:{frame_index}"
    top, left, bottom, right = bbox
    return TrackletBox(
        tracklet_id=str(tracklet_id),
        frame_index=frame_index,
        top=top,
        left=left,
        bottom=bottom,
        right=right,
        centroid_y=finite_float(row.get("y")),
        centroid_x=finite_float(row.get("x")),
    )


def rows_from_tracklet_container(items: Any) -> list[dict[str, Any]]:
    """Flatten JSON tracklet containers into one row per labeled box."""

    if not isinstance(items, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        parent_tracklet_id = tracklet_id_from_row(item)
        nested_rows: list[Any] | None = None
        for key in NESTED_BOX_KEYS:
            nested = item.get(key)
            if isinstance(nested, list):
                nested_rows = nested
                break
        if nested_rows is None:
            rows.append(dict(item))
            continue
        for child in nested_rows:
            if not isinstance(child, dict):
                continue
            row = dict(child)
            if parent_tracklet_id is not None and tracklet_id_from_row(row) is None:
                row["tracklet_id"] = parent_tracklet_id
            rows.append(row)
    return rows


def rows_from_frame_container(items: Any) -> tuple[list[dict[str, Any]], set[int]]:
    """Flatten a JSON ``frames`` container into box rows and scored frames."""

    if not isinstance(items, list):
        return [], parse_frame_set(items)
    if not any(isinstance(item, dict) for item in items):
        return [], parse_frame_set(items)

    rows: list[dict[str, Any]] = []
    scored_frames: set[int] = set()
    for frame in items:
        if not isinstance(frame, dict):
            continue
        frame_index = row_frame_index(frame)
        if frame_index is not None:
            scored_frames.add(frame_index)
        nested = frame.get("boxes") or frame.get("detections") or frame.get("tracklets")
        if isinstance(nested, list):
            for child in nested:
                if not isinstance(child, dict):
                    continue
                row = dict(child)
                if frame_index is not None and row_frame_index(row) is None:
                    row["frame_index"] = frame_index
                rows.append(row)
        elif bbox_tuple_from_row(frame) is not None:
            rows.append(dict(frame))
    return rows, scored_frames


def label_rows_from_json(data: Any) -> tuple[list[dict[str, Any]], set[int]]:
    """Extract sparse tracklet rows and scored-frame indices from JSON labels."""

    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)], set()
    if not isinstance(data, dict):
        raise ValueError("tracklet JSON must be an object or a list of objects")

    rows: list[dict[str, Any]] = []
    scored_frames: set[int] = set()
    for key in SCORED_FRAME_KEYS:
        scored_frames.update(parse_frame_set(data.get(key)))

    frame_rows, frame_scored = rows_from_frame_container(data.get("frames"))
    rows.extend(frame_rows)
    scored_frames.update(frame_scored)

    for key in TRACKLET_CONTAINER_KEYS:
        rows.extend(rows_from_tracklet_container(data.get(key)))

    if not rows and bbox_tuple_from_row(data) is not None:
        rows.append(data)
    return rows, scored_frames


def load_tracklet_truth(path: Path) -> TrackletTruth:
    """Load sparse real-data tracklet labels from CSV or JSON.

    CSV rows require ``tracklet_id`` (or ``particle_id``/``event_id``/``track_id``),
    ``frame_index``, and crop-local bbox columns. A row with only ``frame_index``
    marks an explicitly scored empty frame.
    """

    if not path.is_file():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".csv":
        rows: list[dict[str, Any]] = list(read_csv_rows(path))
        scored_frames: set[int] = set()
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            validate_reviewed_truth_status(data)
        rows, scored_frames = label_rows_from_json(data)
    else:
        raise ValueError("tracklet labels must be a CSV or JSON file")

    boxes: list[TrackletBox] = []
    skipped_rows = 0
    for row in rows:
        frame_index = row_frame_index(row)
        if frame_index is not None:
            scored_frames.add(frame_index)
        if bbox_tuple_from_row(row) is None:
            skipped_rows += 1
            continue
        box = tracklet_box_from_row(row, require_tracklet_id=True)
        if box is None:
            skipped_rows += 1
            continue
        boxes.append(box)

    if rows and not boxes and not scored_frames:
        raise ValueError(
            "tracklet labels did not contain usable frame indices or bounding boxes"
        )

    return TrackletTruth(
        boxes=boxes,
        scored_frames=scored_frames,
        source=path,
        label_rows=len(rows),
        skipped_label_rows=skipped_rows,
    )


def load_tracklet_predictions(path: Path) -> TrackletPredictions:
    """Load PyRecEst track rows from ``tracks.csv`` or ``filtered_tracks.csv``."""

    if not path.is_file():
        raise FileNotFoundError(path)
    rows = read_csv_rows(path)
    boxes: list[TrackletBox] = []
    skipped_rows = 0
    for row in rows:
        box = tracklet_box_from_row(
            row,
            id_keys=PREDICTION_ID_KEYS,
            require_tracklet_id=True,
        )
        if box is None:
            skipped_rows += 1
            continue
        boxes.append(box)
    return TrackletPredictions(
        boxes=boxes,
        source=path,
        prediction_rows=len(rows),
        skipped_prediction_rows=skipped_rows,
    )


def box_dict(box: TrackletBox) -> dict[str, float]:
    """Convert a ``TrackletBox`` to the bbox dictionary used by ``bbox_iou``."""

    result = {
        "top": box.top,
        "left": box.left,
        "bottom": box.bottom,
        "right": box.right,
    }
    if box.centroid_y is not None and box.centroid_x is not None:
        result["centroid_y"] = box.centroid_y
        result["centroid_x"] = box.centroid_x
    return result


def box_center(box: TrackletBox) -> tuple[float, float]:
    """Return centroid if available, otherwise the center of the half-open box."""

    if box.centroid_y is not None and box.centroid_x is not None:
        return box.centroid_y, box.centroid_x
    return 0.5 * (box.top + box.bottom - 1.0), 0.5 * (box.left + box.right - 1.0)


def center_distance_px(a: TrackletBox, b: TrackletBox) -> float:
    """Return Euclidean center distance in crop-local pixels."""

    ay, ax = box_center(a)
    by, bx = box_center(b)
    return float(math.hypot(ay - by, ax - bx))


def group_indices_by_frame(boxes: list[TrackletBox]) -> dict[int, list[int]]:
    """Group box indices by frame index."""

    grouped: dict[int, list[int]] = {}
    for index, box in enumerate(boxes):
        grouped.setdefault(box.frame_index, []).append(index)
    return grouped


def greedy_frame_matches(
    truth_boxes: list[TrackletBox],
    prediction_boxes: list[TrackletBox],
    *,
    scored_frames: set[int],
    iou_threshold: float,
) -> tuple[list[TrackletMatch], set[int], set[int]]:
    """Greedily match truth and predicted boxes frame-by-frame by descending IoU."""

    truth_by_frame = group_indices_by_frame(truth_boxes)
    prediction_by_frame = group_indices_by_frame(prediction_boxes)
    unmatched_truth = set(range(len(truth_boxes)))
    unmatched_predictions = set(range(len(prediction_boxes)))
    matches: list[TrackletMatch] = []

    for frame_index in sorted(scored_frames):
        candidates: list[tuple[float, int, int]] = []
        for truth_index in truth_by_frame.get(frame_index, []):
            truth = box_dict(truth_boxes[truth_index])
            for prediction_index in prediction_by_frame.get(frame_index, []):
                prediction = box_dict(prediction_boxes[prediction_index])
                candidates.append((bbox_iou(prediction, truth), truth_index, prediction_index))

        matched_truth_in_frame: set[int] = set()
        matched_prediction_in_frame: set[int] = set()
        for iou, truth_index, prediction_index in sorted(candidates, reverse=True):
            if iou < iou_threshold:
                break
            if truth_index in matched_truth_in_frame or prediction_index in matched_prediction_in_frame:
                continue
            matched_truth_in_frame.add(truth_index)
            matched_prediction_in_frame.add(prediction_index)
            unmatched_truth.discard(truth_index)
            unmatched_predictions.discard(prediction_index)
            truth_box = truth_boxes[truth_index]
            prediction_box = prediction_boxes[prediction_index]
            matches.append(
                TrackletMatch(
                    frame_index=frame_index,
                    truth_index=truth_index,
                    prediction_index=prediction_index,
                    truth_tracklet_id=truth_box.tracklet_id,
                    predicted_track_id=prediction_box.tracklet_id,
                    iou=float(iou),
                    centroid_error_px=center_distance_px(prediction_box, truth_box),
                )
            )

    return matches, unmatched_truth, unmatched_predictions


def mean_or_none(values: Iterable[float]) -> float | None:
    """Return the finite mean of ``values`` or ``None`` for an empty input."""

    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return None if arr.size == 0 else float(np.mean(arr))


def f1_score(precision: float | None, recall: float | None) -> float | None:
    """Return harmonic mean of precision and recall when both are defined."""

    if precision is None or recall is None:
        return None
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def ratio_or_none(numerator: int, denominator: int) -> float | None:
    """Return ``numerator / denominator`` or ``None`` for a zero denominator."""

    return None if denominator == 0 else float(numerator / denominator)


def detection_precision(true_positives: int, false_positives: int, false_negatives: int) -> float | None:
    """Precision with a useful value for perfectly empty scored frames."""

    denominator = true_positives + false_positives
    if denominator:
        return float(true_positives / denominator)
    return 1.0 if false_negatives == 0 else 0.0


def detection_recall(true_positives: int, false_positives: int, false_negatives: int) -> float | None:
    """Recall with a useful value for perfectly empty scored frames."""

    denominator = true_positives + false_negatives
    if denominator:
        return float(true_positives / denominator)
    return 1.0 if false_positives == 0 else None


def tracklet_fragmentation(
    truth_boxes: list[TrackletBox],
    matches: list[TrackletMatch],
) -> tuple[int, int]:
    """Return truth tracklets with gaps after first match and total extra fragments."""

    matched_prediction_by_truth_index = {
        match.truth_index: match.predicted_track_id for match in matches
    }
    truth_indices_by_tracklet: dict[str, list[int]] = {}
    for index, box in enumerate(truth_boxes):
        truth_indices_by_tracklet.setdefault(box.tracklet_id, []).append(index)

    fragmented_tracklets = 0
    extra_fragments = 0
    for indices in truth_indices_by_tracklet.values():
        ordered = sorted(indices, key=lambda item: truth_boxes[item].frame_index)
        matched_segments = 0
        in_segment = False
        for truth_index in ordered:
            is_matched = truth_index in matched_prediction_by_truth_index
            if is_matched and not in_segment:
                matched_segments += 1
                in_segment = True
            elif not is_matched:
                in_segment = False
        if matched_segments > 1:
            fragmented_tracklets += 1
            extra_fragments += matched_segments - 1
    return fragmented_tracklets, extra_fragments


def identity_switch_summary(matches: list[TrackletMatch]) -> tuple[int, int]:
    """Count truth tracklets whose matched PyRecEst ID changes over time."""

    matches_by_truth: dict[str, list[TrackletMatch]] = {}
    for match in matches:
        matches_by_truth.setdefault(match.truth_tracklet_id, []).append(match)

    switched_tracklets = 0
    identity_switches = 0
    for truth_matches in matches_by_truth.values():
        previous_prediction: str | None = None
        switched = False
        for match in sorted(truth_matches, key=lambda item: item.frame_index):
            if previous_prediction is not None and match.predicted_track_id != previous_prediction:
                identity_switches += 1
                switched = True
            previous_prediction = match.predicted_track_id
        if switched:
            switched_tracklets += 1
    return identity_switches, switched_tracklets


def evaluate_tracklets(
    truth: TrackletTruth,
    predictions: TrackletPredictions,
    *,
    iou_threshold: float = 0.25,
) -> tuple[dict[str, Any], list[TrackletMatch], set[int], set[int]]:
    """Compute sparse real-tracklet metrics from manual labels and PyRecEst tracks.

    The returned ``hota``, ``det_a``, ``ass_a``, and ``loc_a`` are a local
    HOTA-style summary for sparse labels, not a byte-for-byte TrackEval output.
    Predictions outside manually scored frames are ignored.
    """

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")

    scored_frames = set(truth.scored_frames)
    scored_frames.update(box.frame_index for box in truth.boxes)
    truth_boxes = [box for box in truth.boxes if box.frame_index in scored_frames]
    prediction_boxes = [box for box in predictions.boxes if box.frame_index in scored_frames]

    matches, unmatched_truth, unmatched_predictions = greedy_frame_matches(
        truth_boxes,
        prediction_boxes,
        scored_frames=scored_frames,
        iou_threshold=iou_threshold,
    )

    true_positives = len(matches)
    false_positives = len(unmatched_predictions)
    false_negatives = len(unmatched_truth)
    det_denominator = true_positives + false_positives + false_negatives
    det_a = None if det_denominator == 0 else float(true_positives / det_denominator)
    precision = detection_precision(true_positives, false_positives, false_negatives)
    recall = detection_recall(true_positives, false_positives, false_negatives)

    truth_tracklets = {box.tracklet_id for box in truth_boxes}
    predicted_tracklets = {box.tracklet_id for box in prediction_boxes}
    matched_truth_tracklets = {match.truth_tracklet_id for match in matches}
    matched_predicted_tracklets = {match.predicted_track_id for match in matches}

    truth_box_counts: dict[str, int] = {}
    predicted_box_counts: dict[str, int] = {}
    pair_counts: dict[tuple[str, str], int] = {}
    for box in truth_boxes:
        truth_box_counts[box.tracklet_id] = truth_box_counts.get(box.tracklet_id, 0) + 1
    for box in prediction_boxes:
        predicted_box_counts[box.tracklet_id] = predicted_box_counts.get(box.tracklet_id, 0) + 1
    for match in matches:
        key = (match.truth_tracklet_id, match.predicted_track_id)
        pair_counts[key] = pair_counts.get(key, 0) + 1

    association_scores: list[float] = []
    association_precision_scores: list[float] = []
    association_recall_scores: list[float] = []
    for match in matches:
        key = (match.truth_tracklet_id, match.predicted_track_id)
        tpa = pair_counts[key]
        fna = truth_box_counts[match.truth_tracklet_id] - tpa
        fpa = predicted_box_counts[match.predicted_track_id] - tpa
        association_scores.append(tpa / (tpa + fna + fpa))
        association_recall_scores.append(tpa / (tpa + fna))
        association_precision_scores.append(tpa / (tpa + fpa))

    ass_a = mean_or_none(association_scores)
    ass_pr = mean_or_none(association_precision_scores)
    ass_re = mean_or_none(association_recall_scores)
    loc_a = mean_or_none(match.iou for match in matches)
    if det_a is None or ass_a is None:
        hota = None if det_denominator == 0 else 0.0
    else:
        hota = float(math.sqrt(det_a * ass_a))

    identity_switches, tracklets_with_id_switch = identity_switch_summary(matches)
    fragmented_truth_tracklets, extra_truth_fragments = tracklet_fragmentation(
        truth_boxes,
        matches,
    )

    truth_frames_with_boxes = {box.frame_index for box in truth_boxes}
    predictions_on_empty_scored_frames = sum(
        1 for box in prediction_boxes if box.frame_index not in truth_frames_with_boxes
    )

    mean_truth_coverage = mean_or_none(
        ratio_or_none(
            sum(1 for match in matches if match.truth_tracklet_id == truth_id),
            count,
        )
        for truth_id, count in truth_box_counts.items()
        if count > 0
    )
    mean_predicted_purity = mean_or_none(
        ratio_or_none(
            max(
                (count for (truth_id, pred_id), count in pair_counts.items() if pred_id == predicted_id),
                default=0,
            ),
            count,
        )
        for predicted_id, count in predicted_box_counts.items()
        if count > 0
    )

    metrics: dict[str, Any] = {
        "available": bool(scored_frames),
        "iou_threshold": float(iou_threshold),
        "scored_frames": len(scored_frames),
        "truth_boxes": len(truth_boxes),
        "predicted_boxes": len(prediction_boxes),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
        "det_a": det_a,
        "ass_a": ass_a,
        "ass_pr": ass_pr,
        "ass_re": ass_re,
        "loc_a": loc_a,
        "hota": hota,
        "mean_matched_iou": loc_a,
        "mean_centroid_error_px": mean_or_none(match.centroid_error_px for match in matches),
        "truth_tracklets": len(truth_tracklets),
        "predicted_tracklets": len(predicted_tracklets),
        "matched_truth_tracklets": len(matched_truth_tracklets),
        "matched_predicted_tracklets": len(matched_predicted_tracklets),
        "false_positive_tracklets": len(predicted_tracklets - matched_predicted_tracklets),
        "false_negative_tracklets": len(truth_tracklets - matched_truth_tracklets),
        "tracklet_precision": ratio_or_none(len(matched_predicted_tracklets), len(predicted_tracklets)),
        "tracklet_recall": ratio_or_none(len(matched_truth_tracklets), len(truth_tracklets)),
        "identity_switches": identity_switches,
        "tracklets_with_id_switch": tracklets_with_id_switch,
        "fragmented_truth_tracklets": fragmented_truth_tracklets,
        "extra_truth_fragments": extra_truth_fragments,
        "merged_prediction_tracklets": sum(
            1
            for predicted_id in predicted_tracklets
            if len({truth_id for truth_id, pred_id in pair_counts if pred_id == predicted_id}) > 1
        ),
        "split_truth_tracklets": sum(
            1
            for truth_id in truth_tracklets
            if len({pred_id for matched_truth_id, pred_id in pair_counts if matched_truth_id == truth_id}) > 1
        ),
        "mean_truth_tracklet_frame_coverage": mean_truth_coverage,
        "mean_predicted_tracklet_purity": mean_predicted_purity,
        "false_positives_on_empty_scored_frames": predictions_on_empty_scored_frames,
        "false_positives_per_scored_frame": ratio_or_none(false_positives, len(scored_frames)),
        "prediction_rows": predictions.prediction_rows,
        "skipped_prediction_rows": predictions.skipped_prediction_rows,
        "label_rows": truth.label_rows,
        "skipped_label_rows": truth.skipped_label_rows,
        "truth_source": str(truth.source),
        "prediction_source": str(predictions.source),
    }
    return metrics, matches, unmatched_truth, unmatched_predictions


def format_metric_value(value: Any) -> str:
    """Format metric values for Markdown output."""

    if value is None:
        return "n/a"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def box_csv_fields(prefix: str, box: TrackletBox | None) -> dict[str, Any]:
    """Return prefixed bbox fields for match CSV output."""

    if box is None:
        return {
            f"{prefix}_bbox_top": "",
            f"{prefix}_bbox_left": "",
            f"{prefix}_bbox_bottom": "",
            f"{prefix}_bbox_right": "",
        }
    return {
        f"{prefix}_bbox_top": box.top,
        f"{prefix}_bbox_left": box.left,
        f"{prefix}_bbox_bottom": box.bottom,
        f"{prefix}_bbox_right": box.right,
    }


def match_csv_rows(
    truth_boxes: list[TrackletBox],
    prediction_boxes: list[TrackletBox],
    matches: list[TrackletMatch],
    unmatched_truth: set[int],
    unmatched_predictions: set[int],
) -> list[dict[str, Any]]:
    """Build rows for a TP/FP/FN match-inspection CSV."""

    rows: list[dict[str, Any]] = []
    for match in matches:
        truth_box = truth_boxes[match.truth_index]
        prediction_box = prediction_boxes[match.prediction_index]
        rows.append(
            {
                "frame_index": match.frame_index,
                "status": "true_positive",
                "truth_tracklet_id": match.truth_tracklet_id,
                "predicted_track_id": match.predicted_track_id,
                "iou": match.iou,
                "centroid_error_px": match.centroid_error_px,
                **box_csv_fields("truth", truth_box),
                **box_csv_fields("predicted", prediction_box),
            }
        )
    for truth_index in sorted(unmatched_truth, key=lambda item: (truth_boxes[item].frame_index, truth_boxes[item].tracklet_id)):
        truth_box = truth_boxes[truth_index]
        rows.append(
            {
                "frame_index": truth_box.frame_index,
                "status": "false_negative",
                "truth_tracklet_id": truth_box.tracklet_id,
                "predicted_track_id": "",
                "iou": "",
                "centroid_error_px": "",
                **box_csv_fields("truth", truth_box),
                **box_csv_fields("predicted", None),
            }
        )
    for prediction_index in sorted(
        unmatched_predictions,
        key=lambda item: (prediction_boxes[item].frame_index, prediction_boxes[item].tracklet_id),
    ):
        prediction_box = prediction_boxes[prediction_index]
        rows.append(
            {
                "frame_index": prediction_box.frame_index,
                "status": "false_positive",
                "truth_tracklet_id": "",
                "predicted_track_id": prediction_box.tracklet_id,
                "iou": "",
                "centroid_error_px": "",
                **box_csv_fields("truth", None),
                **box_csv_fields("predicted", prediction_box),
            }
        )
    return sorted(rows, key=lambda item: (int(item["frame_index"]), str(item["status"]), str(item["truth_tracklet_id"]), str(item["predicted_track_id"])))


def write_matches_csv(
    path: Path,
    truth_boxes: list[TrackletBox],
    prediction_boxes: list[TrackletBox],
    matches: list[TrackletMatch],
    unmatched_truth: set[int],
    unmatched_predictions: set[int],
) -> None:
    """Write frame-level TP/FP/FN assignments for manual review."""

    rows = match_csv_rows(
        truth_boxes,
        prediction_boxes,
        matches,
        unmatched_truth,
        unmatched_predictions,
    )
    fieldnames = [
        "frame_index",
        "status",
        "truth_tracklet_id",
        "predicted_track_id",
        "iou",
        "centroid_error_px",
        "truth_bbox_top",
        "truth_bbox_left",
        "truth_bbox_bottom",
        "truth_bbox_right",
        "predicted_bbox_top",
        "predicted_bbox_left",
        "predicted_bbox_bottom",
        "predicted_bbox_right",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_tracklet_report(path: Path, metrics: dict[str, Any]) -> None:
    """Write a compact Markdown report for sparse tracklet evaluation."""

    rows = [
        ("scored frames", "scored_frames"),
        ("truth boxes", "truth_boxes"),
        ("predicted boxes", "predicted_boxes"),
        ("true positives", "true_positives"),
        ("false positives", "false_positives"),
        ("false negatives", "false_negatives"),
        ("precision", "precision"),
        ("recall", "recall"),
        ("F1", "f1"),
        ("DetA", "det_a"),
        ("AssA", "ass_a"),
        ("LocA", "loc_a"),
        ("HOTA-style", "hota"),
        ("identity switches", "identity_switches"),
        ("fragmented truth tracklets", "fragmented_truth_tracklets"),
        ("false-positive tracklets", "false_positive_tracklets"),
        ("false positives on empty scored frames", "false_positives_on_empty_scored_frames"),
    ]
    lines = [
        "# Sparse tracklet evaluation",
        "",
        f"Truth labels: `{metrics.get('truth_source')}`",
        f"Predicted tracks: `{metrics.get('prediction_source')}`",
        f"IoU threshold: {format_metric_value(metrics.get('iou_threshold'))}",
        "",
        "The HOTA-style values are computed on sparse, manually scored frames only. "
        "Predicted PyRecEst track rows outside those frames are ignored.",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for label, key in rows:
        lines.append(f"| {label} | {format_metric_value(metrics.get(key))} |")
    lines.extend(
        [
            "",
            "Use `tracklet_matches.csv` to inspect every frame-level true positive, "
            "false positive, and false negative assignment.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def default_prediction_path(output_dir: Path) -> Path:
    """Prefer filtered PyRecEst tracks, falling back to raw tracker trajectories."""

    filtered = output_dir / "filtered_tracks.csv"
    if filtered.is_file():
        return filtered
    return output_dir / "tracks.csv"


def generate_tracklet_evaluation_report(
    *,
    output_dir: Path,
    truth_path: Path,
    prediction_path: Path | None = None,
    metrics_path: Path | None = None,
    report_path: Path | None = None,
    matches_path: Path | None = None,
    iou_threshold: float = 0.25,
) -> TrackletEvaluationArtifacts:
    """Load sparse tracklet annotations, score PyRecEst tracks, and write artifacts."""

    output_dir = Path(output_dir)
    prediction_path = default_prediction_path(output_dir) if prediction_path is None else Path(prediction_path)
    metrics_path = output_dir / "tracklet_metrics.json" if metrics_path is None else Path(metrics_path)
    report_path = output_dir / "tracklet_report.md" if report_path is None else Path(report_path)
    matches_path = output_dir / "tracklet_matches.csv" if matches_path is None else Path(matches_path)

    truth = load_tracklet_truth(Path(truth_path))
    predictions = load_tracklet_predictions(prediction_path)
    metrics, matches, unmatched_truth, unmatched_predictions = evaluate_tracklets(
        truth,
        predictions,
        iou_threshold=iou_threshold,
    )

    scored_frames = sorted(set(truth.scored_frames) | {box.frame_index for box in truth.boxes})
    metrics_payload = {
        "tracklet_evaluation": {
            "type": "sparse_real_tracklet",
            "truth_path": str(truth.source),
            "prediction_path": str(predictions.source),
            "iou_threshold": float(iou_threshold),
            "scored_frames": scored_frames,
            "note": "HOTA-style values are a local sparse-label summary, not official TrackEval output.",
        },
        "summary": metrics,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    write_matches_csv(
        matches_path,
        [box for box in truth.boxes if box.frame_index in set(scored_frames)],
        [box for box in predictions.boxes if box.frame_index in set(scored_frames)],
        matches,
        unmatched_truth,
        unmatched_predictions,
    )
    write_tracklet_report(report_path, metrics)
    return TrackletEvaluationArtifacts(
        metrics=metrics_path,
        report=report_path,
        matches=matches_path,
    )
