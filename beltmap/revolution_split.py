"""Train/eval splits by integer belt revolution for ghost diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

REVOLUTION_SPLIT_FRAME_FIELDS = [
    "frame_index",
    "image",
    "revolution_index",
    "split",
    "selected_for_map_training",
]
REVOLUTION_SPLIT_REVOLUTION_FIELDS = [
    "revolution_index",
    "split",
    "n_frames",
    "n_selected_for_map_training",
]
REVOLUTION_SPLIT_DETECTION_SUMMARY_FIELDS = [
    "stage",
    "group",
    "split",
    "revolution_index",
    "n_revolutions",
    "n_frames",
    "n_detections",
    "detections_per_frame",
    "mean_area_px",
    "mean_peak_signal",
]
REVOLUTION_SPLIT_SCORE_SUMMARY_FIELDS = [
    "stage",
    "group",
    "split",
    "revolution_index",
    "n_revolutions",
    "n_frames",
    "n_detections",
    "n_rejected",
    "rejected_fraction",
    "mean_overlap_fraction",
    "mean_artifact_probability",
]


@dataclass(frozen=True)
class RevolutionSplit:
    """Frame-level train/eval assignment derived from belt revolution indices."""

    revolution_by_frame: tuple[int, ...]
    frame_split: tuple[str, ...]
    train_revolutions: tuple[int, ...]
    eval_revolutions: tuple[int, ...]
    train_frame_indices: tuple[int, ...]
    eval_frame_indices: tuple[int, ...]

    @property
    def frame_count(self) -> int:
        return len(self.revolution_by_frame)

    @property
    def revolutions(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.revolution_by_frame)))


def parse_revolution_indices(value: str) -> tuple[int, ...]:
    """Parse comma-separated revolution indices and closed integer ranges.

    Examples
    --------
    ``"1,3,5-7"`` becomes ``(1, 3, 5, 6, 7)``.  Revolution indices are
    expected to be non-negative because they are counted from the first selected
    frame.
    """

    text = value.strip()
    if not text:
        return ()
    result: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, stop_text = token.split("-", 1)
            start = int(start_text.strip())
            stop = int(stop_text.strip())
            if stop < start:
                raise ValueError(
                    "revolution ranges must be increasing; " f"got {token!r}"
                )
            result.extend(range(start, stop + 1))
        else:
            result.append(int(token))
    if any(index < 0 for index in result):
        raise ValueError("revolution indices must be non-negative")
    return tuple(sorted(set(result)))


def build_revolution_split(
    revolution_by_frame: Sequence[int],
    *,
    eval_every: int = 3,
    eval_offset: int = 0,
    eval_revolutions: Sequence[int] = (),
    min_train_revolutions: int = 1,
    min_eval_revolutions: int = 1,
) -> RevolutionSplit:
    """Assign frames to map-training and held-out evaluation revolutions.

    When ``eval_revolutions`` is non-empty it is used explicitly.  Otherwise
    every ``eval_every``-th observed revolution is held out, with
    ``eval_offset`` selecting the modulo class.  All remaining revolutions are
    used as the map-building training pool.
    """

    revolutions = _normalize_revolution_by_frame(revolution_by_frame)
    unique_revolutions = tuple(sorted(set(revolutions)))
    eval_every = _positive_integer_value(eval_every, "eval_every")
    eval_offset = _nonnegative_integer_value(eval_offset, "eval_offset")
    min_train_revolutions = _positive_integer_value(
        min_train_revolutions,
        "min_train_revolutions",
    )
    min_eval_revolutions = _positive_integer_value(
        min_eval_revolutions,
        "min_eval_revolutions",
    )

    explicit_eval = tuple(
        sorted(
            {
                _nonnegative_integer_value(index, "eval_revolutions")
                for index in eval_revolutions
            }
        )
    )
    observed = set(unique_revolutions)
    if explicit_eval:
        missing = sorted(set(explicit_eval) - observed)
        if missing:
            preview = ", ".join(str(index) for index in missing[:8])
            raise ValueError(
                "REVOLUTION_SPLIT_EVAL_REVOLUTIONS contains revolutions not "
                f"present in the selected sequence; first missing: {preview}"
            )
        eval_set = set(explicit_eval)
    else:
        modulo = eval_offset % eval_every
        eval_set = {
            revolution
            for revolution in unique_revolutions
            if revolution % eval_every == modulo
        }
    train_set = set(unique_revolutions) - eval_set
    if len(train_set) < min_train_revolutions:
        raise ValueError(
            "revolution split leaves too few training revolutions: "
            f"{len(train_set)} < {min_train_revolutions}"
        )
    if len(eval_set) < min_eval_revolutions:
        raise ValueError(
            "revolution split leaves too few evaluation revolutions: "
            f"{len(eval_set)} < {min_eval_revolutions}"
        )

    frame_split = tuple(
        "eval" if revolution in eval_set else "train" for revolution in revolutions
    )
    train_frame_indices = tuple(
        index for index, split in enumerate(frame_split) if split == "train"
    )
    eval_frame_indices = tuple(
        index for index, split in enumerate(frame_split) if split == "eval"
    )
    return RevolutionSplit(
        revolution_by_frame=revolutions,
        frame_split=frame_split,
        train_revolutions=tuple(sorted(train_set)),
        eval_revolutions=tuple(sorted(eval_set)),
        train_frame_indices=train_frame_indices,
        eval_frame_indices=eval_frame_indices,
    )


def revolution_split_frame_rows(
    split: RevolutionSplit,
    *,
    image_names: Sequence[str],
    selected_train_frame_indices: Sequence[int] = (),
) -> list[dict]:
    """Return per-frame train/eval assignment rows for CSV output."""

    if len(image_names) != split.frame_count:
        raise ValueError("image_names must match the split frame count")
    selected = {
        _nonnegative_integer_value(index, "selected_train_frame_indices")
        for index in selected_train_frame_indices
    }
    return [
        {
            "frame_index": index,
            "image": image_names[index],
            "revolution_index": split.revolution_by_frame[index],
            "split": split.frame_split[index],
            "selected_for_map_training": index in selected,
        }
        for index in range(split.frame_count)
    ]


def revolution_split_revolution_rows(
    split: RevolutionSplit,
    *,
    selected_train_frame_indices: Sequence[int] = (),
) -> list[dict]:
    """Return per-revolution split composition rows for CSV output."""

    selected = {
        _nonnegative_integer_value(index, "selected_train_frame_indices")
        for index in selected_train_frame_indices
    }
    rows: list[dict] = []
    for revolution in split.revolutions:
        frame_indices = _frame_indices_for_revolution(split, revolution)
        rows.append(
            {
                "revolution_index": revolution,
                "split": split.frame_split[frame_indices[0]],
                "n_frames": len(frame_indices),
                "n_selected_for_map_training": sum(
                    1 for index in frame_indices if index in selected
                ),
            }
        )
    return rows


def revolution_split_detection_summary_rows(
    split: RevolutionSplit,
    detections_by_frame: Sequence[Sequence[object]],
    *,
    stage: str,
) -> list[dict]:
    """Summarize detection rates by split and individual revolution."""

    _validate_frame_sequence_length(split, detections_by_frame, "detections_by_frame")
    rows: list[dict] = []
    for split_label, frame_indices, revolutions in _split_groups(split):
        detections = _items_for_frames(detections_by_frame, frame_indices)
        rows.append(
            _detection_summary_row(
                stage=stage,
                group="split",
                split_label=split_label,
                revolution_index="",
                n_revolutions=len(revolutions),
                frame_indices=frame_indices,
                detections=detections,
            )
        )
    for revolution in split.revolutions:
        frame_indices = _frame_indices_for_revolution(split, revolution)
        detections = _items_for_frames(detections_by_frame, frame_indices)
        rows.append(
            _detection_summary_row(
                stage=stage,
                group="revolution",
                split_label=split.frame_split[frame_indices[0]],
                revolution_index=revolution,
                n_revolutions=1,
                frame_indices=frame_indices,
                detections=detections,
            )
        )
    return rows


def revolution_split_score_summary_rows(
    split: RevolutionSplit,
    scores_by_frame: Sequence[Sequence[object]],
    *,
    stage: str,
) -> list[dict]:
    """Summarize train-artifact-map scores by split and revolution."""

    _validate_frame_sequence_length(split, scores_by_frame, "scores_by_frame")
    rows: list[dict] = []
    for split_label, frame_indices, revolutions in _split_groups(split):
        scores = _items_for_frames(scores_by_frame, frame_indices)
        rows.append(
            _score_summary_row(
                stage=stage,
                group="split",
                split_label=split_label,
                revolution_index="",
                n_revolutions=len(revolutions),
                frame_indices=frame_indices,
                scores=scores,
            )
        )
    for revolution in split.revolutions:
        frame_indices = _frame_indices_for_revolution(split, revolution)
        scores = _items_for_frames(scores_by_frame, frame_indices)
        rows.append(
            _score_summary_row(
                stage=stage,
                group="revolution",
                split_label=split.frame_split[frame_indices[0]],
                revolution_index=revolution,
                n_revolutions=1,
                frame_indices=frame_indices,
                scores=scores,
            )
        )
    return rows


def _normalize_revolution_by_frame(
    revolution_by_frame: Sequence[int],
) -> tuple[int, ...]:
    arr = np.asarray(revolution_by_frame)
    if arr.ndim != 1:
        raise ValueError("revolution_by_frame must be one-dimensional")
    if arr.size == 0:
        raise ValueError("revolution_by_frame must contain at least one frame")
    return tuple(
        _nonnegative_integer_value(value, "revolution indices")
        for value in arr.tolist()
    )


def _frame_indices_for_revolution(
    split: RevolutionSplit, revolution: int
) -> tuple[int, ...]:
    return tuple(
        index
        for index, frame_revolution in enumerate(split.revolution_by_frame)
        if frame_revolution == revolution
    )


def _frame_indices_for_split(
    split: RevolutionSplit, split_label: str
) -> tuple[int, ...]:
    return tuple(
        index for index, value in enumerate(split.frame_split) if value == split_label
    )


def _split_groups(split: RevolutionSplit):
    yield "train", _frame_indices_for_split(split, "train"), split.train_revolutions
    yield "eval", _frame_indices_for_split(split, "eval"), split.eval_revolutions


def _items_for_frames(
    items_by_frame: Sequence[Sequence[object]], frame_indices: Sequence[int]
) -> list[object]:
    return [item for index in frame_indices for item in items_by_frame[index]]


def _validate_frame_sequence_length(
    split: RevolutionSplit,
    items_by_frame: Sequence[Sequence[object]],
    name: str,
) -> None:
    if len(items_by_frame) != split.frame_count:
        raise ValueError(f"{name} must match the split frame count")


def _detection_summary_row(
    *,
    stage: str,
    group: str,
    split_label: str,
    revolution_index: int | str,
    n_revolutions: int,
    frame_indices: Sequence[int],
    detections: Sequence[object],
) -> dict:
    n_frames = len(frame_indices)
    n_detections = len(detections)
    return {
        "stage": stage,
        "group": group,
        "split": split_label,
        "revolution_index": revolution_index,
        "n_revolutions": n_revolutions,
        "n_frames": n_frames,
        "n_detections": n_detections,
        "detections_per_frame": n_detections / n_frames if n_frames else "",
        "mean_area_px": _finite_mean_attr(detections, "area_px"),
        "mean_peak_signal": _finite_mean_attr(detections, "peak_signal"),
    }


def _score_summary_row(
    *,
    stage: str,
    group: str,
    split_label: str,
    revolution_index: int | str,
    n_revolutions: int,
    frame_indices: Sequence[int],
    scores: Sequence[object],
) -> dict:
    n_frames = len(frame_indices)
    n_scores = len(scores)
    n_rejected = sum(1 for score in scores if bool(getattr(score, "rejected", False)))
    return {
        "stage": stage,
        "group": group,
        "split": split_label,
        "revolution_index": revolution_index,
        "n_revolutions": n_revolutions,
        "n_frames": n_frames,
        "n_detections": n_scores,
        "n_rejected": n_rejected,
        "rejected_fraction": n_rejected / n_scores if n_scores else "",
        "mean_overlap_fraction": _finite_mean_attr(scores, "overlap_fraction"),
        "mean_artifact_probability": _finite_mean_attr(scores, "artifact_probability"),
    }


def _finite_mean_attr(items: Sequence[object], attr: str) -> float | str:
    values: list[float] = []
    for item in items:
        value = getattr(item, attr, None)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            values.append(numeric)
    if not values:
        return ""
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _nonnegative_integer_value(value: int, name: str) -> int:
    parsed = float(value)
    if not np.isfinite(parsed) or not parsed.is_integer() or parsed < 0:
        raise ValueError(f"{name} must be a finite non-negative integer")
    return int(parsed)


def _positive_integer_value(value: int, name: str) -> int:
    parsed = _nonnegative_integer_value(value, name)
    if parsed < 1:
        raise ValueError(f"{name} must be a finite positive integer")
    return parsed
