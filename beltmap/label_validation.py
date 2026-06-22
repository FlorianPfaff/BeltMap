from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

REVIEWED_GROUND_TRUTH_STATUS = "reviewed_ground_truth"
_PARTICLE_KEYS = ("particles", "annotations", "labels", "detections")
_REVIEW_KEYS = ("frame_reviews", "review_frames")
_FRAME_KEYS = ("frame_index", "frame", "image_index")
_BBOX_FIELD_SETS = (
    ("top", "left", "bottom", "right"),
    ("bbox_top", "bbox_left", "bbox_bottom", "bbox_right"),
    ("y_min", "x_min", "y_max", "x_max"),
    ("y1", "x1", "y2", "x2"),
)
_EVENT_ID_KEYS = ("event_id", "particle_id", "track_id", "id")
_EMPTY_REVIEW_STATUSES = {"reviewed_empty", "empty", "confirmed_empty", "no_particle", "no_particles"}
_PARTICLE_REVIEW_STATUSES = {"reviewed_with_particles", "reviewed_particles", "particles"}
_NEEDS_REVIEW_STATUSES = {"needs_review", "pending", "unreviewed", "todo", ""}
_TRUE_STRINGS = {"true", "1", "yes", "y"}
_FALSE_STRINGS = {"false", "0", "no", "n", ""}


@dataclass(frozen=True)
class LabelParticle:
    """One particle box from a label JSON."""

    frame_index: int
    top: float
    left: float
    bottom: float
    right: float
    event_id: str | None = None


@dataclass
class LabelValidationReport:
    """Machine-readable truth-label state summary."""

    truth_path: Path
    status: str | None
    requires_manual_review: bool | None
    n_scored_frames: int
    n_particle_boxes: int
    n_empty_frames: int
    n_frame_reviews: int
    n_needs_review: int
    n_reviewed_with_particles: int
    n_reviewed_empty: int
    n_unaccounted_scored_frames: int
    is_valid_for_metrics: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "truth_path": str(self.truth_path),
            "status": self.status,
            "requires_manual_review": self.requires_manual_review,
            "n_scored_frames": self.n_scored_frames,
            "n_particle_boxes": self.n_particle_boxes,
            "n_empty_frames": self.n_empty_frames,
            "n_frame_reviews": self.n_frame_reviews,
            "n_needs_review": self.n_needs_review,
            "n_reviewed_with_particles": self.n_reviewed_with_particles,
            "n_reviewed_empty": self.n_reviewed_empty,
            "n_unaccounted_scored_frames": self.n_unaccounted_scored_frames,
            "is_valid_for_metrics": self.is_valid_for_metrics,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }

    def format_text(self) -> str:
        lines = [
            f"Truth path: {self.truth_path}",
            f"status: {self.status if self.status is not None else 'n/a'}",
            f"requires_manual_review: {self.requires_manual_review}",
            f"scored_frames: {self.n_scored_frames}",
            f"particle_boxes: {self.n_particle_boxes}",
            f"empty_frames: {self.n_empty_frames}",
            f"frame_reviews: {self.n_frame_reviews}",
            f"needs_review: {self.n_needs_review}",
            f"reviewed_with_particles: {self.n_reviewed_with_particles}",
            f"reviewed_empty: {self.n_reviewed_empty}",
            f"unaccounted_scored_frames: {self.n_unaccounted_scored_frames}",
            f"is_valid_for_metrics: {self.is_valid_for_metrics}",
        ]
        if self.errors:
            lines.append("errors:")
            lines.extend(f"  - {message}" for message in self.errors)
        if self.warnings:
            lines.append("warnings:")
            lines.extend(f"  - {message}" for message in self.warnings)
        return "\n".join(lines) + "\n"


def finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def finite_int(value: Any) -> int | None:
    parsed = finite_float(value)
    if parsed is None or not parsed.is_integer():
        return None
    return int(parsed)


def bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_STRINGS:
            return True
        if lowered in _FALSE_STRINGS:
            return False
    return None


def frame_index_from_row(row: dict[str, Any]) -> int | None:
    for key in _FRAME_KEYS:
        if key in row:
            frame = finite_int(row.get(key))
            if frame is not None:
                return frame
    return None


def bbox_from_row(row: dict[str, Any]) -> tuple[float, float, float, float] | None:
    for keys in _BBOX_FIELD_SETS:
        if all(key in row for key in keys):
            values = [finite_float(row.get(key)) for key in keys]
            if all(value is not None for value in values):
                top, left, bottom, right = (float(value) for value in values if value is not None)
                return top, left, bottom, right
    return None


def event_id_from_row(row: dict[str, Any]) -> str | None:
    for key in _EVENT_ID_KEYS:
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return str(value)
    return None


def first_list_payload(payload: dict[str, Any], keys: Iterable[str]) -> list[Any]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return []


def parse_particles(payload: dict[str, Any], *, errors: list[str]) -> list[LabelParticle]:
    rows = first_list_payload(payload, _PARTICLE_KEYS)
    particles: list[LabelParticle] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"particle row {index} is not an object")
            continue
        frame_index = frame_index_from_row(row)
        bbox = bbox_from_row(row)
        if frame_index is None:
            errors.append(f"particle row {index} has no valid frame_index")
            continue
        if bbox is None:
            errors.append(f"particle row {index} has no valid bbox")
            continue
        top, left, bottom, right = bbox
        if not (bottom > top and right > left):
            errors.append(
                f"particle row {index} has non-positive bbox size: "
                f"top={top}, left={left}, bottom={bottom}, right={right}"
            )
            continue
        particles.append(
            LabelParticle(
                frame_index=frame_index,
                top=top,
                left=left,
                bottom=bottom,
                right=right,
                event_id=event_id_from_row(row),
            )
        )
    return particles


def parse_frame_set(values: Any, *, key: str, errors: list[str]) -> set[int]:
    if values is None:
        return set()
    if not isinstance(values, list):
        errors.append(f"{key} must be a list")
        return set()
    result: set[int] = set()
    for index, value in enumerate(values):
        if isinstance(value, dict):
            frame = frame_index_from_row(value)
        else:
            frame = finite_int(value)
        if frame is None:
            errors.append(f"{key}[{index}] has no valid frame index")
        else:
            result.add(frame)
    return result


def review_status(row: dict[str, Any]) -> str:
    value = row.get("review_status", row.get("status", ""))
    return str(value).strip().lower()


def validated_label_state(truth_path: Path | str) -> LabelValidationReport:
    """Validate a JSON truth file for metric readiness.

    The validator is intentionally conservative. A JSON label file is metric-ready only
    when it is explicitly marked as reviewed ground truth, manual review is disabled,
    and every scored frame has either particle boxes or an explicit empty-frame
    confirmation.
    """

    path = Path(truth_path)
    errors: list[str] = []
    warnings: list[str] = []
    if not path.is_file():
        return LabelValidationReport(
            truth_path=path,
            status=None,
            requires_manual_review=None,
            n_scored_frames=0,
            n_particle_boxes=0,
            n_empty_frames=0,
            n_frame_reviews=0,
            n_needs_review=0,
            n_reviewed_with_particles=0,
            n_reviewed_empty=0,
            n_unaccounted_scored_frames=0,
            is_valid_for_metrics=False,
            errors=[f"truth path does not exist: {path}"],
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return LabelValidationReport(
            truth_path=path,
            status=None,
            requires_manual_review=None,
            n_scored_frames=0,
            n_particle_boxes=0,
            n_empty_frames=0,
            n_frame_reviews=0,
            n_needs_review=0,
            n_reviewed_with_particles=0,
            n_reviewed_empty=0,
            n_unaccounted_scored_frames=0,
            is_valid_for_metrics=False,
            errors=[f"invalid JSON: {exc}"],
        )
    if not isinstance(payload, dict):
        errors.append("truth JSON must contain an object at the top level")
        payload = {}

    status = payload.get("status")
    status_text = str(status).strip() if status is not None else None
    requires_manual_review = bool_or_none(payload.get("requires_manual_review"))
    scored_frames = parse_frame_set(payload.get("scored_frames"), key="scored_frames", errors=errors)
    empty_frames = parse_frame_set(payload.get("empty_frames"), key="empty_frames", errors=errors)
    particles = parse_particles(payload, errors=errors)
    particles_by_frame: dict[int, list[LabelParticle]] = defaultdict(list)
    for particle in particles:
        particles_by_frame[particle.frame_index].append(particle)
    particle_frames = set(particles_by_frame)

    review_rows = first_list_payload(payload, _REVIEW_KEYS)
    if review_rows and not all(isinstance(row, dict) for row in review_rows):
        errors.append("frame review rows must all be objects")
        review_rows = [row for row in review_rows if isinstance(row, dict)]

    needs_review = 0
    reviewed_with_particles = 0
    reviewed_empty = 0
    review_empty_frames: set[int] = set()
    reviewed_particle_frames: set[int] = set()
    reviewed_frames: set[int] = set()
    for index, row in enumerate(review_rows):
        if not isinstance(row, dict):
            continue
        frame = frame_index_from_row(row)
        status_value = review_status(row)
        confirmed_empty = bool_or_none(row.get("confirmed_empty"))
        if frame is None:
            errors.append(f"frame review row {index} has no valid frame_index")
            continue
        reviewed_frames.add(frame)
        if status_value in _NEEDS_REVIEW_STATUSES:
            needs_review += 1
        if status_value in _EMPTY_REVIEW_STATUSES or confirmed_empty is True:
            reviewed_empty += 1
            review_empty_frames.add(frame)
        if status_value in _PARTICLE_REVIEW_STATUSES:
            reviewed_with_particles += 1
            reviewed_particle_frames.add(frame)
        if confirmed_empty is True and frame in particle_frames:
            errors.append(f"frame {frame} is confirmed empty but also has particle boxes")

    effective_empty_frames = set(empty_frames) | review_empty_frames
    frames_with_boxes_and_empty = particle_frames & effective_empty_frames
    if frames_with_boxes_and_empty:
        listed = ", ".join(str(frame) for frame in sorted(frames_with_boxes_and_empty)[:10])
        errors.append(f"frame(s) marked empty and containing particles: {listed}")

    event_ids = [particle.event_id for particle in particles if particle.event_id is not None]
    duplicated_event_ids = sorted(event_id for event_id, count in Counter(event_ids).items() if count > 1)
    if duplicated_event_ids:
        listed = ", ".join(duplicated_event_ids[:10])
        errors.append(f"duplicate event_id values: {listed}")

    if not scored_frames:
        errors.append("scored_frames is empty or missing")
    outside_scored = sorted((particle_frames | effective_empty_frames) - scored_frames)
    if outside_scored:
        listed = ", ".join(str(frame) for frame in outside_scored[:10])
        warnings.append(f"labels reference frame(s) outside scored_frames: {listed}")

    accounted_frames = (particle_frames | effective_empty_frames) & scored_frames
    unaccounted_frames = sorted(scored_frames - accounted_frames)

    if status_text != REVIEWED_GROUND_TRUTH_STATUS:
        errors.append(
            f"status must be {REVIEWED_GROUND_TRUTH_STATUS!r} for metrics, "
            f"got {status_text!r}"
        )
    if requires_manual_review is not False:
        errors.append("requires_manual_review must be false for metrics")
    if needs_review:
        errors.append(f"{needs_review} frame review row(s) are still marked needs_review")
    if unaccounted_frames:
        listed = ", ".join(str(frame) for frame in unaccounted_frames[:10])
        suffix = "" if len(unaccounted_frames) <= 10 else f" ... (+{len(unaccounted_frames) - 10} more)"
        errors.append(
            "scored frame(s) lack particle boxes or explicit empty confirmation: "
            f"{listed}{suffix}"
        )

    if reviewed_particle_frames - particle_frames:
        listed = ", ".join(str(frame) for frame in sorted(reviewed_particle_frames - particle_frames)[:10])
        errors.append(f"reviewed_with_particles frame(s) have no particle boxes: {listed}")

    return LabelValidationReport(
        truth_path=path,
        status=status_text,
        requires_manual_review=requires_manual_review,
        n_scored_frames=len(scored_frames),
        n_particle_boxes=len(particles),
        n_empty_frames=len(effective_empty_frames & scored_frames),
        n_frame_reviews=len(review_rows),
        n_needs_review=needs_review,
        n_reviewed_with_particles=reviewed_with_particles,
        n_reviewed_empty=reviewed_empty,
        n_unaccounted_scored_frames=len(unaccounted_frames),
        is_valid_for_metrics=not errors,
        errors=errors,
        warnings=warnings,
    )
