from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .yolo_export import DETECTION_FIELDS, natural_key

IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
RECURRENCE_FEATURE_FIELDS = [
    "frame_index",
    "label",
    "source",
    "x",
    "y",
    "bbox_top",
    "bbox_left",
    "bbox_bottom",
    "bbox_right",
    "score_original",
    "confidence_original",
    "belt_y_px",
    "belt_x_px",
    "original_local_max",
    "original_bg99",
    "original_excess",
    "revisit_count",
    "recurrent_revisit_count",
    "max_recurrence_ratio",
    "mean_recurrence_ratio",
    "belt_fixedness_score",
    "transient_score",
    "adjusted_score",
    "hard_reject",
    "causal_read",
    "revisit_frames_json",
    "revisit_excess_json",
    "revisit_ratio_json",
    "revisit_patch_correlation_json",
]


@dataclass(frozen=True)
class CropRegion:
    """Crop in source-image coordinates."""

    top: int
    left: int
    height: int
    width: int

    @classmethod
    def parse(cls, text: str) -> "CropRegion":
        parts = [int(float(part.strip())) for part in text.split(",")]
        if len(parts) != 4:
            raise ValueError("crop region must be top,left,height,width")
        top, left, height, width = parts
        if height <= 0 or width <= 0:
            raise ValueError("crop height/width must be positive")
        return cls(top=top, left=left, height=height, width=width)


@dataclass(frozen=True)
class RecurrenceConfig:
    """Settings for belt-coordinate recurrence scoring of frame detections."""

    crop_region: CropRegion = CropRegion(0, 0, 0, 0)
    map_height_px: float = 0.0
    belt_velocity_px_per_frame: float | None = None
    phase_offset_px: float = 0.0
    max_revolutions: int = 1
    revisit_search_window_frames: int = 2
    recurrence_threshold: float = 0.5
    min_recurrent_revisits: int = 1
    min_original_excess: float = 1.0
    signal_margin_px: int = 2
    background_margin_px: int = 12
    patch_correlation_margin_px: int = 4
    source: str = "yolo_recurrence"


@dataclass(frozen=True)
class PatchStats:
    local_max: float | None
    bg99: float | None
    excess: float | None
    patch: np.ndarray | None


@dataclass(frozen=True)
class RecurrenceSummary:
    output_dir: Path
    n_input_detections: int
    n_hard_kept: int
    n_hard_rejected: int
    n_rerank_detections: int
    n_frames: int
    feature_csv: Path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_float(value: Any, *, default: float | None = None) -> float | None:
    if value is None or str(value).strip() == "":
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def finite_int(value: Any, *, default: int | None = None) -> int | None:
    parsed = finite_float(value)
    if parsed is None:
        return default
    return int(round(parsed))


def infer_frame_index_from_name(name: str, *, pattern: str = r"(\d+)") -> int:
    matches = list(re.finditer(pattern, name))
    if not matches:
        raise ValueError(f"could not infer frame index from {name!r}")
    match = matches[-1]
    value = match.group(1) if match.groups() else match.group(0)
    return int(value)


class ImageSequence:
    """Lazy loader for source images addressed by frame index."""

    def __init__(
        self,
        root: Path,
        *,
        crop_region: CropRegion,
        frame_index_pattern: str = r"(\d+)",
    ) -> None:
        if not root.is_dir():
            raise FileNotFoundError(root)
        self.root = root
        self.crop_region = crop_region
        self._paths = sorted(
            (path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS),
            key=natural_key,
        )
        if not self._paths:
            raise ValueError(f"no supported images found below {root}")
        self._by_frame: dict[int, Path] = {}
        for path in self._paths:
            try:
                frame_index = infer_frame_index_from_name(
                    path.stem,
                    pattern=frame_index_pattern,
                )
            except ValueError:
                continue
            self._by_frame.setdefault(frame_index, path)
        self._cache: dict[int, np.ndarray] = {}

    def has_frame(self, frame_index: int) -> bool:
        if frame_index in self._by_frame:
            return True
        return 0 <= frame_index < len(self._paths)

    def path_for_frame(self, frame_index: int) -> Path:
        if frame_index in self._by_frame:
            return self._by_frame[frame_index]
        if 0 <= frame_index < len(self._paths):
            return self._paths[frame_index]
        raise KeyError(frame_index)

    def crop_for_frame(self, frame_index: int) -> np.ndarray:
        if frame_index in self._cache:
            return self._cache[frame_index]
        path = self.path_for_frame(frame_index)
        with Image.open(path) as image:
            gray = image.convert("L")
            if self.crop_region.height > 0 and self.crop_region.width > 0:
                if gray.size != (self.crop_region.width, self.crop_region.height):
                    gray = gray.crop(
                        (
                            self.crop_region.left,
                            self.crop_region.top,
                            self.crop_region.left + self.crop_region.width,
                            self.crop_region.top + self.crop_region.height,
                        )
                    )
            arr = np.asarray(gray, dtype=np.float64)
        self._cache[frame_index] = arr
        return arr


def load_phase_estimates(path: Path) -> dict[int, float]:
    rows = read_csv(path)
    result: dict[int, float] = {}
    for row in rows:
        frame_text = row.get("frame_index") or row.get("source_frame_index") or row.get("image_index")
        phase_text = row.get("phase_px") or row.get("smoothed_phase_px") or row.get("refined_phase_px")
        if frame_text in (None, "") or phase_text in (None, ""):
            continue
        frame_index = int(float(frame_text))
        result[frame_index] = float(phase_text)
    return result


def infer_map_height(
    *,
    map_height_px: float | None,
    belt_map_path: Path | None,
) -> float:
    if map_height_px is not None and map_height_px > 0:
        return float(map_height_px)
    if belt_map_path is not None and belt_map_path.is_file():
        arr = np.load(belt_map_path, mmap_mode="r")
        return float(arr.shape[0])
    raise ValueError("map height must be supplied via --map-height-px or --belt-map-path")


def estimate_belt_velocity_from_phase(phase_by_frame: Mapping[int, float], map_height_px: float) -> float | None:
    if len(phase_by_frame) < 2:
        return None
    deltas: list[float] = []
    for a, b in zip(sorted(phase_by_frame)[:-1], sorted(phase_by_frame)[1:]):
        if b != a + 1:
            continue
        delta = (phase_by_frame[b] - phase_by_frame[a]) % map_height_px
        if 0 < delta < 0.5 * map_height_px:
            deltas.append(float(delta))
    if not deltas:
        return None
    return float(np.median(deltas))


def phase_for_frame(
    frame_index: int,
    *,
    phase_by_frame: Mapping[int, float],
    config: RecurrenceConfig,
) -> float | None:
    if frame_index in phase_by_frame:
        return float(phase_by_frame[frame_index])
    if config.belt_velocity_px_per_frame is None:
        return None
    return float((config.phase_offset_px + frame_index * config.belt_velocity_px_per_frame) % config.map_height_px)


def detection_center(row: Mapping[str, Any]) -> tuple[float, float]:
    y = finite_float(row.get("y"))
    x = finite_float(row.get("x"))
    if y is not None and x is not None:
        return y, x
    top = finite_float(row.get("bbox_top"))
    left = finite_float(row.get("bbox_left"))
    bottom = finite_float(row.get("bbox_bottom"))
    right = finite_float(row.get("bbox_right"))
    if None in (top, left, bottom, right):
        raise ValueError(f"row has neither y/x nor bbox center fields: {row}")
    assert top is not None and left is not None and bottom is not None and right is not None
    return 0.5 * (top + bottom), 0.5 * (left + right)


def detection_half_size(row: Mapping[str, Any]) -> tuple[float, float]:
    top = finite_float(row.get("bbox_top"))
    left = finite_float(row.get("bbox_left"))
    bottom = finite_float(row.get("bbox_bottom"))
    right = finite_float(row.get("bbox_right"))
    if None in (top, left, bottom, right):
        return 8.0, 8.0
    assert top is not None and left is not None and bottom is not None and right is not None
    return max(1.0, 0.5 * (bottom - top)), max(1.0, 0.5 * (right - left))


def _clip_box(
    *,
    center_y: float,
    center_x: float,
    half_y: float,
    half_x: float,
    height: int,
    width: int,
) -> tuple[int, int, int, int] | None:
    top = max(0, int(math.floor(center_y - half_y)))
    bottom = min(height, int(math.ceil(center_y + half_y)))
    left = max(0, int(math.floor(center_x - half_x)))
    right = min(width, int(math.ceil(center_x + half_x)))
    if bottom <= top or right <= left:
        return None
    return top, left, bottom, right


def patch_stats(
    image: np.ndarray,
    *,
    center_y: float,
    center_x: float,
    half_y: float,
    half_x: float,
    signal_margin_px: int,
    background_margin_px: int,
    patch_correlation_margin_px: int,
) -> PatchStats:
    height, width = image.shape[:2]
    signal = _clip_box(
        center_y=center_y,
        center_x=center_x,
        half_y=half_y + signal_margin_px,
        half_x=half_x + signal_margin_px,
        height=height,
        width=width,
    )
    if signal is None:
        return PatchStats(None, None, None, None)
    top, left, bottom, right = signal
    signal_patch = image[top:bottom, left:right]
    local_max = float(np.max(signal_patch)) if signal_patch.size else None

    outer = _clip_box(
        center_y=center_y,
        center_x=center_x,
        half_y=half_y + background_margin_px,
        half_x=half_x + background_margin_px,
        height=height,
        width=width,
    )
    bg99 = None
    if outer is not None:
        ot, ol, ob, oright = outer
        outer_patch = image[ot:ob, ol:oright]
        mask = np.ones(outer_patch.shape, dtype=bool)
        inner_top = max(0, top - ot)
        inner_bottom = min(mask.shape[0], bottom - ot)
        inner_left = max(0, left - ol)
        inner_right = min(mask.shape[1], right - ol)
        mask[inner_top:inner_bottom, inner_left:inner_right] = False
        annulus = outer_patch[mask]
        if annulus.size:
            bg99 = float(np.percentile(annulus, 99.0))
    if bg99 is None:
        bg99 = float(np.percentile(image, 99.0)) if image.size else None

    patch = None
    patch_box = _clip_box(
        center_y=center_y,
        center_x=center_x,
        half_y=half_y + patch_correlation_margin_px,
        half_x=half_x + patch_correlation_margin_px,
        height=height,
        width=width,
    )
    if patch_box is not None:
        pt, pl, pb, pr = patch_box
        patch = image[pt:pb, pl:pr].copy()
    excess = None if local_max is None or bg99 is None else float(local_max - bg99)
    return PatchStats(local_max=local_max, bg99=bg99, excess=excess, patch=patch)


def patch_correlation(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None or a.shape != b.shape or a.size < 4:
        return None
    av = a.astype(np.float64).ravel()
    bv = b.astype(np.float64).ravel()
    av = av - float(np.mean(av))
    bv = bv - float(np.mean(bv))
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 0:
        return None
    return float(np.dot(av, bv) / denom)


def projected_image_y(
    belt_y: float,
    *,
    phase_px: float,
    map_height_px: float,
) -> float:
    return float((belt_y - phase_px) % map_height_px)


def candidate_revisit_frames(
    frame_index: int,
    *,
    config: RecurrenceConfig,
    phase_by_frame: Mapping[int, float],
    images: ImageSequence,
) -> list[int]:
    velocity = config.belt_velocity_px_per_frame
    if velocity is None:
        velocity = estimate_belt_velocity_from_phase(phase_by_frame, config.map_height_px)
    if velocity is None or velocity <= 0:
        return []
    period_frames = config.map_height_px / velocity
    candidates: list[int] = []
    for revolution_offset in range(-config.max_revolutions, config.max_revolutions + 1):
        if revolution_offset == 0:
            continue
        center = frame_index + revolution_offset * period_frames
        for delta in range(-config.revisit_search_window_frames, config.revisit_search_window_frames + 1):
            revisit = int(round(center + delta))
            if revisit == frame_index or revisit < 0 or not images.has_frame(revisit):
                continue
            if revisit not in candidates:
                candidates.append(revisit)
    return sorted(candidates, key=lambda value: abs(value - frame_index))


def score_detection_recurrence(
    row: Mapping[str, str],
    *,
    images: ImageSequence,
    phase_by_frame: Mapping[int, float],
    config: RecurrenceConfig,
) -> tuple[dict[str, Any], dict[str, str], dict[str, str] | None]:
    frame_index = int(float(row["frame_index"]))
    label = str(row.get("label", ""))
    y, x = detection_center(row)
    half_y, half_x = detection_half_size(row)
    phase = phase_for_frame(frame_index, phase_by_frame=phase_by_frame, config=config)
    if phase is None:
        feature = _empty_feature_row(row, reason="missing phase for detection frame")
        return feature, _rerank_row(row, adjusted_score=finite_float(row.get("score"), default=1.0) or 1.0), dict(row)

    belt_y = float((y + phase) % config.map_height_px)
    original = patch_stats(
        images.crop_for_frame(frame_index),
        center_y=y,
        center_x=x,
        half_y=half_y,
        half_x=half_x,
        signal_margin_px=config.signal_margin_px,
        background_margin_px=config.background_margin_px,
        patch_correlation_margin_px=config.patch_correlation_margin_px,
    )
    original_excess = original.excess
    revisit_frames: list[int] = []
    revisit_excesses: list[float | None] = []
    revisit_ratios: list[float] = []
    revisit_correlations: list[float | None] = []

    for revisit_frame in candidate_revisit_frames(
        frame_index,
        config=config,
        phase_by_frame=phase_by_frame,
        images=images,
    ):
        revisit_phase = phase_for_frame(revisit_frame, phase_by_frame=phase_by_frame, config=config)
        if revisit_phase is None:
            continue
        revisit_y = projected_image_y(
            belt_y,
            phase_px=revisit_phase,
            map_height_px=config.map_height_px,
        )
        if not 0.0 <= revisit_y < images.crop_region.height:
            continue
        revisit = patch_stats(
            images.crop_for_frame(revisit_frame),
            center_y=revisit_y,
            center_x=x,
            half_y=half_y,
            half_x=half_x,
            signal_margin_px=config.signal_margin_px,
            background_margin_px=config.background_margin_px,
            patch_correlation_margin_px=config.patch_correlation_margin_px,
        )
        revisit_frames.append(revisit_frame)
        revisit_excesses.append(revisit.excess)
        denom = max(float(original_excess or 0.0), config.min_original_excess)
        ratio = 0.0 if revisit.excess is None else float(max(0.0, revisit.excess) / denom)
        revisit_ratios.append(ratio)
        revisit_correlations.append(patch_correlation(original.patch, revisit.patch))

    recurrent_revisit_count = sum(1 for ratio in revisit_ratios if ratio >= config.recurrence_threshold)
    max_ratio = max(revisit_ratios, default=0.0)
    mean_ratio = float(np.mean(revisit_ratios)) if revisit_ratios else 0.0
    if not revisit_ratios:
        belt_fixedness = 0.0
        read = "no usable revisit frames"
    else:
        support_factor = min(1.0, recurrent_revisit_count / max(1, config.min_recurrent_revisits))
        belt_fixedness = min(1.0, max_ratio * support_factor)
        read = (
            "belt-coordinate recurrence above threshold"
            if recurrent_revisit_count >= config.min_recurrent_revisits
            else "no strong belt-coordinate recurrence"
        )
    transient_score = float(max(0.0, min(1.0, 1.0 - belt_fixedness)))
    original_score = finite_float(row.get("score"), default=finite_float(row.get("confidence"), default=1.0))
    if original_score is None:
        original_score = 1.0
    adjusted_score = float(original_score * transient_score)
    hard_reject = recurrent_revisit_count >= config.min_recurrent_revisits

    feature = {
        "frame_index": frame_index,
        "label": label,
        "source": row.get("source", ""),
        "x": x,
        "y": y,
        "bbox_top": row.get("bbox_top", ""),
        "bbox_left": row.get("bbox_left", ""),
        "bbox_bottom": row.get("bbox_bottom", ""),
        "bbox_right": row.get("bbox_right", ""),
        "score_original": original_score,
        "confidence_original": finite_float(row.get("confidence"), default=original_score),
        "belt_y_px": belt_y,
        "belt_x_px": x,
        "original_local_max": original.local_max,
        "original_bg99": original.bg99,
        "original_excess": original_excess,
        "revisit_count": len(revisit_frames),
        "recurrent_revisit_count": recurrent_revisit_count,
        "max_recurrence_ratio": max_ratio,
        "mean_recurrence_ratio": mean_ratio,
        "belt_fixedness_score": belt_fixedness,
        "transient_score": transient_score,
        "adjusted_score": adjusted_score,
        "hard_reject": hard_reject,
        "causal_read": read,
        "revisit_frames_json": json.dumps(revisit_frames),
        "revisit_excess_json": json.dumps(revisit_excesses),
        "revisit_ratio_json": json.dumps(revisit_ratios),
        "revisit_patch_correlation_json": json.dumps(revisit_correlations),
    }
    rerank = _rerank_row(row, adjusted_score=adjusted_score)
    hard = None if hard_reject else dict(row)
    return feature, rerank, hard


def _empty_feature_row(row: Mapping[str, str], *, reason: str) -> dict[str, Any]:
    y, x = detection_center(row)
    score = finite_float(row.get("score"), default=finite_float(row.get("confidence"), default=1.0)) or 1.0
    return {
        "frame_index": int(float(row["frame_index"])),
        "label": row.get("label", ""),
        "source": row.get("source", ""),
        "x": x,
        "y": y,
        "bbox_top": row.get("bbox_top", ""),
        "bbox_left": row.get("bbox_left", ""),
        "bbox_bottom": row.get("bbox_bottom", ""),
        "bbox_right": row.get("bbox_right", ""),
        "score_original": score,
        "confidence_original": finite_float(row.get("confidence"), default=score),
        "belt_y_px": "",
        "belt_x_px": x,
        "original_local_max": "",
        "original_bg99": "",
        "original_excess": "",
        "revisit_count": 0,
        "recurrent_revisit_count": 0,
        "max_recurrence_ratio": 0.0,
        "mean_recurrence_ratio": 0.0,
        "belt_fixedness_score": 0.0,
        "transient_score": 1.0,
        "adjusted_score": score,
        "hard_reject": False,
        "causal_read": reason,
        "revisit_frames_json": "[]",
        "revisit_excess_json": "[]",
        "revisit_ratio_json": "[]",
        "revisit_patch_correlation_json": "[]",
    }


def _rerank_row(row: Mapping[str, str], *, adjusted_score: float) -> dict[str, str]:
    payload = dict(row)
    original_score = payload.get("score", payload.get("confidence", ""))
    payload.setdefault("original_score", original_score)
    payload["score"] = f"{adjusted_score:.8f}"
    payload["confidence"] = f"{adjusted_score:.8f}"
    return payload


def write_beltmap_run(
    output_dir: Path,
    detections: Sequence[Mapping[str, str]],
    *,
    frames: Sequence[int],
    metadata: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    detections_sorted = sorted(
        (dict(row) for row in detections),
        key=lambda row: (int(float(row["frame_index"])), int(float(row.get("label", 0) or 0))),
    )
    fields = list(DETECTION_FIELDS)
    for optional in ("original_score",):
        if any(optional in row for row in detections_sorted) and optional not in fields:
            fields.append(optional)
    write_csv(output_dir / "detections.csv", detections_sorted, fields)
    counts = {int(frame): 0 for frame in frames}
    for row in detections_sorted:
        frame = int(float(row["frame_index"]))
        counts[frame] = counts.get(frame, 0) + 1
    write_csv(
        output_dir / "detections_per_frame.csv",
        [
            {"frame_index": frame, "n_detections": counts.get(frame, 0)}
            for frame in sorted(counts)
        ],
        ["frame_index", "n_detections"],
    )
    (output_dir / "metadata.json").write_text(json.dumps(dict(metadata), indent=2) + "\n", encoding="utf-8")
    (output_dir / "config_resolved.json").write_text(
        json.dumps({"mode": "yolo_recurrence", "detection": {"score_field": "score"}}, indent=2)
        + "\n",
        encoding="utf-8",
    )


def write_report(path: Path, *, summary: RecurrenceSummary, config: RecurrenceConfig) -> None:
    path.write_text(
        "\n".join(
            [
                "# YOLO Belt-Coordinate Recurrence Scoring",
                "",
                "This report scores frame-based YOLO detections with belt-coordinate history.",
                "High recurrence means the same belt coordinate remains bright in another belt revolution,",
                "which is evidence for belt-fixed dirt/artifact behavior rather than a transient particle.",
                "",
                "## Summary",
                "",
                f"- Input detections: {summary.n_input_detections}",
                f"- Hard-filter kept: {summary.n_hard_kept}",
                f"- Hard-filter rejected: {summary.n_hard_rejected}",
                f"- Rerank detections: {summary.n_rerank_detections}",
                f"- Frames represented: {summary.n_frames}",
                f"- Features CSV: `{summary.feature_csv}`",
                "",
                "## Configuration",
                "",
                f"- Max revolutions checked: {config.max_revolutions}",
                f"- Revisit search window: +/- {config.revisit_search_window_frames} frames",
                f"- Recurrence ratio threshold: {config.recurrence_threshold}",
                f"- Minimum recurrent revisits for hard rejection: {config.min_recurrent_revisits}",
                f"- Signal margin: {config.signal_margin_px} px",
                f"- Background margin: {config.background_margin_px} px",
                "",
                "## Outputs",
                "",
                "- `features.csv`: recurrence features per input detection.",
                "- `hard_filter/`: BeltMap-style run after removing recurrent detections.",
                "- `rerank/`: BeltMap-style run with score = YOLO confidence x transient score.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def score_yolo_recurrence(
    *,
    detections_csv: Path,
    images_dir: Path,
    output_dir: Path,
    phase_estimates_csv: Path | None = None,
    belt_map_path: Path | None = None,
    map_height_px: float | None = None,
    belt_velocity_px_per_frame: float | None = None,
    phase_offset_px: float = 0.0,
    crop_region: CropRegion = CropRegion(0, 0, 0, 0),
    frame_index_pattern: str = r"(\d+)",
    max_revolutions: int = 1,
    revisit_search_window_frames: int = 2,
    recurrence_threshold: float = 0.5,
    min_recurrent_revisits: int = 1,
    min_original_excess: float = 1.0,
    signal_margin_px: int = 2,
    background_margin_px: int = 12,
    patch_correlation_margin_px: int = 4,
) -> RecurrenceSummary:
    detections = read_csv(detections_csv)
    if not detections:
        raise ValueError(f"no detections found in {detections_csv}")
    map_height = infer_map_height(map_height_px=map_height_px, belt_map_path=belt_map_path)
    phase_by_frame = load_phase_estimates(phase_estimates_csv) if phase_estimates_csv else {}
    velocity = belt_velocity_px_per_frame
    if velocity is None:
        velocity = estimate_belt_velocity_from_phase(phase_by_frame, map_height)
    config = RecurrenceConfig(
        crop_region=crop_region,
        map_height_px=map_height,
        belt_velocity_px_per_frame=velocity,
        phase_offset_px=phase_offset_px,
        max_revolutions=max_revolutions,
        revisit_search_window_frames=revisit_search_window_frames,
        recurrence_threshold=recurrence_threshold,
        min_recurrent_revisits=min_recurrent_revisits,
        min_original_excess=min_original_excess,
        signal_margin_px=signal_margin_px,
        background_margin_px=background_margin_px,
        patch_correlation_margin_px=patch_correlation_margin_px,
    )
    if config.map_height_px <= 0:
        raise ValueError("map height must be positive")
    if velocity is None or velocity <= 0:
        raise ValueError("belt velocity must be supplied or inferred from phase_estimates.csv")
    if config.max_revolutions < 1:
        raise ValueError("max_revolutions must be positive")
    if config.revisit_search_window_frames < 0:
        raise ValueError("revisit_search_window_frames must be non-negative")
    if not 0 <= config.recurrence_threshold:
        raise ValueError("recurrence_threshold must be non-negative")
    if config.min_recurrent_revisits < 1:
        raise ValueError("min_recurrent_revisits must be positive")

    images = ImageSequence(images_dir, crop_region=crop_region, frame_index_pattern=frame_index_pattern)
    features: list[dict[str, Any]] = []
    hard: list[dict[str, str]] = []
    rerank: list[dict[str, str]] = []
    for row in detections:
        feature, rerank_row, hard_row = score_detection_recurrence(
            row,
            images=images,
            phase_by_frame=phase_by_frame,
            config=config,
        )
        features.append(feature)
        rerank.append(rerank_row)
        if hard_row is not None:
            hard.append(hard_row)

    frames = sorted({int(float(row["frame_index"])) for row in detections})
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_csv = output_dir / "features.csv"
    write_csv(feature_csv, features, RECURRENCE_FEATURE_FIELDS)
    metadata = {
        "mode": "yolo_recurrence",
        "source_detections_csv": str(detections_csv),
        "images_dir": str(images_dir),
        "phase_estimates_csv": None if phase_estimates_csv is None else str(phase_estimates_csv),
        "belt_map_path": None if belt_map_path is None else str(belt_map_path),
        "map_height_px": map_height,
        "belt_velocity_px_per_frame": velocity,
        "config": asdict(config),
        "n_input_detections": len(detections),
        "n_hard_kept": len(hard),
        "n_hard_rejected": len(detections) - len(hard),
        "n_rerank_detections": len(rerank),
    }
    write_beltmap_run(output_dir / "hard_filter", hard, frames=frames, metadata=metadata)
    write_beltmap_run(output_dir / "rerank", rerank, frames=frames, metadata=metadata)
    summary = RecurrenceSummary(
        output_dir=output_dir,
        n_input_detections=len(detections),
        n_hard_kept=len(hard),
        n_hard_rejected=len(detections) - len(hard),
        n_rerank_detections=len(rerank),
        n_frames=len(frames),
        feature_csv=feature_csv,
    )
    (output_dir / "summary.json").write_text(
        json.dumps({**metadata, "summary": asdict(summary)}, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    write_report(output_dir / "report.md", summary=summary, config=config)
    return summary
