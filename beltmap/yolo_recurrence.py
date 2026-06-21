from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from beltmap.compare_runs import (
    generate_comparison_report,
    load_labeled_detection_truth,
    parse_run_spec,
)
from beltmap.phase import BeltMotionModel, render_belt_view
from beltmap.recurrent_artifacts import belt_revolution_indices
from beltmap.rendering import BeltRegion
from beltmap.yolo_export import IMAGE_EXTENSIONS, infer_frame_index, natural_key


DEFAULT_HARD_RATIO_THRESHOLD = 0.40
DEFAULT_HARD_MIN_REVISITS = 2
DEFAULT_PATCH_MARGIN_PX = 4
DEFAULT_MIN_PATCH_SIZE_PX = 9
DEFAULT_EXCESS_FLOOR = 1.0
FEATURE_FIELDNAMES = [
    "frame_index",
    "label",
    "source",
    "y",
    "x",
    "bbox_top",
    "bbox_left",
    "bbox_bottom",
    "bbox_right",
    "confidence",
    "score",
    "belt_y",
    "patch_top",
    "patch_left",
    "patch_bottom",
    "patch_right",
    "original_excess",
    "revisit_frame_prev",
    "revisit_y_prev",
    "revisit_excess_prev",
    "recurrence_ratio_prev",
    "patch_correlation_prev",
    "revisit_frame_next",
    "revisit_y_next",
    "revisit_excess_next",
    "recurrence_ratio_next",
    "patch_correlation_next",
    "max_recurrence_ratio",
    "belt_fixedness_score",
    "transient_score",
    "valid_revisits",
    "high_recurrence_revisits",
    "hard_reject",
    "raw_match_role",
    "error_taxonomy",
]
RUN_EXTRA_FIELDS = [
    "mean_signal",
    "peak_signal",
    "yolo_confidence_original",
    "adjusted_score",
    "belt_fixedness_score",
    "transient_score",
    "max_recurrence_ratio",
    "high_recurrence_revisits",
    "hard_reject",
]


@dataclass(frozen=True)
class PatchBox:
    top: int
    left: int
    bottom: int
    right: int

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def width(self) -> int:
        return self.right - self.left


@dataclass(frozen=True)
class PatchEvidence:
    frame_index: int
    center_y: float
    raw_patch: NDArray[np.floating]
    background_patch: NDArray[np.floating]
    residual_patch: NDArray[np.floating]
    excess: float


@dataclass(frozen=True)
class YoloRecurrenceConfig:
    frame_count: int = 500
    belt_region: BeltRegion = BeltRegion(0, 220, 1330, 1800)
    hard_ratio_threshold: float = DEFAULT_HARD_RATIO_THRESHOLD
    hard_min_revisits: int = DEFAULT_HARD_MIN_REVISITS
    patch_margin_px: int = DEFAULT_PATCH_MARGIN_PX
    min_patch_size_px: int = DEFAULT_MIN_PATCH_SIZE_PX
    excess_floor: float = DEFAULT_EXCESS_FLOOR
    froc_max_thresholds: int = 250
    bootstrap_samples: int = 0
    bootstrap_block_length_frames: int = 5


@dataclass(frozen=True)
class YoloRecurrenceSummary:
    output_dir: Path
    features_csv: Path
    hard_run_dir: Path
    rerank_run_dir: Path
    report_md: Path
    contact_sheet_png: Path
    compare_summary_csv: Path | None
    n_detections: int
    n_hard_rejected: int
    n_raw_false_positives_removed: int
    n_raw_true_positives_removed: int


def run_yolo_recurrence_filter(
    *,
    yolo_run_dir: Path,
    beltmap_reference_dir: Path,
    source_image_dir: Path,
    output_dir: Path,
    truth_path: Path | None,
    config: YoloRecurrenceConfig,
) -> YoloRecurrenceSummary:
    """Compute belt-coordinate recurrence features and exported YOLO post-filter runs."""

    yolo_run_dir = yolo_run_dir.resolve()
    beltmap_reference_dir = beltmap_reference_dir.resolve()
    source_image_dir = source_image_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    detection_rows, detection_fieldnames = read_csv_rows_with_fieldnames(
        yolo_run_dir / "detections.csv"
    )
    per_frame_rows = read_csv_rows(yolo_run_dir / "detections_per_frame.csv")
    metadata = read_json(beltmap_reference_dir / "metadata.json")
    belt_map = np.load(beltmap_reference_dir / "belt_map.npy")
    if belt_map.ndim != 2:
        raise ValueError("belt_map.npy must contain a 2-D map")
    phase_by_frame = load_phase_px_by_frame(
        beltmap_reference_dir / "phase_estimates.csv",
        frame_count=config.frame_count,
    )
    source_images = find_source_images(source_image_dir)

    if config.belt_region.width != belt_map.shape[1]:
        raise ValueError(
            "belt_region width must match belt_map width: "
            f"{config.belt_region.width} != {belt_map.shape[1]}"
        )
    belt_velocity = metadata_float(metadata, "belt_velocity_px_per_frame")
    period_px = metadata_float(
        metadata,
        "belt_period_px_input",
        default=metadata_float(metadata, "belt_map_height_px", default=float(belt_map.shape[0])),
    )
    reference_phase_px = metadata_float(metadata, "reference_phase_px", default=0.0)
    revolution_by_frame = belt_revolution_indices(
        config.frame_count,
        BeltMotionModel(
            image_velocity_px_per_frame=belt_velocity,
            period_px=period_px,
            reference_phase_px=reference_phase_px,
        ),
    )
    raw_match_roles = (
        match_detection_roles(detection_rows, truth_path=truth_path)
        if truth_path is not None
        else {}
    )

    crop_cache: dict[int, NDArray[np.floating]] = {}
    feature_rows: list[dict[str, Any]] = []
    feature_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for row in detection_rows:
        feature = score_detection_recurrence(
            row,
            belt_map=belt_map,
            phase_by_frame=phase_by_frame,
            revolution_by_frame=revolution_by_frame,
            source_images=source_images,
            crop_cache=crop_cache,
            config=config,
        )
        key = row_key(row)
        role = raw_match_roles.get(key, "unscored")
        feature["raw_match_role"] = role
        feature["error_taxonomy"] = error_taxonomy(feature, role=role)
        feature_rows.append(feature)
        feature_by_key[key] = feature

    write_csv(output_dir / "yolo_recurrence_features.csv", feature_rows, FEATURE_FIELDNAMES)
    hard_run_dir = output_dir.parent / "beltmap_runs" / "yolo11_raw_recurrence_hard_test"
    rerank_run_dir = output_dir.parent / "beltmap_runs" / "yolo11_raw_recurrence_rerank_test"
    hard_rows = [
        enrich_detection_row(
            row,
            feature_by_key[row_key(row)],
            rerank=False,
        )
        for row in detection_rows
        if not bool_value(feature_by_key[row_key(row)]["hard_reject"])
    ]
    rerank_rows = [
        enrich_detection_row(
            row,
            feature_by_key[row_key(row)],
            rerank=True,
        )
        for row in detection_rows
    ]
    write_beltmap_run(
        hard_run_dir,
        rows=hard_rows,
        per_frame_rows=per_frame_rows,
        source_run=yolo_run_dir,
        mode="yolo11_raw_recurrence_hard_test",
        config=config,
        source_fieldnames=detection_fieldnames,
    )
    write_beltmap_run(
        rerank_run_dir,
        rows=rerank_rows,
        per_frame_rows=per_frame_rows,
        source_run=yolo_run_dir,
        mode="yolo11_raw_recurrence_rerank_test",
        config=config,
        source_fieldnames=detection_fieldnames,
    )

    contact_sheet_path = output_dir / "yolo_fp_fn_recurrence_contact_sheet.png"
    write_contact_sheet(
        contact_sheet_path,
        feature_rows,
        source_images=source_images,
        config=config,
        crop_cache=crop_cache,
    )
    compare_summary_csv = None
    if truth_path is not None:
        compare_dir = output_dir / "compare"
        specs = [
            parse_run_spec(f"yolo11_raw={yolo_run_dir}"),
            parse_run_spec(f"yolo11_raw_recurrence_hard={hard_run_dir}"),
            parse_run_spec(f"yolo11_raw_recurrence_rerank={rerank_run_dir}"),
        ]
        if (beltmap_reference_dir / "detections.csv").is_file():
            specs.append(parse_run_spec(f"beltmap_reference={beltmap_reference_dir}"))
        artifacts = generate_comparison_report(
            specs,
            report_dir=compare_dir,
            truth_path=truth_path,
            truth_iou_threshold=0.25,
            froc_max_thresholds=config.froc_max_thresholds,
            bootstrap_samples=config.bootstrap_samples,
            bootstrap_block_length_frames=config.bootstrap_block_length_frames,
            make_metric_plots=False,
            make_contact_sheets=False,
        )
        compare_summary_csv = artifacts.summary_csv

    n_hard_rejected = sum(bool_value(row["hard_reject"]) for row in feature_rows)
    n_fp_removed = sum(
        bool_value(row["hard_reject"]) and row.get("raw_match_role") == "FP"
        for row in feature_rows
    )
    n_tp_removed = sum(
        bool_value(row["hard_reject"]) and row.get("raw_match_role") == "TP"
        for row in feature_rows
    )
    report_path = output_dir / "yolo_recurrence_report.md"
    write_recurrence_report(
        report_path,
        feature_rows,
        truth_path=truth_path,
        compare_summary_csv=compare_summary_csv,
        hard_run_dir=hard_run_dir,
        rerank_run_dir=rerank_run_dir,
    )
    return YoloRecurrenceSummary(
        output_dir=output_dir,
        features_csv=output_dir / "yolo_recurrence_features.csv",
        hard_run_dir=hard_run_dir,
        rerank_run_dir=rerank_run_dir,
        report_md=report_path,
        contact_sheet_png=contact_sheet_path,
        compare_summary_csv=compare_summary_csv,
        n_detections=len(detection_rows),
        n_hard_rejected=n_hard_rejected,
        n_raw_false_positives_removed=n_fp_removed,
        n_raw_true_positives_removed=n_tp_removed,
    )


def score_detection_recurrence(
    row: Mapping[str, Any],
    *,
    belt_map: NDArray[np.floating],
    phase_by_frame: Sequence[float],
    revolution_by_frame: Sequence[int],
    source_images: Mapping[int, Path],
    crop_cache: dict[int, NDArray[np.floating]],
    config: YoloRecurrenceConfig,
) -> dict[str, Any]:
    frame_index = int_value(row["frame_index"], name="frame_index")
    label = int_value(row["label"], name="label")
    y = float_value(row["y"], name="y")
    x = float_value(row["x"], name="x")
    bbox = bbox_from_row(row)
    patch = detection_patch_box(
        bbox,
        image_shape=config.belt_region.shape,
        margin_px=config.patch_margin_px,
        min_size_px=config.min_patch_size_px,
    )
    phase = phase_by_frame[frame_index]
    belt_y = (y + phase) % belt_map.shape[0]
    original = patch_evidence(
        frame_index,
        center_y=0.5 * (patch.top + patch.bottom),
        patch=patch,
        belt_map=belt_map,
        phase_by_frame=phase_by_frame,
        source_images=source_images,
        crop_cache=crop_cache,
        config=config,
    )
    revisit_features: dict[str, Any] = {}
    recurrence_scores: list[float] = []
    ratios: list[float] = []
    valid_revisits = 0
    high_revisits = 0
    for suffix, offset in (("prev", -1), ("next", 1)):
        revisit = find_revisit(
            frame_index=frame_index,
            belt_y=belt_y,
            x=x,
            patch_shape=(patch.height, patch.width),
            revolution_offset=offset,
            phase_by_frame=phase_by_frame,
            revolution_by_frame=revolution_by_frame,
            source_images=source_images,
            image_shape=config.belt_region.shape,
            map_height=belt_map.shape[0],
        )
        if revisit is None:
            revisit_features.update(
                {
                    f"revisit_frame_{suffix}": "",
                    f"revisit_y_{suffix}": "",
                    f"revisit_excess_{suffix}": "",
                    f"recurrence_ratio_{suffix}": "",
                    f"patch_correlation_{suffix}": "",
                }
            )
            continue
        revisit_frame, revisit_y, revisit_patch = revisit
        evidence = patch_evidence(
            revisit_frame,
            center_y=revisit_y,
            patch=revisit_patch,
            belt_map=belt_map,
            phase_by_frame=phase_by_frame,
            source_images=source_images,
            crop_cache=crop_cache,
            config=config,
        )
        ratio = recurrence_ratio(evidence.excess, original.excess, floor=config.excess_floor)
        corr = patch_correlation(original.residual_patch, evidence.residual_patch)
        recurrent_strength = max(0.0, min(1.0, ratio)) * max(0.0, corr)
        recurrence_scores.append(recurrent_strength)
        ratios.append(ratio)
        valid_revisits += 1
        if ratio >= config.hard_ratio_threshold:
            high_revisits += 1
        revisit_features.update(
            {
                f"revisit_frame_{suffix}": revisit_frame,
                f"revisit_y_{suffix}": revisit_y,
                f"revisit_excess_{suffix}": evidence.excess,
                f"recurrence_ratio_{suffix}": ratio,
                f"patch_correlation_{suffix}": corr,
            }
        )

    max_ratio = max(ratios) if ratios else 0.0
    belt_fixedness = second_largest(recurrence_scores) if len(recurrence_scores) >= 2 else 0.0
    transient = float(np.clip(1.0 - belt_fixedness, 0.05, 1.0))
    hard_reject = high_revisits >= config.hard_min_revisits
    result: dict[str, Any] = {
        "frame_index": frame_index,
        "label": label,
        "source": row.get("source", ""),
        "y": y,
        "x": x,
        "bbox_top": bbox.top,
        "bbox_left": bbox.left,
        "bbox_bottom": bbox.bottom,
        "bbox_right": bbox.right,
        "confidence": row.get("confidence", row.get("score", "")),
        "score": row.get("score", row.get("confidence", "")),
        "belt_y": belt_y,
        "patch_top": patch.top,
        "patch_left": patch.left,
        "patch_bottom": patch.bottom,
        "patch_right": patch.right,
        "original_excess": original.excess,
        "max_recurrence_ratio": max_ratio,
        "belt_fixedness_score": belt_fixedness,
        "transient_score": transient,
        "valid_revisits": valid_revisits,
        "high_recurrence_revisits": high_revisits,
        "hard_reject": hard_reject,
    }
    result.update(revisit_features)
    return result


def find_revisit(
    *,
    frame_index: int,
    belt_y: float,
    x: float,
    patch_shape: tuple[int, int],
    revolution_offset: int,
    phase_by_frame: Sequence[float],
    revolution_by_frame: Sequence[int],
    source_images: Mapping[int, Path],
    image_shape: tuple[int, int],
    map_height: int,
) -> tuple[int, float, PatchBox] | None:
    source_revolution = int(revolution_by_frame[frame_index])
    target_revolution = source_revolution + revolution_offset
    candidates: list[tuple[float, int, float, PatchBox]] = []
    for candidate_frame, phase in enumerate(phase_by_frame):
        if candidate_frame == frame_index:
            continue
        if candidate_frame not in source_images:
            continue
        if int(revolution_by_frame[candidate_frame]) != target_revolution:
            continue
        projected_y = (belt_y - phase) % map_height
        if projected_y < 0.0 or projected_y >= image_shape[0]:
            continue
        patch = centered_patch_box(
            y=projected_y,
            x=x,
            height=patch_shape[0],
            width=patch_shape[1],
            image_shape=image_shape,
        )
        if patch is None:
            continue
        candidates.append((abs(projected_y - image_shape[0] / 2.0), candidate_frame, projected_y, patch))
    if not candidates:
        return None
    _distance, selected_frame, projected_y, selected_patch = min(
        candidates,
        key=lambda item: (abs(item[2] - image_shape[0] / 2.0), abs(item[1] - frame_index)),
    )
    return selected_frame, projected_y, selected_patch


def patch_evidence(
    frame_index: int,
    *,
    center_y: float,
    patch: PatchBox,
    belt_map: NDArray[np.floating],
    phase_by_frame: Sequence[float],
    source_images: Mapping[int, Path],
    crop_cache: dict[int, NDArray[np.floating]],
    config: YoloRecurrenceConfig,
) -> PatchEvidence:
    crop = load_crop(frame_index, source_images=source_images, crop_cache=crop_cache, region=config.belt_region)
    raw_patch = crop[patch.top : patch.bottom, patch.left : patch.right]
    background_patch = render_belt_view(
        belt_map,
        phase_by_frame[frame_index] + patch.top,
        patch.height,
        x_slice=slice(patch.left, patch.right),
        periodic=True,
    )
    residual = raw_patch - background_patch
    excess = patch_excess(raw_patch, background_patch)
    return PatchEvidence(
        frame_index=frame_index,
        center_y=center_y,
        raw_patch=raw_patch,
        background_patch=background_patch,
        residual_patch=residual,
        excess=excess,
    )


def detection_patch_box(
    bbox: PatchBox,
    *,
    image_shape: tuple[int, int],
    margin_px: int,
    min_size_px: int,
) -> PatchBox:
    center_y = 0.5 * (bbox.top + bbox.bottom)
    center_x = 0.5 * (bbox.left + bbox.right)
    height = max(min_size_px, bbox.height + 2 * margin_px)
    width = max(min_size_px, bbox.width + 2 * margin_px)
    patch = centered_patch_box(
        y=center_y,
        x=center_x,
        height=height,
        width=width,
        image_shape=image_shape,
    )
    if patch is None:
        return clipped_patch_box(
            top=bbox.top - margin_px,
            left=bbox.left - margin_px,
            bottom=bbox.bottom + margin_px,
            right=bbox.right + margin_px,
            image_shape=image_shape,
        )
    return patch


def centered_patch_box(
    *,
    y: float,
    x: float,
    height: int,
    width: int,
    image_shape: tuple[int, int],
) -> PatchBox | None:
    height = max(1, int(height))
    width = max(1, int(width))
    top = int(round(y - 0.5 * height))
    left = int(round(x - 0.5 * width))
    bottom = top + height
    right = left + width
    if top < 0 or left < 0 or bottom > image_shape[0] or right > image_shape[1]:
        return None
    return PatchBox(top=top, left=left, bottom=bottom, right=right)


def clipped_patch_box(
    *,
    top: int,
    left: int,
    bottom: int,
    right: int,
    image_shape: tuple[int, int],
) -> PatchBox:
    clipped = PatchBox(
        top=max(0, int(top)),
        left=max(0, int(left)),
        bottom=min(image_shape[0], int(bottom)),
        right=min(image_shape[1], int(right)),
    )
    if clipped.bottom <= clipped.top or clipped.right <= clipped.left:
        raise ValueError("detection patch clips to an empty region")
    return clipped


def patch_excess(raw_patch: NDArray[np.floating], background_patch: NDArray[np.floating]) -> float:
    finite_raw = np.asarray(raw_patch, dtype=np.float64)
    finite_bg = np.asarray(background_patch, dtype=np.float64)
    mask = np.isfinite(finite_raw) & np.isfinite(finite_bg)
    if not mask.any():
        return 0.0
    return float(np.max(finite_raw[mask]) - np.percentile(finite_bg[mask], 99))


def recurrence_ratio(revisit_excess: float, original_excess: float, *, floor: float) -> float:
    denominator = max(float(original_excess), float(floor))
    if denominator <= 0 or not np.isfinite(denominator):
        denominator = float(floor)
    value = max(0.0, float(revisit_excess)) / denominator
    return 0.0 if not np.isfinite(value) else float(value)


def patch_correlation(
    a: NDArray[np.floating],
    b: NDArray[np.floating],
) -> float:
    if a.shape != b.shape:
        return 0.0
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(arr_a) & np.isfinite(arr_b)
    if int(np.count_nonzero(mask)) < 4:
        return 0.0
    va = arr_a[mask] - np.median(arr_a[mask])
    vb = arr_b[mask] - np.median(arr_b[mask])
    denom = math.sqrt(float(np.sum(va * va) * np.sum(vb * vb)))
    if denom <= 1e-12 or not np.isfinite(denom):
        return 0.0
    corr = float(np.sum(va * vb) / denom)
    return float(np.clip(corr, -1.0, 1.0))


def second_largest(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(sorted(values)[-2])


def load_crop(
    frame_index: int,
    *,
    source_images: Mapping[int, Path],
    crop_cache: dict[int, NDArray[np.floating]],
    region: BeltRegion,
) -> NDArray[np.floating]:
    cached = crop_cache.get(frame_index)
    if cached is not None:
        return cached
    path = source_images.get(frame_index)
    if path is None:
        raise ValueError(f"source image is missing frame {frame_index}")
    with Image.open(path) as image:
        gray = image.convert("L")
        crop = gray.crop((region.left, region.top, region.left + region.width, region.top + region.height))
    arr = np.asarray(crop, dtype=np.float64)
    crop_cache[frame_index] = arr
    return arr


def enrich_detection_row(
    row: Mapping[str, Any],
    feature: Mapping[str, Any],
    *,
    rerank: bool,
) -> dict[str, Any]:
    enriched = dict(row)
    original_score = float_value(row.get("confidence", row.get("score", 1.0)), name="confidence")
    adjusted = original_score * float(feature["transient_score"])
    enriched["mean_signal"] = f"{adjusted:.8f}" if rerank else row.get("mean_signal", row.get("score", ""))
    enriched["peak_signal"] = f"{adjusted:.8f}" if rerank else row.get("peak_signal", row.get("score", ""))
    enriched["score"] = f"{adjusted:.8f}" if rerank else row.get("score", row.get("confidence", ""))
    enriched["confidence"] = f"{adjusted:.8f}" if rerank else row.get("confidence", row.get("score", ""))
    enriched["yolo_confidence_original"] = f"{original_score:.8f}"
    enriched["adjusted_score"] = f"{adjusted:.8f}"
    enriched["belt_fixedness_score"] = format_float(feature["belt_fixedness_score"])
    enriched["transient_score"] = format_float(feature["transient_score"])
    enriched["max_recurrence_ratio"] = format_float(feature["max_recurrence_ratio"])
    enriched["high_recurrence_revisits"] = feature["high_recurrence_revisits"]
    enriched["hard_reject"] = feature["hard_reject"]
    enriched["source"] = (
        "yolo11_raw_recurrence_rerank" if rerank else "yolo11_raw_recurrence_hard"
    )
    return enriched


def write_beltmap_run(
    output_dir: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    per_frame_rows: Sequence[Mapping[str, Any]],
    source_run: Path,
    mode: str,
    config: YoloRecurrenceConfig,
    source_fieldnames: Sequence[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(source_fieldnames)
    for field in RUN_EXTRA_FIELDS:
        if field not in fields:
            fields.append(field)
    write_csv(output_dir / "detections.csv", rows, fields)
    counts: dict[int, int] = {}
    for row in rows:
        counts[int_value(row["frame_index"], name="frame_index")] = counts.get(
            int_value(row["frame_index"], name="frame_index"),
            0,
        ) + 1
    per_frame = [
        {
            "frame_index": int_value(row["frame_index"], name="frame_index"),
            "n_detections": counts.get(int_value(row["frame_index"], name="frame_index"), 0),
        }
        for row in per_frame_rows
    ]
    write_csv(output_dir / "detections_per_frame.csv", per_frame, ["frame_index", "n_detections"])
    metadata = {
        "mode": mode,
        "source_run": str(source_run),
        "n_images": len(per_frame_rows),
        "n_detections": len(rows),
        "hard_ratio_threshold": config.hard_ratio_threshold,
        "hard_min_revisits": config.hard_min_revisits,
        "patch_margin_px": config.patch_margin_px,
        "min_patch_size_px": config.min_patch_size_px,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output_dir / "config_resolved.json").write_text(
        json.dumps({"mode": mode, "detection": {"score_field": "score"}}, indent=2) + "\n",
        encoding="utf-8",
    )


def match_detection_roles(
    detection_rows: Sequence[Mapping[str, Any]],
    *,
    truth_path: Path,
    iou_threshold: float = 0.25,
) -> dict[tuple[int, int], str]:
    truth = load_labeled_detection_truth(truth_path)
    roles = {row_key(row): "FP" for row in detection_rows}
    for frame in sorted({int_value(row["frame_index"], name="frame_index") for row in detection_rows}):
        frame_detections = [
            row for row in detection_rows if int_value(row["frame_index"], name="frame_index") == frame
        ]
        frame_truth = [
            row for row in truth.get("particles", []) if int(row["frame_index"]) == frame
        ]
        pairs: list[tuple[float, int, int]] = []
        for det_index, det in enumerate(frame_detections):
            det_box = bbox_from_row(det)
            for truth_index, truth_row in enumerate(frame_truth):
                score = iou(
                    det_box,
                    PatchBox(
                        top=int(math.floor(float(truth_row["top"]))),
                        left=int(math.floor(float(truth_row["left"]))),
                        bottom=int(math.ceil(float(truth_row["bottom"]))),
                        right=int(math.ceil(float(truth_row["right"]))),
                    ),
                )
                if score >= iou_threshold:
                    pairs.append((score, det_index, truth_index))
        matched_detections: set[int] = set()
        matched_truth: set[int] = set()
        for _score, det_index, truth_index in sorted(pairs, reverse=True):
            if det_index in matched_detections or truth_index in matched_truth:
                continue
            matched_detections.add(det_index)
            matched_truth.add(truth_index)
            roles[row_key(frame_detections[det_index])] = "TP"
    return roles


def iou(a: PatchBox, b: PatchBox) -> float:
    top = max(a.top, b.top)
    left = max(a.left, b.left)
    bottom = min(a.bottom, b.bottom)
    right = min(a.right, b.right)
    if bottom <= top or right <= left:
        return 0.0
    intersection = float((bottom - top) * (right - left))
    union = float(a.height * a.width + b.height * b.width) - intersection
    return intersection / union if union > 0 else 0.0


def error_taxonomy(feature: Mapping[str, Any], *, role: str) -> str:
    valid = int_value(feature["valid_revisits"], name="valid_revisits")
    hard_reject = bool_value(feature["hard_reject"])
    max_ratio = float(feature["max_recurrence_ratio"])
    if valid == 0:
        return f"{role.lower()}_no_valid_revisits"
    if role == "FP" and hard_reject:
        return "fp_recurrent_removed"
    if role == "FP" and max_ratio < DEFAULT_HARD_RATIO_THRESHOLD:
        return "fp_low_recurrence_evidence"
    if role == "TP" and hard_reject:
        return "tp_high_recurrence_accidentally_removed"
    if max_ratio >= DEFAULT_HARD_RATIO_THRESHOLD:
        return f"{role.lower()}_recurrent_but_not_hard_rejected"
    return f"{role.lower()}_inconclusive_low_recurrence"


def write_recurrence_report(
    path: Path,
    feature_rows: Sequence[Mapping[str, Any]],
    *,
    truth_path: Path | None,
    compare_summary_csv: Path | None,
    hard_run_dir: Path,
    rerank_run_dir: Path,
) -> None:
    taxonomy_counts: dict[str, int] = {}
    for row in feature_rows:
        taxonomy = str(row.get("error_taxonomy", "unknown"))
        taxonomy_counts[taxonomy] = taxonomy_counts.get(taxonomy, 0) + 1
    n_rejected = sum(bool_value(row["hard_reject"]) for row in feature_rows)
    n_fp_removed = sum(
        bool_value(row["hard_reject"]) and row.get("raw_match_role") == "FP"
        for row in feature_rows
    )
    n_tp_removed = sum(
        bool_value(row["hard_reject"]) and row.get("raw_match_role") == "TP"
        for row in feature_rows
    )
    lines = [
        "# YOLO11 + BeltMap recurrence post-filter",
        "",
        f"Truth path: `{truth_path}`" if truth_path is not None else "Truth path: n/a",
        f"Detections scored: {len(feature_rows)}",
        f"Hard-filter rejected detections: {n_rejected}",
        f"YOLO false positives removed: {n_fp_removed}",
        f"YOLO true positives accidentally removed: {n_tp_removed}",
        f"Hard-filter run: `{hard_run_dir}`",
        f"Rerank run: `{rerank_run_dir}`",
        f"Compare summary: `{compare_summary_csv}`" if compare_summary_csv else "Compare summary: n/a",
        "",
        "## Error taxonomy",
        "",
        "| category | count |",
        "| --- | ---: |",
    ]
    for key, count in sorted(taxonomy_counts.items()):
        lines.append(f"| {key} | {count} |")
    if compare_summary_csv and compare_summary_csv.is_file():
        lines.extend(["", "## Comparison summary", ""])
        lines.extend(compare_summary_csv.read_text(encoding="utf-8").splitlines())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_contact_sheet(
    path: Path,
    feature_rows: Sequence[Mapping[str, Any]],
    *,
    source_images: Mapping[int, Path],
    config: YoloRecurrenceConfig,
    crop_cache: dict[int, NDArray[np.floating]],
) -> None:
    selected = select_contact_rows(feature_rows)
    tile_w, tile_h = 220, 190
    columns = ["original", "previous", "next"]
    image = Image.new("RGB", (tile_w * len(columns), tile_h * max(1, len(selected))), "white")
    draw = ImageDraw.Draw(image)
    for row_index, row in enumerate(selected):
        for col_index, column in enumerate(columns):
            frame_value = row["frame_index"] if column == "original" else row.get(f"revisit_frame_{'prev' if column == 'previous' else 'next'}", "")
            if frame_value == "":
                tile = Image.new("RGB", (tile_w, tile_h), (245, 245, 245))
            else:
                frame = int_value(frame_value, name="frame_index")
                crop = load_crop(frame, source_images=source_images, crop_cache=crop_cache, region=config.belt_region)
                tile = crop_thumbnail(crop, tile_w, tile_h)
            x0 = col_index * tile_w
            y0 = row_index * tile_h
            image.paste(tile, (x0, y0))
            draw.rectangle((x0, y0, x0 + tile_w - 1, y0 + tile_h - 1), outline=(40, 40, 40))
            label = (
                f"{column} f{frame_value}\n"
                f"det {row['frame_index']}:{row['label']} {row.get('raw_match_role', '')}\n"
                f"ratio {row.get('max_recurrence_ratio', '')} hard {row.get('hard_reject', '')}"
            )
            draw.multiline_text((x0 + 4, y0 + 4), label, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def select_contact_rows(feature_rows: Sequence[Mapping[str, Any]], *, limit: int = 8) -> list[Mapping[str, Any]]:
    rejected = [row for row in feature_rows if bool_value(row.get("hard_reject"))]
    false_positives = [row for row in feature_rows if row.get("raw_match_role") == "FP"]
    true_positive_high = [
        row
        for row in feature_rows
        if row.get("raw_match_role") == "TP" and float(row.get("max_recurrence_ratio") or 0.0) >= DEFAULT_HARD_RATIO_THRESHOLD
    ]
    chosen: list[Mapping[str, Any]] = []
    for pool in (rejected, false_positives, true_positive_high, list(feature_rows)):
        for row in sorted(pool, key=lambda item: float(item.get("max_recurrence_ratio") or 0.0), reverse=True):
            if row not in chosen:
                chosen.append(row)
            if len(chosen) >= limit:
                return chosen
    return chosen


def crop_thumbnail(crop: NDArray[np.floating], width: int, height: int) -> Image.Image:
    arr = np.asarray(crop, dtype=np.float64)
    low, high = np.percentile(arr[np.isfinite(arr)], [1, 99]) if np.isfinite(arr).any() else (0.0, 1.0)
    if high <= low:
        high = low + 1.0
    scaled = np.clip((arr - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(scaled, mode="L").convert("RGB").resize((width, height))


def bbox_from_row(row: Mapping[str, Any]) -> PatchBox:
    return PatchBox(
        top=int_value(row["bbox_top"], name="bbox_top"),
        left=int_value(row["bbox_left"], name="bbox_left"),
        bottom=int_value(row["bbox_bottom"], name="bbox_bottom"),
        right=int_value(row["bbox_right"], name="bbox_right"),
    )


def row_key(row: Mapping[str, Any]) -> tuple[int, int]:
    return (
        int_value(row["frame_index"], name="frame_index"),
        int_value(row["label"], name="label"),
    )


def load_phase_px_by_frame(path: Path, *, frame_count: int) -> list[float]:
    phases: list[float | None] = [None for _ in range(frame_count)]
    for row in read_csv_rows(path):
        frame = int_value(row.get("frame_index"), name="frame_index")
        if 0 <= frame < frame_count:
            phases[frame] = float_value(row.get("phase_px"), name="phase_px")
    missing = [index for index, value in enumerate(phases) if value is None]
    if missing:
        preview = ", ".join(str(index) for index in missing[:8])
        raise ValueError(f"phase_estimates.csv is missing {len(missing)} frame(s); first: {preview}")
    return [float(value) for value in phases]


def find_source_images(source_image_dir: Path) -> dict[int, Path]:
    if not source_image_dir.is_dir():
        raise FileNotFoundError(source_image_dir)
    result: dict[int, Path] = {}
    for path in sorted(source_image_dir.rglob("*"), key=natural_key):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        frame = infer_frame_index(path.stem)
        if frame in result:
            raise ValueError(f"duplicate source image frame index {frame}: {result[frame]} and {path}")
        result[frame] = path
    if not result:
        raise ValueError(f"no source images found below {source_image_dir}")
    return result


def parse_belt_region(value: str) -> BeltRegion:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("belt region must be top,left,height,width")
    top, left, height, width = [int(float(part)) for part in parts]
    return BeltRegion(top=top, left=left, height=height, width=width)


def metadata_float(metadata: Mapping[str, Any], key: str, *, default: float | None = None) -> float:
    value = metadata.get(key, default)
    if value is None:
        raise ValueError(f"metadata.json is missing required field {key!r}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"metadata field {key!r} must be finite")
    return parsed


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    rows, _fieldnames = read_csv_rows_with_fieldnames(path)
    return rows


def read_csv_rows_with_fieldnames(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def float_value(value: Any, *, name: str) -> float:
    if value in (None, ""):
        raise ValueError(f"{name} is required")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def int_value(value: Any, *, name: str) -> int:
    parsed = float_value(value, name=name)
    if not parsed.is_integer():
        raise ValueError(f"{name} must be integer-valued")
    return int(parsed)


def bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def format_float(value: Any) -> str:
    parsed = float(value)
    if not math.isfinite(parsed):
        return ""
    return f"{parsed:.8f}"
