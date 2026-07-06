from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from beltmap import yolo_recurrence as _yolo_recurrence

_PATCHED_ATTR = "_beltmap_yolo_recurrence_contact_patched"
_ORIGINAL_ATTR = "_beltmap_yolo_recurrence_contact_original"


_LABEL_H = 46
_PATCH_MARGIN = 4


def _unwrap_patched_callable(func: Any) -> Any:
    return getattr(func, _ORIGINAL_ATTR, func)


def _float_or_zero(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if np.isfinite(parsed) else 0.0


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _int_value(value: Any) -> int:
    return int(float(value))


def _int_or_zero(value: Any) -> int:
    if value in (None, ""):
        return 0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0
    return int(parsed) if np.isfinite(parsed) else 0


def _patch_box_from_feature(row: Mapping[str, Any], column: str) -> _yolo_recurrence.PatchBox | None:
    top = _int_value(row["patch_top"])
    left = _int_value(row["patch_left"])
    bottom = _int_value(row["patch_bottom"])
    right = _int_value(row["patch_right"])
    height = bottom - top
    width = right - left
    if height <= 0 or width <= 0:
        return None
    if column == "original":
        return _yolo_recurrence.PatchBox(top=top, left=left, bottom=bottom, right=right)
    suffix = "prev" if column == "previous" else "next"
    revisit_y = row.get(f"revisit_y_{suffix}")
    if revisit_y in (None, ""):
        return None
    center_x = 0.5 * (left + right)
    try:
        return _yolo_recurrence.centered_patch_box(
            y=float(revisit_y),
            x=center_x,
            height=height,
            width=width,
            image_shape=(10**9, 10**9),
        )
    except ValueError:
        return None


def _clip_patch_to_crop(
    patch: _yolo_recurrence.PatchBox,
    crop_shape: tuple[int, int],
) -> _yolo_recurrence.PatchBox:
    return _yolo_recurrence.clipped_patch_box(
        top=patch.top,
        left=patch.left,
        bottom=patch.bottom,
        right=patch.right,
        image_shape=crop_shape,
    )


def _feature_sort_key(row: Mapping[str, Any]) -> tuple[int, float, float, int]:
    """Rank contact-sheet rows by actual shape-supported recurrence evidence.

    The original contact-sheet selector sorted mostly by ``max_recurrence_ratio``.
    That highlighted dense-flow coincidences where the same belt coordinate was
    bright in another revolution but the patch shape was not actually correlated.
    Use the same shape-supported evidence as the hard filter/rerank path.
    """

    return (
        1 if _bool_value(row.get("hard_reject")) else 0,
        _float_or_zero(row.get("belt_fixedness_score")),
        _float_or_zero(row.get("max_recurrence_ratio")),
        _int_or_zero(row.get("valid_revisits", 0)),
    )


def contact_sheet_rows(
    feature_rows: Sequence[Mapping[str, Any]],
    *,
    limit: int = 8,
) -> list[Mapping[str, Any]]:
    rejected = [row for row in feature_rows if _bool_value(row.get("hard_reject"))]
    false_positives = [row for row in feature_rows if row.get("raw_match_role") == "FP"]
    true_positive_supported = [
        row
        for row in feature_rows
        if row.get("raw_match_role") == "TP" and _float_or_zero(row.get("belt_fixedness_score")) > 0.0
    ]
    chosen: list[Mapping[str, Any]] = []
    for pool in (rejected, false_positives, true_positive_supported, list(feature_rows)):
        for row in sorted(pool, key=_feature_sort_key, reverse=True):
            if row not in chosen:
                chosen.append(row)
            if len(chosen) >= limit:
                return chosen
    return chosen


def _thumbnail_patch(patch: NDArray[np.floating], width: int, height: int) -> Image.Image:
    arr = np.asarray(patch, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    low, high = np.percentile(finite, [1, 99]) if finite.size else (0.0, 1.0)
    if high <= low:
        high = low + 1.0
    scaled = np.clip((arr - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(scaled, mode="L").convert("RGB").resize((width, height))


def _draw_original_detection_box(
    draw: ImageDraw.ImageDraw,
    row: Mapping[str, Any],
    patch: _yolo_recurrence.PatchBox,
    *,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> None:
    bbox_left = _float_or_zero(row.get("bbox_left")) - patch.left
    bbox_top = _float_or_zero(row.get("bbox_top")) - patch.top
    bbox_right = _float_or_zero(row.get("bbox_right")) - patch.left
    bbox_bottom = _float_or_zero(row.get("bbox_bottom")) - patch.top
    sx = width / max(1, patch.width)
    sy = height / max(1, patch.height)
    draw.rectangle(
        (
            x0 + bbox_left * sx,
            y0 + _LABEL_H + bbox_top * sy,
            x0 + bbox_right * sx,
            y0 + _LABEL_H + bbox_bottom * sy,
        ),
        outline=(255, 0, 255),
        width=2,
    )


_original_write_contact_sheet = _unwrap_patched_callable(_yolo_recurrence.write_contact_sheet)
_original_select_contact_rows = _unwrap_patched_callable(_yolo_recurrence.select_contact_rows)


def patch_focused_write_contact_sheet(
    path: Any,
    feature_rows: Sequence[Mapping[str, Any]],
    *,
    source_images: Mapping[int, Any],
    config: Any,
    crop_cache: dict[int, NDArray[np.floating]],
) -> None:
    """Write a contact sheet centered on the scored detection/revisit patches.

    The unpatched implementation downscaled the whole BeltMap crop into each
    cell.  On 1800x1330 crops, YOLO boxes and projected revisits become nearly
    invisible, which hides the exact recurrence evidence that the diagnostic is
    supposed to audit.  This view crops to the detection patch and corresponding
    previous/next projected belt-coordinate patches.
    """

    selected = contact_sheet_rows(feature_rows)
    tile_w, tile_h = 220, 190
    patch_h = tile_h - _LABEL_H
    columns = ("original", "previous", "next")
    image = Image.new("RGB", (tile_w * len(columns), tile_h * max(1, len(selected))), "white")
    draw = ImageDraw.Draw(image)

    for row_index, row in enumerate(selected):
        for col_index, column in enumerate(columns):
            suffix = "prev" if column == "previous" else "next"
            frame_value = row.get("frame_index") if column == "original" else row.get(f"revisit_frame_{suffix}", "")
            x0 = col_index * tile_w
            y0 = row_index * tile_h
            patch = None
            if frame_value not in (None, ""):
                try:
                    frame = _int_value(frame_value)
                    crop = _yolo_recurrence.load_crop(
                        frame,
                        source_images=source_images,
                        crop_cache=crop_cache,
                        region=config.belt_region,
                    )
                    candidate_patch = _patch_box_from_feature(row, column)
                    if candidate_patch is not None:
                        patch = _clip_patch_to_crop(candidate_patch, crop.shape)
                        patch_arr = crop[patch.top : patch.bottom, patch.left : patch.right]
                        tile = _thumbnail_patch(patch_arr, tile_w, patch_h)
                    else:
                        tile = Image.new("RGB", (tile_w, patch_h), (245, 245, 245))
                except (OSError, ValueError):
                    tile = Image.new("RGB", (tile_w, patch_h), (245, 245, 245))
            else:
                tile = Image.new("RGB", (tile_w, patch_h), (245, 245, 245))

            image.paste(tile, (x0, y0 + _LABEL_H))
            draw.rectangle((x0, y0, x0 + tile_w - 1, y0 + tile_h - 1), outline=(40, 40, 40))
            if patch is not None and column == "original":
                _draw_original_detection_box(
                    draw,
                    row,
                    patch,
                    x0=x0,
                    y0=y0,
                    width=tile_w,
                    height=patch_h,
                )
            label = (
                f"{column} f{frame_value}\n"
                f"det {row.get('frame_index')}:{row.get('label')} {row.get('raw_match_role', '')}\n"
                f"fixed {row.get('belt_fixedness_score', '')} hard {row.get('hard_reject', '')}"
            )
            draw.multiline_text((x0 + 4, y0 + 4), label, fill=(0, 0, 0))

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


setattr(patch_focused_write_contact_sheet, _PATCHED_ATTR, True)
setattr(patch_focused_write_contact_sheet, _ORIGINAL_ATTR, _original_write_contact_sheet)
_yolo_recurrence.select_contact_rows = contact_sheet_rows
_yolo_recurrence.write_contact_sheet = patch_focused_write_contact_sheet
