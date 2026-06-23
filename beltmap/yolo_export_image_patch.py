from __future__ import annotations

from pathlib import Path

from PIL import Image

from beltmap import yolo_export as _yolo_export


_original_yolo_prediction_to_detection_row = _yolo_export.yolo_prediction_to_detection_row


def duplicate_safe_find_images(
    images_dir: Path,
    *,
    frame_index_pattern: str = _yolo_export.DEFAULT_FRAME_INDEX_PATTERN,
) -> dict[str, _yolo_export.ImageRecord]:
    """Return image records and reject ambiguous stems/frame indices.

    YOLO labels are matched to images by stem, while BeltMap comparison uses one
    detection-count row per frame index.  Silently accepting duplicate stems or
    multiple image files that parse to the same frame index can make labels attach
    to the wrong image or emit duplicate per-frame rows.  Treat those cases as
    invalid input instead of overwriting an earlier image record.
    """

    if not images_dir.is_dir():
        raise FileNotFoundError(images_dir)

    records: dict[str, _yolo_export.ImageRecord] = {}
    frame_paths: dict[int, Path] = {}
    for path in sorted(images_dir.rglob("*"), key=_yolo_export.natural_key):
        if path.suffix.lower() not in _yolo_export.IMAGE_EXTENSIONS:
            continue
        frame_index = _yolo_export.infer_frame_index(path.stem, pattern=frame_index_pattern)
        existing_stem = records.get(path.stem)
        if existing_stem is not None:
            raise ValueError(
                f"duplicate image stem {path.stem!r}: {existing_stem.path} and {path}"
            )
        existing_frame_path = frame_paths.get(frame_index)
        if existing_frame_path is not None:
            raise ValueError(
                f"duplicate image frame index {frame_index}: {existing_frame_path} and {path}"
            )
        with Image.open(path) as image:
            width, height = image.size
        records[path.stem] = _yolo_export.ImageRecord(
            path=path,
            stem=path.stem,
            frame_index=frame_index,
            width=int(width),
            height=int(height),
        )
        frame_paths[frame_index] = path
    if not records:
        raise ValueError(f"no supported images found below {images_dir}")
    return records


def discrete_area_yolo_prediction_to_detection_row(
    prediction: _yolo_export.YoloPrediction,
    *,
    image: _yolo_export.ImageRecord,
    label_index: int,
    source: str,
) -> dict[str, str]:
    """Return a BeltMap detection row with area matching the emitted integer box.

    ``beltmap.yolo_export`` emits integer half-open ``bbox_*`` fields by flooring
    the top/left edges and ceiling the bottom/right edges.  The original exporter
    computed ``area_px`` from the unclipped floating-point YOLO box before that
    integer expansion.  Tiny or highly fractional YOLO boxes can therefore report
    an area that is smaller than the exported half-open box area, which affects
    downstream small-component statistics and any area-based filtering.  Use the
    discrete emitted box area instead.
    """

    row = _original_yolo_prediction_to_detection_row(
        prediction,
        image=image,
        label_index=label_index,
        source=source,
    )
    top = int(row["bbox_top"])
    left = int(row["bbox_left"])
    bottom = int(row["bbox_bottom"])
    right = int(row["bbox_right"])
    row["area_px"] = str((bottom - top) * (right - left))
    return row


_yolo_export.find_images = duplicate_safe_find_images
_yolo_export.yolo_prediction_to_detection_row = discrete_area_yolo_prediction_to_detection_row
