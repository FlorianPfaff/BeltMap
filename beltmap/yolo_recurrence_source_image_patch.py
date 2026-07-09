from __future__ import annotations

from pathlib import Path
from typing import Any

from beltmap import yolo_recurrence as _yolo_recurrence

_PATCHED_ATTR = "_beltmap_yolo_recurrence_source_images_patched"
_ORIGINAL_ATTR = "_beltmap_yolo_recurrence_source_images_original"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the original callable behind our wrapper, if already patched."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_find_source_images = _unwrap_patched_callable(_yolo_recurrence.find_source_images)


def skip_auxiliary_find_source_images(source_image_dir: Path) -> dict[int, Path]:
    """Return parseable source-frame images while ignoring auxiliary image assets.

    YOLO recurrence operates on one source image per frame. Prediction folders and
    debug directories may also contain supported image files such as previews,
    logos, or contact sheets whose stems do not encode a frame index. Ignore
    those auxiliary assets, while preserving duplicate-frame validation for
    parseable source-frame images.
    """

    if not source_image_dir.is_dir():
        raise FileNotFoundError(source_image_dir)

    result: dict[int, Path] = {}
    for path in sorted(source_image_dir.rglob("*"), key=_yolo_recurrence.natural_key):
        if path.suffix.lower() not in _yolo_recurrence.IMAGE_EXTENSIONS:
            continue
        try:
            frame = _yolo_recurrence.infer_frame_index(path.stem)
        except ValueError:
            continue
        if frame in result:
            raise ValueError(
                f"duplicate source image frame index {frame}: {result[frame]} and {path}"
            )
        result[frame] = path
    if not result:
        raise ValueError(f"no source images found below {source_image_dir}")
    return result


setattr(skip_auxiliary_find_source_images, _PATCHED_ATTR, True)
setattr(
    skip_auxiliary_find_source_images,
    _ORIGINAL_ATTR,
    _original_find_source_images,
)

_yolo_recurrence.find_source_images = skip_auxiliary_find_source_images
