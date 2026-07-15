from __future__ import annotations

from pathlib import Path
from typing import Any

from beltmap.cli import annotation_audit_review as _annotation_audit_review
from beltmap.yolo_export import IMAGE_EXTENSIONS, infer_frame_index, natural_key

_PATCHED_ATTR = "_beltmap_annotation_audit_source_image_patched"
_ORIGINAL_ATTR = "_beltmap_annotation_audit_original_find_source_image"


def _unwrap_patched_callable(func: Any) -> Any:
    return getattr(func, _ORIGINAL_ATTR, func)


_original_find_source_image = _unwrap_patched_callable(
    _annotation_audit_review.find_source_image
)


def supported_source_image_lookup(source_image_dir: Path, frame_index: int) -> Path | None:
    """Find a source image for ``frame_index`` across supported image formats.

    Keep the historical Brick-20g exact-path preference, then fall back to the
    same extension set and frame-index inference used by the YOLO export path.
    The old implementation only globbed a BMP 5-digit suffix and a PNG 6-digit
    suffix, which made JPEG/TIFF context frames appear missing in the review UI
    even though those images are supported elsewhere in BeltMap.
    """

    direct = _original_find_source_image(source_image_dir, frame_index)
    if direct is not None and direct.is_file():
        return direct

    for path in sorted(source_image_dir.rglob("*"), key=natural_key):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            inferred_frame = infer_frame_index(path.stem)
        except ValueError:
            continue
        if inferred_frame == frame_index:
            return path
    return None


setattr(supported_source_image_lookup, _PATCHED_ATTR, True)
setattr(supported_source_image_lookup, _ORIGINAL_ATTR, _original_find_source_image)
_annotation_audit_review.find_source_image = supported_source_image_lookup
