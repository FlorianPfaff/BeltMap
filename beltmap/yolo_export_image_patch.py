from __future__ import annotations

from pathlib import Path

from PIL import Image

from beltmap import yolo_export as _yolo_export


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


_yolo_export.find_images = duplicate_safe_find_images
