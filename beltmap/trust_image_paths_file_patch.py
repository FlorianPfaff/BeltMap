from __future__ import annotations

from pathlib import Path

from beltmap import trust as _trust


def file_only_image_paths(image_dir: Path) -> list[Path]:
    """Return naturally sorted regular image files below ``image_dir``.

    ``Path.rglob('*')`` yields directories as well as files. A directory whose
    name ends in a supported image extension must not be counted as an image or
    passed to downstream Pillow/hash readers.
    """

    return sorted(
        [
            path
            for path in image_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in _trust.IMAGE_EXTENSIONS
            and not path.name.startswith("._")
        ],
        key=_trust.natural_key,
    )


_trust.image_paths = file_only_image_paths
