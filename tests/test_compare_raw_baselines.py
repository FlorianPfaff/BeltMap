import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.compare_raw_baselines import (
    DETECTION_FIELDS,
    load_existing_beltmap_detections,
)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_load_existing_beltmap_detections_rejects_fractional_frame_stride(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "frame_000.bmp"
    Image.new("L", (4, 4), 64).save(image_path)

    run_dir = tmp_path / "beltmap"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        json.dumps({"n_images": 1, "frame_stride": 1.5}),
        encoding="utf-8",
    )
    write_csv(run_dir / "detections.csv", [], DETECTION_FIELDS)

    with pytest.raises(ValueError, match="frame_stride"):
        load_existing_beltmap_detections(
            run_dir,
            paths=[image_path],
            image_dir=image_dir,
            current_frame_stride=1,
            strict_frame_match=True,
        )
