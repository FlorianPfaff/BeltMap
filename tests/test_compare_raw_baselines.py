from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts import compare_raw_baselines as crb


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def valid_args(**updates):
    values = {
        "belt_velocity_px_per_frame": 59.16,
        "threshold": 5.0,
        "low_threshold": None,
        "min_area_px": 4,
        "max_area_px": None,
        "min_bbox_width_px": 3,
        "min_bbox_height_px": 3,
        "max_bbox_aspect_ratio": 4.0,
        "min_bbox_extent": 0.15,
        "split_min_projection_gap_px": 1,
        "split_min_component_area_px": 4,
        "min_track_length": 2,
        "tracking_max_frame_gap": 2.0,
        "track_filter_min_length": 5,
        "track_filter_min_velocity_ratio_y": 0.0,
        "track_filter_max_velocity_ratio_y": 1.1,
        "track_filter_max_abs_x_velocity_px_per_frame": None,
    }
    values.update(updates)
    return argparse.Namespace(**values)


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
    write_csv(run_dir / "detections.csv", [], crb.DETECTION_FIELDS)

    with pytest.raises(ValueError, match="frame_stride"):
        crb.load_existing_beltmap_detections(
            run_dir,
            paths=[image_path],
            image_dir=image_dir,
            current_frame_stride=1,
            strict_frame_match=True,
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"threshold": float("nan")}, "--threshold must be finite"),
        ({"belt_velocity_px_per_frame": float("nan")}, "--belt-velocity-px-per-frame must be finite"),
        ({"low_threshold": 6.0}, "--low-threshold must be less than or equal"),
        ({"tracking_max_frame_gap": float("nan")}, "--tracking-max-frame-gap must be finite"),
        (
            {"track_filter_min_velocity_ratio_y": 1.2, "track_filter_max_velocity_ratio_y": 1.0},
            "--track-filter-min-velocity-ratio-y must be less than or equal",
        ),
    ],
)
def test_raw_baseline_numeric_args_reject_invalid_values(updates, message):
    with pytest.raises(SystemExit, match=message):
        crb.validate_numeric_args(valid_args(**updates))
