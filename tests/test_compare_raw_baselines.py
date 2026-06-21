from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pytest
import numpy as np
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


@pytest.mark.parametrize("value", ["0,0,3.5,4", "-1,0,3,4"])
def test_parse_region_rejects_fractional_or_negative_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        crb.parse_region(value)


def test_crop_rejects_regions_outside_frame():
    frame = np.zeros((5, 5), dtype=np.float32)

    with pytest.raises(ValueError, match="exceeds"):
        crb.crop(frame, (4, 0, 2, 2))


def test_sample_indices_rejects_empty_population_or_sample_count():
    with pytest.raises(ValueError, match="count"):
        crb.sample_indices(0, 1)
    with pytest.raises(ValueError, match="sample_count"):
        crb.sample_indices(3, 0)


@pytest.mark.parametrize("percentiles", [(99, 1), (float("nan"), 99), (1, 101)])
def test_robust_display_scale_rejects_invalid_percentiles(percentiles):
    with pytest.raises(ValueError, match="percentiles"):
        crb.robust_display_scale([np.array([1.0, 2.0])], percentiles=percentiles)


def test_save_scaled_png_handles_nonfinite_values(tmp_path):
    path = tmp_path / "preview.png"

    crb.save_scaled_png(np.array([[0.0, np.nan, np.inf]]), path, scale=(0.0, 1.0))

    assert path.is_file()


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"frame_index": "-1"}, "frame_index"),
        ({"label": "0"}, "label"),
        ({"area_px": "0"}, "area_px"),
        ({"bbox_bottom": "1"}, "positive area"),
    ],
)
def test_parse_detection_rejects_invalid_geometry_or_ids(updates, message):
    row = {
        "frame_index": "0",
        "label": "1",
        "y": "2",
        "x": "3",
        "area_px": "4",
        "bbox_top": "1",
        "bbox_left": "1",
        "bbox_bottom": "3",
        "bbox_right": "3",
    }
    row.update(updates)

    with pytest.raises(ValueError, match=message):
        crb.parse_detection(row)


def test_infer_run_frame_count_rejects_boolean_or_fractional_metadata(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(ValueError, match="n_images"):
        crb.infer_run_frame_count(run_dir, [], {"n_images": True})
    with pytest.raises(ValueError, match="n_images"):
        crb.infer_run_frame_count(run_dir, [], {"n_images": 1.5})


def test_parse_optional_float_rejects_nonfinite_values():
    with pytest.raises(argparse.ArgumentTypeError, match="finite"):
        crb.parse_optional_float("nan")


@pytest.mark.parametrize("value", ["1.5", "-1", "nan"])
def test_parse_preview_frames_rejects_invalid_indices(value):
    with pytest.raises(ValueError, match="preview frame"):
        crb.parse_preview_frames(value, frame_count=10)


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
        (
            {"belt_velocity_px_per_frame": float("nan")},
            "--belt-velocity-px-per-frame must be finite",
        ),
        ({"low_threshold": 6.0}, "--low-threshold must be less than or equal"),
        (
            {"tracking_max_frame_gap": float("nan")},
            "--tracking-max-frame-gap must be finite",
        ),
        (
            {
                "track_filter_min_velocity_ratio_y": 1.2,
                "track_filter_max_velocity_ratio_y": 1.0,
            },
            "--track-filter-min-velocity-ratio-y must be less than or equal",
        ),
    ],
)
def test_raw_baseline_numeric_args_reject_invalid_values(updates, message):
    with pytest.raises(SystemExit, match=message):
        crb.validate_numeric_args(valid_args(**updates))


def test_raw_baseline_summary_preserves_zero_detections_per_frame(tmp_path):
    crb.write_summary(
        [
            {
                "label": "raw_zscore",
                "source_run": "",
                "same_tracker_recomputed": False,
                "n_images": 4,
                "n_detections": 0,
                "detections_per_frame": 0.0,
                "n_tracks": 0,
                "n_velocity_estimates": 0,
                "n_filtered_velocity_estimates": 0,
                "detection_area_median_px": None,
                "elapsed_s": 0.25,
                "output_dir": str(tmp_path / "raw_zscore"),
            }
        ],
        tmp_path,
    )

    report = (tmp_path / "raw_baseline_summary.md").read_text(encoding="utf-8")

    assert "| raw_zscore | 0 | 0 | 0 | 0 | 0 |  | 0.2 |" in report
    assert "nan" not in report.lower()
