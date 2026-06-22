import csv
import json

import numpy as np
import pytest
from PIL import Image

from beltmap.trust import (
    _metadata_crop_shape,
    confidence_rows,
    edge_audit_rows,
    events_from_tracks,
    frame_quality_metrics,
    parse_region,
    physical_validation_summary,
    plan_map_epochs,
    quality_report,
    run_drift_report,
    scale_calibration_from_points,
    sequence_report,
    write_csv_rows,
    write_json,
    write_run_trust_artifacts,
)


def test_sequence_report_detects_missing_and_duplicate_frames(tmp_path):
    arr = np.zeros((4, 5), dtype=np.uint8)
    for name in ["frame_000.png", "frame_002.png", "other_002.png"]:
        Image.fromarray(arr).save(tmp_path / name)

    report = sequence_report(tmp_path)

    assert report["n_images"] == 3
    assert report["missing_frame_numbers"] == [1]
    assert 2 in report["duplicate_frame_numbers"]


def test_write_json_sanitizes_nonfinite_numpy_scalars(tmp_path):
    path = tmp_path / "payload.json"

    write_json(path, {"bad": np.float64(float("nan"))})
    text = path.read_text(encoding="utf-8")

    assert "NaN" not in text
    assert '"bad": null' in text


def test_write_csv_rows_sanitizes_nonfinite_numpy_scalars(tmp_path):
    path = tmp_path / "payload.csv"

    write_csv_rows(path, [{"value": np.float64(float("nan"))}], ["value"])

    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))

    assert rows[0]["value"] == ""


def test_parse_region_rejects_negative_fractional_or_boolean_values():
    for value in ["-1,0,10,10", "0,0,10.5,10", [True, 0, 10, 10]]:
        with pytest.raises(ValueError):
            parse_region(value)


def test_frame_quality_metrics_rejects_invalid_region_and_thresholds(tmp_path):
    Image.fromarray(np.zeros((4, 5), dtype=np.uint8)).save(tmp_path / "frame.png")

    with pytest.raises(ValueError, match="bounds"):
        frame_quality_metrics(tmp_path / "frame.png", region=(0, 0, 8, 5))
    with pytest.raises(ValueError, match="greater"):
        frame_quality_metrics(
            tmp_path / "frame.png",
            saturation_threshold=5.0,
            dark_threshold=5.0,
        )


def test_quality_report_rejects_nonpositive_sample_limit(tmp_path):
    with pytest.raises(ValueError, match="sample_limit"):
        quality_report(tmp_path, sample_limit=0)


def test_quality_report_rejects_fractional_sample_limit(tmp_path):
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(tmp_path / "frame_000.png")

    with pytest.raises(ValueError, match="sample_limit"):
        quality_report(tmp_path, sample_limit=1.5)


def test_edge_audit_flags_truncated_detection():
    rows = [
        {
            "frame_index": "0",
            "bbox_top": "0",
            "bbox_left": "4",
            "bbox_bottom": "3",
            "bbox_right": "8",
            "area_px": "8",
        },
        {
            "frame_index": "0",
            "bbox_top": "5",
            "bbox_left": "5",
            "bbox_bottom": "8",
            "bbox_right": "9",
            "area_px": "8",
        },
    ]

    audited = edge_audit_rows(rows, height=10, width=10)

    assert audited[0]["touches_top_edge"] is True
    assert audited[0]["is_truncated"] is True
    assert audited[1]["is_truncated"] is False


def test_edge_audit_rejects_invalid_geometry_and_skips_degenerate_boxes():
    with pytest.raises(ValueError, match="height and width"):
        edge_audit_rows([], height=0, width=10)
    with pytest.raises(ValueError, match="margin"):
        edge_audit_rows([], height=10, width=10, margin_px=-1)

    rows = edge_audit_rows(
        [
            {
                "bbox_top": "4",
                "bbox_left": "1",
                "bbox_bottom": "4",
                "bbox_right": "3",
            },
            {
                "bbox_top": "1",
                "bbox_left": "1",
                "bbox_bottom": "4",
                "bbox_right": "3",
            },
        ],
        height=10,
        width=10,
    )

    assert len(rows) == 1
    assert rows[0]["bbox_top"] == "1"


def test_events_from_tracks_aggregates_rows():
    rows = [
        {"track_id": "4", "frame_index": "0", "y": "1", "x": "2", "peak_signal": "5"},
        {"track_id": "4", "frame_index": "1", "y": "3", "x": "2", "peak_signal": "7"},
    ]

    events = events_from_tracks(rows)

    assert len(events) == 1
    assert events[0]["track_id"] == 4
    assert events[0]["n_observations"] == 2
    assert events[0]["velocity_y_px_per_frame"] == 2.0


def test_events_from_tracks_requires_aligned_frame_coordinate_pairs():
    rows = [
        {"track_id": "4", "frame_index": "0", "y": ""},
        {"track_id": "4", "frame_index": "", "y": "100"},
        {"track_id": "4", "frame_index": "10", "y": "20"},
    ]

    events = events_from_tracks(rows)

    assert events[0]["velocity_y_px_per_frame"] == ""


def test_events_from_tracks_accepts_float_formatted_track_ids():
    rows = [
        {"track_id": "4.0", "frame_index": "0", "y": "1", "x": "2"},
        {"track_id": "4.5", "frame_index": "1", "y": "3", "x": "2"},
    ]

    events = events_from_tracks(rows)

    assert len(events) == 1
    assert events[0]["track_id"] == 4
    assert events[0]["n_observations"] == 1


def test_confidence_rows_penalizes_truncation_and_artifacts():
    rows = [
        {"peak_signal": "8", "area_px": "20", "is_truncated": "False"},
        {
            "peak_signal": "8",
            "area_px": "20",
            "is_truncated": "True",
            "recurrent_artifact_overlap_fraction": "0.5",
        },
    ]

    scored = confidence_rows(rows, threshold=5.0)

    assert scored[0]["detection_confidence"] > scored[1]["detection_confidence"]


def test_confidence_rows_rejects_invalid_threshold_and_clamps_overlap():
    with pytest.raises(ValueError, match="threshold"):
        confidence_rows([], threshold=0)

    base = confidence_rows(
        [
            {
                "peak_signal": "8",
                "area_px": "20",
                "recurrent_artifact_overlap_fraction": "0",
            }
        ],
        threshold=5.0,
    )[0]["detection_confidence"]
    negative_overlap = confidence_rows(
        [
            {
                "peak_signal": "8",
                "area_px": "20",
                "recurrent_artifact_overlap_fraction": "-1",
            }
        ],
        threshold=5.0,
    )[0]["detection_confidence"]
    overfull_overlap = confidence_rows(
        [
            {
                "peak_signal": "8",
                "area_px": "20",
                "recurrent_artifact_overlap_fraction": "2",
            }
        ],
        threshold=5.0,
    )[0]["detection_confidence"]

    assert negative_overlap == base
    assert overfull_overlap == 0.0


def test_write_run_trust_artifacts_falls_back_for_zero_detection_threshold(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "detection_threshold": 0.0,
                "belt_region": {"top": 0, "left": 0, "height": 10, "width": 10},
            }
        ),
        encoding="utf-8",
    )
    (out / "detections.csv").write_text(
        "frame_index,bbox_top,bbox_left,bbox_bottom,bbox_right,area_px,peak_signal\n"
        "0,2,2,4,4,20,0\n",
        encoding="utf-8",
    )

    write_run_trust_artifacts(output_dir=out)

    with (out / "detection_confidence.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert float(rows[0]["detection_confidence"]) == pytest.approx(0.26894142137)


def test_plan_map_epochs_with_overlap():
    epochs = plan_map_epochs(100, epoch_count=4, overlap_frames=5)

    assert len(epochs) == 4
    assert epochs[0]["frame_start"] == 0
    assert epochs[1]["train_frame_start"] == 20
    assert epochs[-1]["frame_stop"] == 100


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"frame_count": 100.5}, "frame_count"),
        ({"frame_count": 100, "epoch_count": 2.5}, "epoch_count"),
        ({"frame_count": 100, "epoch_length_frames": 10.5}, "epoch_length_frames"),
        ({"frame_count": 100, "overlap_frames": 1.5}, "overlap_frames"),
    ],
)
def test_plan_map_epochs_rejects_fractional_integer_config(kwargs, message):
    with pytest.raises(ValueError, match=message):
        plan_map_epochs(**kwargs)


def test_run_drift_report_uses_aligned_phase_rows(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "phase_estimates.csv").write_text(
        "frame_index,correction_px,loss\n" "0,,\n" "1,10,\n" "3,20,4\n" "5,,8\n",
        encoding="utf-8",
    )

    report = run_drift_report(out)

    assert report["phase_correction_slope_px_per_frame"] == 5.0
    assert report["registration_loss_slope_per_frame"] == 2.0


def test_metadata_crop_shape_ignores_invalid_nonnegative_fields():
    assert _metadata_crop_shape({"belt_region": {"height": -1, "width": True}}) == (
        0,
        0,
    )
    assert _metadata_crop_shape({"first_image_shape": ["nan", 20]}) == (0, 20)


def test_scale_calibration_from_points():
    calibration = scale_calibration_from_points((0, 0), (0, 100), known_distance_mm=50)

    assert calibration.px_per_mm == 2.0
    assert calibration.mm_per_px == 0.5


def test_scale_calibration_rejects_nonfinite_inputs():
    with pytest.raises(ValueError, match="point_a"):
        scale_calibration_from_points((0, float("nan")), (0, 100), known_distance_mm=50)

    with pytest.raises(ValueError, match="known_distance_mm"):
        scale_calibration_from_points((0, 0), (0, 100), known_distance_mm=float("nan"))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"frame_rate_hz": float("nan")}, "frame_rate_hz"),
        ({"analysis_duration_s": -1.0}, "analysis_duration_s"),
        ({"particle_mass_g": -0.1}, "particle_mass_g"),
        ({"expected_particle_flux_per_s": -1.0}, "expected_particle_flux_per_s"),
    ],
)
def test_physical_validation_rejects_invalid_numeric_inputs(tmp_path, kwargs, message):
    out = tmp_path / "outputs"
    out.mkdir()

    with pytest.raises(ValueError, match=message):
        physical_validation_summary(out, **kwargs)
