import csv
import json
from pathlib import Path

import numpy as np
import pytest

from beltmap.benchmark import (
    bbox_iou,
    circular_signed_error_px,
    compute_benchmark_metrics,
    detection_metrics,
    event_metrics,
    finite_int,
    generate_benchmark_report,
    source_frame_index,
    track_metrics,
)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_synthetic_benchmark_case(tmp_path: Path) -> tuple[Path, Path]:
    truth_dir = tmp_path / "data" / "images"
    output_dir = tmp_path / "outputs"
    truth_dir.mkdir(parents=True)
    output_dir.mkdir()

    true_belt = np.arange(24, dtype=np.float32).reshape(6, 4)
    reconstructed = np.roll(true_belt, shift=-1, axis=0)
    np.save(truth_dir / "true_belt_map.npy", true_belt)
    np.save(output_dir / "belt_map.npy", reconstructed)

    truth = {
        "frames": 3,
        "height": 6,
        "width": 4,
        "belt_period_px": 6,
        "belt_shift_px_per_frame": 2,
        "true_belt_velocity_y_px_per_frame": 2,
        "true_phase_px_by_frame": [0, 4, 2],
        "true_belt_map_npy": "true_belt_map.npy",
        "particle_shift_y_px_per_frame": 1,
        "true_particle_velocity_y_px_per_frame": 1,
        "true_velocity_ratio_y": 0.5,
        "particle_size_px": 2,
        "particles": [
            {"frame_index": 0, "top": 1, "left": 1, "bottom": 3, "right": 3},
            {"frame_index": 1, "top": 2, "left": 1, "bottom": 4, "right": 3},
            {"frame_index": 2, "top": 3, "left": 1, "bottom": 5, "right": 3},
        ],
    }
    truth_path = truth_dir / "synthetic_metadata.json"
    truth_path.write_text(json.dumps(truth), encoding="utf-8")

    write_csv(
        output_dir / "phase_estimates.csv",
        [
            {"frame_index": 0, "image": "frame_000.png", "phase_px": 0.0},
            {"frame_index": 1, "image": "frame_001.png", "phase_px": 4.1},
            {"frame_index": 2, "image": "frame_002.png", "phase_px": 1.9},
        ],
        ["frame_index", "image", "phase_px"],
    )
    write_csv(
        output_dir / "detections.csv",
        [
            {
                "frame_index": 0,
                "image": "frame_000.png",
                "bbox_top": 1,
                "bbox_left": 1,
                "bbox_bottom": 3,
                "bbox_right": 3,
                "y": 1.5,
                "x": 1.5,
            },
            {
                "frame_index": 1,
                "image": "frame_001.png",
                "bbox_top": 2,
                "bbox_left": 1,
                "bbox_bottom": 4,
                "bbox_right": 3,
                "y": 2.5,
                "x": 1.5,
            },
            {
                "frame_index": 2,
                "image": "frame_002.png",
                "bbox_top": 3,
                "bbox_left": 1,
                "bbox_bottom": 5,
                "bbox_right": 3,
                "y": 3.5,
                "x": 1.5,
            },
        ],
        [
            "frame_index",
            "image",
            "bbox_top",
            "bbox_left",
            "bbox_bottom",
            "bbox_right",
            "y",
            "x",
        ],
    )
    write_csv(
        output_dir / "velocities.csv",
        [
            {
                "track_id": 0,
                "n_detections": 3,
                "velocity_y_px_per_frame": 1.05,
                "velocity_ratio_y": 0.525,
            }
        ],
        ["track_id", "n_detections", "velocity_y_px_per_frame", "velocity_ratio_y"],
    )
    write_csv(
        output_dir / "tracks.csv",
        [
            {
                "track_id": 0,
                "track_detection_index": 0,
                "frame_index": 0,
                "image": "frame_000.png",
                "bbox_top": 1,
                "bbox_left": 1,
                "bbox_bottom": 3,
                "bbox_right": 3,
                "y": 1.5,
                "x": 1.5,
            },
            {
                "track_id": 0,
                "track_detection_index": 1,
                "frame_index": 1,
                "image": "frame_001.png",
                "bbox_top": 2,
                "bbox_left": 1,
                "bbox_bottom": 4,
                "bbox_right": 3,
                "y": 2.5,
                "x": 1.5,
            },
            {
                "track_id": 0,
                "track_detection_index": 2,
                "frame_index": 2,
                "image": "frame_002.png",
                "bbox_top": 3,
                "bbox_left": 1,
                "bbox_bottom": 5,
                "bbox_right": 3,
                "y": 3.5,
                "x": 1.5,
            },
        ],
        [
            "track_id",
            "track_detection_index",
            "frame_index",
            "image",
            "bbox_top",
            "bbox_left",
            "bbox_bottom",
            "bbox_right",
            "y",
            "x",
        ],
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "n_images": 3,
                "belt_velocity_px_per_frame": 2.0,
                "belt_map_height_px": 6,
                "n_phase_estimates": 3,
                "n_detections": 3,
                "n_tracks": 1,
                "n_velocity_estimates": 1,
                "elapsed_s": 1.5,
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "progress.jsonl").write_text(
        json.dumps({"stage": "done", "rss_mb": 123.4}) + "\n",
        encoding="utf-8",
    )
    return output_dir, truth_path


def test_circular_signed_error_uses_shortest_periodic_difference():
    assert circular_signed_error_px(1.0, 9.0, 10.0) == pytest.approx(2.0)
    assert circular_signed_error_px(9.0, 1.0, 10.0) == pytest.approx(-2.0)


def test_bbox_iou_for_overlapping_half_open_boxes():
    a = {"top": 0.0, "left": 0.0, "bottom": 3.0, "right": 3.0}
    b = {"top": 1.0, "left": 1.0, "bottom": 4.0, "right": 4.0}

    assert bbox_iou(a, b) == pytest.approx(4 / 14)


def test_finite_int_accepts_float_like_integer_strings():
    assert finite_int("7.0") == 7
    assert finite_int("7.5") is None


def test_source_frame_index_ignores_crop_coordinate_suffixes():
    assert source_frame_index({"image": "frame_000123_crop_0_220_1330_1800.png"}) == 123
    assert source_frame_index({"image": "sample_0042.png"}) == 42
    assert (
        source_frame_index(
            {
                "image": "sample_0042_crop_0_220.png",
                "frame_index": "7",
            }
        )
        == 7
    )


def test_detection_metrics_no_matches_report_zero_f1():
    truth = {
        "particles": [
            {
                "frame_index": 0,
                "top": 0,
                "left": 0,
                "bottom": 10,
                "right": 10,
            }
        ]
    }
    detections = [
        {
            "frame_index": "0",
            "bbox_top": "20",
            "bbox_left": "20",
            "bbox_bottom": "30",
            "bbox_right": "30",
            "y": "25",
            "x": "25",
        }
    ]

    metrics = detection_metrics(detections, truth, iou_threshold=0.5)

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_detection_metrics_accept_frame_box_truth_layout():
    truth = {
        "frames": [
            {
                "frame_index": 0,
                "boxes": [
                    {"top": 0, "left": 0, "bottom": 10, "right": 10, "event_id": "a"},
                    {"top": 20, "left": 20, "bottom": 25, "right": 25, "event_id": "b"},
                ],
            }
        ]
    }
    detections = [
        {
            "frame_index": "0",
            "bbox_top": "0",
            "bbox_left": "0",
            "bbox_bottom": "10",
            "bbox_right": "10",
            "y": "4.5",
            "x": "4.5",
        }
    ]

    detection = detection_metrics(detections, truth, iou_threshold=0.5)
    events = event_metrics(detections, truth, iou_threshold=0.5)

    assert detection["truth_boxes"] == 2
    assert detection["predicted_boxes"] == 1
    assert detection["true_positives"] == 1
    assert detection["false_negatives"] == 1
    assert detection["recall"] == pytest.approx(0.5)
    assert events["truth_events"] == 2
    assert events["matched_events"] == 1


def test_event_metrics_no_matches_report_zero_f1():
    truth = {
        "particles": [
            {"event_id": "truth", "frame_index": 0, "top": 0, "left": 0, "bottom": 10, "right": 10}
        ]
    }
    detections = [
        {
            "track_id": "pred",
            "frame_index": "0",
            "bbox_top": "20",
            "bbox_left": "20",
            "bbox_bottom": "30",
            "bbox_right": "30",
            "y": "25",
            "x": "25",
        }
    ]

    metrics = event_metrics(detections, truth, iou_threshold=0.5)

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_compute_benchmark_metrics_from_synthetic_truth(tmp_path):
    output_dir, truth_path = make_synthetic_benchmark_case(tmp_path)

    metrics = compute_benchmark_metrics(output_dir=output_dir, truth_path=truth_path)

    assert metrics["phase"]["rmse_px"] == pytest.approx(np.sqrt((0.0**2 + 0.1**2 + 0.1**2) / 3))
    assert metrics["case"]["frames"] == 3
    assert metrics["case"]["truth_boxes"] == 3
    assert metrics["belt_map"]["rmse_gray"] == pytest.approx(0.0)
    assert metrics["belt_map"]["best_cyclic_shift_px"] == 1
    assert metrics["detections"]["precision"] == pytest.approx(1.0)
    assert metrics["detections"]["recall"] == pytest.approx(1.0)
    assert metrics["detections"]["f1"] == pytest.approx(1.0)
    assert metrics["events"]["precision"] == pytest.approx(1.0)
    assert metrics["events"]["recall"] == pytest.approx(1.0)
    assert metrics["events"]["f1"] == pytest.approx(1.0)
    assert metrics["events"]["prediction_source"] == "tracks.csv"
    assert metrics["events"]["prediction_rows"] == 3
    assert metrics["events"]["truth_events"] == 1
    assert metrics["events"]["predicted_events"] == 1
    assert metrics["events"]["matched_events"] == 1
    assert metrics["events"]["birth_false_positive_rate"] == pytest.approx(0.0)
    assert metrics["events"]["missed_event_rate"] == pytest.approx(0.0)
    assert metrics["events"]["mean_truth_frame_coverage"] == pytest.approx(1.0)
    assert metrics["tracks"]["tracks"] == 1
    assert metrics["tracks"]["mean_track_length"] == pytest.approx(3.0)
    assert metrics["tracks"]["median_track_length"] == pytest.approx(3.0)
    assert metrics["tracks"]["single_frame_tracks"] == 0
    assert metrics["filtered_tracks"]["available"] is False
    assert metrics["filtered_events"]["available"] is False
    assert metrics["filtered_events"]["prediction_source"] == "filtered_tracks.csv"
    assert metrics["velocity"]["velocity_y_error_px_per_frame"] == pytest.approx(0.05)
    assert metrics["velocity"]["velocity_y_mean_abs_error_px_per_frame"] == pytest.approx(0.05)
    assert metrics["velocity"]["velocity_y_bias_px_per_frame"] == pytest.approx(0.05)
    assert metrics["velocity"]["velocity_y_error_std_px_per_frame"] == pytest.approx(0.0)
    assert metrics["velocity"]["velocity_ratio_error"] == pytest.approx(0.025)
    assert metrics["velocity"]["prediction_source"] == "velocities.csv"
    assert metrics["velocity"]["truth_matched_track_id"] == 0
    assert metrics["velocity"]["truth_matched_velocity_y_error_px_per_frame"] == pytest.approx(0.05)
    assert metrics["velocity"]["truth_matched_velocity_ratio_error"] == pytest.approx(0.025)
    assert metrics["filtered_velocity"]["available"] is False
    assert metrics["filtered_velocity"]["prediction_source"] == "filtered_velocities.csv"
    assert metrics["runtime"]["frames_per_second"] == pytest.approx(2.0)
    assert metrics["runtime"]["peak_rss_mb"] == pytest.approx(123.4)


def test_compute_benchmark_metrics_scores_events_from_tracks_when_available(tmp_path):
    output_dir, truth_path = make_synthetic_benchmark_case(tmp_path)
    with (output_dir / "tracks.csv").open(newline="", encoding="utf-8") as handle:
        track_rows = list(csv.DictReader(handle))
    track_rows[-1]["track_id"] = "1"
    write_csv(output_dir / "tracks.csv", track_rows, list(track_rows[0]))

    metrics = compute_benchmark_metrics(output_dir=output_dir, truth_path=truth_path)

    assert metrics["detections"]["f1"] == pytest.approx(1.0)
    assert metrics["events"]["prediction_source"] == "tracks.csv"
    assert metrics["events"]["truth_events"] == 1
    assert metrics["events"]["predicted_events"] == 2
    assert metrics["events"]["matched_events"] == 1
    assert metrics["events"]["precision"] == pytest.approx(0.5)
    assert metrics["events"]["birth_false_positive_rate"] == pytest.approx(0.5)
    assert metrics["events"]["recall"] == pytest.approx(1.0)
    assert metrics["events"]["f1"] == pytest.approx(2 / 3)


def test_compute_benchmark_metrics_falls_back_to_detection_events_without_tracks(tmp_path):
    output_dir, truth_path = make_synthetic_benchmark_case(tmp_path)
    (output_dir / "tracks.csv").unlink()

    metrics = compute_benchmark_metrics(output_dir=output_dir, truth_path=truth_path)

    assert metrics["events"]["prediction_source"] == "detections.csv"
    assert metrics["events"]["predicted_events"] == 1
    assert metrics["events"]["f1"] == pytest.approx(1.0)


def test_compute_benchmark_metrics_keeps_empty_track_events_authoritative(tmp_path):
    output_dir, truth_path = make_synthetic_benchmark_case(tmp_path)
    write_csv(
        output_dir / "tracks.csv",
        [],
        [
            "track_id",
            "track_detection_index",
            "frame_index",
            "image",
            "bbox_top",
            "bbox_left",
            "bbox_bottom",
            "bbox_right",
            "y",
            "x",
        ],
    )

    metrics = compute_benchmark_metrics(output_dir=output_dir, truth_path=truth_path)

    assert metrics["detections"]["f1"] == pytest.approx(1.0)
    assert metrics["events"]["prediction_source"] == "tracks.csv"
    assert metrics["events"]["prediction_rows"] == 0
    assert metrics["events"]["predicted_events"] == 0
    assert metrics["events"]["matched_events"] == 0
    assert metrics["events"]["recall"] == pytest.approx(0.0)
    assert metrics["events"]["f1"] is None


def test_compute_benchmark_metrics_reports_filtered_track_events(tmp_path):
    output_dir, truth_path = make_synthetic_benchmark_case(tmp_path)
    with (output_dir / "tracks.csv").open(newline="", encoding="utf-8") as handle:
        track_rows = list(csv.DictReader(handle))
    track_rows[-1]["track_id"] = "1"
    write_csv(output_dir / "tracks.csv", track_rows, list(track_rows[0]))
    write_csv(output_dir / "filtered_tracks.csv", track_rows[:-1], list(track_rows[0]))

    metrics = compute_benchmark_metrics(output_dir=output_dir, truth_path=truth_path)

    assert metrics["events"]["prediction_source"] == "tracks.csv"
    assert metrics["events"]["predicted_events"] == 2
    assert metrics["events"]["f1"] == pytest.approx(2 / 3)
    assert metrics["filtered_events"]["prediction_source"] == "filtered_tracks.csv"
    assert metrics["filtered_events"]["prediction_rows"] == 2
    assert metrics["filtered_events"]["predicted_events"] == 1
    assert metrics["filtered_events"]["f1"] == pytest.approx(1.0)
    assert metrics["filtered_tracks"]["tracks"] == 1
    assert metrics["filtered_tracks"]["median_track_length"] == pytest.approx(2.0)


def test_compute_benchmark_metrics_reports_empty_filtered_track_events(tmp_path):
    output_dir, truth_path = make_synthetic_benchmark_case(tmp_path)
    write_csv(
        output_dir / "filtered_tracks.csv",
        [],
        [
            "track_id",
            "track_detection_index",
            "frame_index",
            "image",
            "bbox_top",
            "bbox_left",
            "bbox_bottom",
            "bbox_right",
            "y",
            "x",
        ],
    )

    metrics = compute_benchmark_metrics(output_dir=output_dir, truth_path=truth_path)

    assert metrics["filtered_events"]["available"] is True
    assert metrics["filtered_events"]["prediction_rows"] == 0
    assert metrics["filtered_events"]["predicted_events"] == 0
    assert metrics["filtered_events"]["matched_events"] == 0
    assert metrics["filtered_events"]["recall"] == pytest.approx(0.0)


def test_compute_benchmark_metrics_reports_filtered_velocity_metrics(tmp_path):
    output_dir, truth_path = make_synthetic_benchmark_case(tmp_path)
    write_csv(
        output_dir / "filtered_velocities.csv",
        [
            {
                "track_id": 7,
                "n_detections": 2,
                "velocity_y_px_per_frame": 0.9,
                "velocity_ratio_y": 0.45,
            }
        ],
        ["track_id", "n_detections", "velocity_y_px_per_frame", "velocity_ratio_y"],
    )

    metrics = compute_benchmark_metrics(output_dir=output_dir, truth_path=truth_path)

    assert metrics["velocity"]["prediction_source"] == "velocities.csv"
    assert metrics["velocity"]["velocity_y_error_px_per_frame"] == pytest.approx(0.05)
    assert metrics["filtered_velocity"]["available"] is True
    assert metrics["filtered_velocity"]["prediction_source"] == "filtered_velocities.csv"
    assert metrics["filtered_velocity"]["velocity_rows"] == 1
    assert metrics["filtered_velocity"]["representative_track_id"] == 7
    assert metrics["filtered_velocity"]["velocity_y_error_px_per_frame"] == pytest.approx(-0.1)
    assert metrics["filtered_velocity"]["velocity_ratio_error"] == pytest.approx(-0.05)
    assert metrics["filtered_velocity"]["truth_matched_track_id"] == 7
    assert metrics["filtered_velocity"]["truth_matched_velocity_y_error_px_per_frame"] == pytest.approx(-0.1)
    assert metrics["filtered_velocity"]["truth_matched_velocity_ratio_error"] == pytest.approx(-0.05)


def test_compute_benchmark_metrics_reports_empty_filtered_velocity_file(tmp_path):
    output_dir, truth_path = make_synthetic_benchmark_case(tmp_path)
    write_csv(
        output_dir / "filtered_velocities.csv",
        [],
        ["track_id", "n_detections", "velocity_y_px_per_frame", "velocity_ratio_y"],
    )

    metrics = compute_benchmark_metrics(output_dir=output_dir, truth_path=truth_path)

    assert metrics["filtered_velocity"]["available"] is False
    assert metrics["filtered_velocity"]["prediction_source"] == "filtered_velocities.csv"
    assert metrics["filtered_velocity"]["reason"] == "No velocity rows found"
    assert metrics["filtered_velocity"]["velocity_rows"] == 0


def test_compute_benchmark_metrics_reports_truth_matched_velocity(tmp_path):
    output_dir, truth_path = make_synthetic_benchmark_case(tmp_path)
    velocity_rows = [
        {
            "track_id": 10,
            "n_detections": 100,
            "velocity_y_px_per_frame": 0.1,
            "velocity_ratio_y": 0.05,
        },
        {
            "track_id": 11,
            "n_detections": 2,
            "velocity_y_px_per_frame": 1.01,
            "velocity_ratio_y": 0.505,
        },
    ]
    write_csv(
        output_dir / "velocities.csv",
        velocity_rows,
        ["track_id", "n_detections", "velocity_y_px_per_frame", "velocity_ratio_y"],
    )
    write_csv(
        output_dir / "filtered_velocities.csv",
        velocity_rows,
        ["track_id", "n_detections", "velocity_y_px_per_frame", "velocity_ratio_y"],
    )

    metrics = compute_benchmark_metrics(output_dir=output_dir, truth_path=truth_path)

    assert metrics["velocity"]["representative_track_id"] == 10
    assert metrics["velocity"]["velocity_y_error_px_per_frame"] == pytest.approx(-0.9)
    assert metrics["velocity"]["velocity_y_mean_abs_error_px_per_frame"] == pytest.approx((0.9 + 0.01) / 2)
    assert metrics["velocity"]["velocity_y_bias_px_per_frame"] == pytest.approx((-0.9 + 0.01) / 2)
    assert metrics["velocity"]["velocity_y_error_std_px_per_frame"] == pytest.approx(np.std([-0.9, 0.01]))
    assert metrics["velocity"]["truth_matched_track_id"] == 11
    assert metrics["velocity"]["truth_matched_velocity_y_error_px_per_frame"] == pytest.approx(0.01)
    assert metrics["velocity"]["truth_matched_velocity_ratio_error"] == pytest.approx(0.005)
    assert metrics["filtered_velocity"]["representative_track_id"] == 10
    assert metrics["filtered_velocity"]["truth_matched_track_id"] == 11


def test_event_metrics_distinguishes_frame_coverage_from_event_recall():
    truth = {
        "particles": [
            {"event_id": "a", "frame_index": 0, "top": 1, "left": 1, "bottom": 3, "right": 3},
            {"event_id": "a", "frame_index": 1, "top": 2, "left": 1, "bottom": 4, "right": 3},
            {"event_id": "a", "frame_index": 2, "top": 3, "left": 1, "bottom": 5, "right": 3},
            {"event_id": "b", "frame_index": 0, "top": 8, "left": 8, "bottom": 10, "right": 10},
        ]
    }
    detections = [
        {
            "track_id": "det-a",
            "frame_index": "1",
            "bbox_top": "2",
            "bbox_left": "1",
            "bbox_bottom": "4",
            "bbox_right": "3",
            "y": "2.5",
            "x": "1.5",
        },
        {
            "track_id": "false-positive",
            "frame_index": "2",
            "bbox_top": "20",
            "bbox_left": "20",
            "bbox_bottom": "22",
            "bbox_right": "22",
            "y": "20.5",
            "x": "20.5",
        },
    ]

    metrics = event_metrics(detections, truth, iou_threshold=0.25)

    assert metrics["truth_events"] == 2
    assert metrics["predicted_events"] == 2
    assert metrics["matched_events"] == 1
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["mean_truth_frame_coverage"] == pytest.approx(1 / 3)
    assert metrics["mean_latency_frames"] == pytest.approx(1.0)


def test_event_metrics_reports_track_fragmentation():
    truth = {
        "particles": [
            {"event_id": "a", "frame_index": 0, "top": 1, "left": 1, "bottom": 3, "right": 3},
            {"event_id": "a", "frame_index": 1, "top": 2, "left": 1, "bottom": 4, "right": 3},
        ]
    }
    detections = [
        {
            "track_id": "fragment-0",
            "frame_index": "0",
            "bbox_top": "1",
            "bbox_left": "1",
            "bbox_bottom": "3",
            "bbox_right": "3",
            "y": "1.5",
            "x": "1.5",
        },
        {
            "track_id": "fragment-1",
            "frame_index": "1",
            "bbox_top": "2",
            "bbox_left": "1",
            "bbox_bottom": "4",
            "bbox_right": "3",
            "y": "2.5",
            "x": "1.5",
        },
    ]

    metrics = event_metrics(detections, truth, iou_threshold=0.25)

    assert metrics["truth_events"] == 1
    assert metrics["predicted_events"] == 2
    assert metrics["fragmented_truth_events"] == 1
    assert metrics["extra_fragment_events"] == 1
    assert metrics["mean_fragments_per_truth_event"] == pytest.approx(2.0)
    assert metrics["track_fragmentation"] == pytest.approx(1.0)


def test_track_metrics_reports_single_frame_tracks():
    rows = [
        {"track_id": "a", "frame_index": "0"},
        {"track_id": "a", "frame_index": "1"},
        {"track_id": "b", "frame_index": "4"},
    ]

    metrics = track_metrics(rows)

    assert metrics["tracks"] == 2
    assert metrics["track_rows"] == 3
    assert metrics["mean_track_length"] == pytest.approx(1.5)
    assert metrics["median_track_length"] == pytest.approx(1.5)
    assert metrics["single_frame_tracks"] == 1
    assert metrics["single_frame_track_fraction"] == pytest.approx(0.5)
    assert metrics["mean_track_span_frames"] == pytest.approx(1.5)


def test_generate_benchmark_report_writes_json_and_markdown(tmp_path):
    output_dir, truth_path = make_synthetic_benchmark_case(tmp_path)

    artifacts = generate_benchmark_report(output_dir=output_dir, truth_path=truth_path)

    assert artifacts.metrics.is_file()
    assert artifacts.report.is_file()
    loaded = json.loads(artifacts.metrics.read_text(encoding="utf-8"))
    assert loaded["benchmark"]["type"] == "synthetic_ground_truth"
    assert loaded["events"]["truth_events"] == 1
    report = artifacts.report.read_text(encoding="utf-8")
    assert "phase RMSE" in report
    assert "event F1" in report
