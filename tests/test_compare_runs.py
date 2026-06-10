import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from beltmap.cli import compare as cli_compare
from beltmap.compare_runs import (
    RunSpec,
    detection_froc_curve,
    finite_int,
    generate_comparison_report,
    load_labeled_detection_truth,
    parse_run_spec,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_run(output_dir: Path, *, threshold: float, count_offset: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "n_images": 3,
                "detection_threshold": threshold,
                "n_detections": 3 + count_offset,
                "n_tracks": 2,
                "n_velocity_estimates": 2,
                "elapsed_s": 12.5,
            }
        ),
        encoding="utf-8",
    )
    write_csv(
        output_dir / "detections_per_frame.csv",
        [
            {"frame_index": 0, "n_detections": 1 + count_offset},
            {"frame_index": 1, "n_detections": 2 + count_offset},
            {"frame_index": 2, "n_detections": count_offset},
        ],
    )
    write_csv(
        output_dir / "detections.csv",
        [
            {
                "frame_index": 0,
                "label": 1,
                "y": 2.0,
                "x": 3.0,
                "area_px": 6 + count_offset,
                "peak_signal": 4.2 + count_offset,
                "bbox_top": 1,
                "bbox_left": 2,
                "bbox_bottom": 4,
                "bbox_right": 5,
            },
            {
                "frame_index": 1,
                "label": 1,
                "y": 4.0,
                "x": 3.0,
                "area_px": 12 + count_offset,
                "peak_signal": 4.8 + count_offset,
                "bbox_top": 3,
                "bbox_left": 2,
                "bbox_bottom": 6,
                "bbox_right": 5,
            },
        ],
    )
    write_csv(
        output_dir / "velocities.csv",
        [
            {"track_id": 0, "n_detections": 5, "velocity_ratio_y": 0.5},
            {"track_id": 1, "n_detections": 10, "velocity_ratio_y": 1.4},
        ],
    )
    write_csv(
        output_dir / "filtered_velocities.csv",
        [
            {"track_id": 0, "n_detections": 5, "velocity_ratio_y": 0.5},
        ],
    )
    write_csv(
        output_dir / "filtered_tracks.csv",
        [
            {
                "track_id": 0,
                "track_detection_index": 0,
                "frame_index": 0,
                "label": 1,
                "y": 2.0,
                "x": 3.0,
                "area_px": 6 + count_offset,
                "bbox_top": 1,
                "bbox_left": 2,
                "bbox_bottom": 4,
                "bbox_right": 5,
            },
        ],
    )
    for frame_index in [0, 2]:
        Image.new("L", (8, 8), 32 + 40 * frame_index).save(
            output_dir / f"residual_frame_{frame_index:06d}.png"
        )


def test_parse_run_spec_supports_label_and_default_label():
    assert parse_run_spec("T3p5=outputs/T3p5") == RunSpec("T3p5", Path("outputs/T3p5"))
    assert parse_run_spec("outputs/T4p0") == RunSpec("T4p0", Path("outputs/T4p0"))


def test_finite_int_rejects_fractional_values():
    assert finite_int("7") == 7
    assert finite_int("7.0") == 7
    assert finite_int("7.5") is None


def test_generate_comparison_report_writes_summary_plots_and_contact_sheet(tmp_path):
    run_a = tmp_path / "T4p0"
    run_b = tmp_path / "T3p5"
    make_run(run_a, threshold=4.0, count_offset=0)
    make_run(run_b, threshold=3.5, count_offset=2)

    artifacts = generate_comparison_report(
        [RunSpec("T4.0", run_a), RunSpec("T3.5", run_b)],
        report_dir=tmp_path / "comparison",
        frames=[0, 2],
    )

    assert artifacts.report.is_file()
    assert artifacts.summary_csv.is_file()
    assert set(artifacts.plots) == {"detections_per_frame", "velocity_ratio_histogram"}
    assert set(artifacts.images) == {
        "detection_contact_sheet",
        "filtered_detection_contact_sheet",
    }
    for path in [*artifacts.plots.values(), *artifacts.images.values()]:
        assert path.is_file()
        assert path.stat().st_size > 0
    report = artifacts.report.read_text(encoding="utf-8")
    assert "# BeltMap run comparison" in report
    assert "T4.0" in report
    assert "T3.5" in report
    assert "Detection contact sheet" in report
    assert "Filtered detection contact sheet" in report
    rows = list(csv.DictReader(artifacts.summary_csv.open(newline="", encoding="utf-8")))
    assert rows[0]["label"] == "T4.0"
    assert rows[0]["detection_threshold"] == "4.0"
    assert rows[0]["small_detection_share_area_lt_50"] == "1.0"
    assert rows[0]["small_accepted_tracks_lt_50"] == "1"
    assert rows[0]["near_threshold_peak_share"] == "1.0"
    assert "small accepted tracks" in report


def test_generate_comparison_report_preserves_zero_image_metadata(tmp_path):
    run_a = tmp_path / "T4p0"
    run_b = tmp_path / "T3p5"
    make_run(run_a, threshold=4.0, count_offset=0)
    make_run(run_b, threshold=3.5, count_offset=2)
    metadata_path = run_a / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["n_images"] = 0
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    artifacts = generate_comparison_report(
        [RunSpec("T4.0", run_a), RunSpec("T3.5", run_b)],
        report_dir=tmp_path / "comparison",
    )

    rows = list(csv.DictReader(artifacts.summary_csv.open(newline="", encoding="utf-8")))
    assert rows[0]["n_images"] == "0"


def test_generate_comparison_report_includes_fixed_and_raw_preview_sheets(tmp_path):
    run_a = tmp_path / "T4p0"
    run_b = tmp_path / "T3p5"
    make_run(run_a, threshold=4.0, count_offset=0)
    make_run(run_b, threshold=3.5, count_offset=2)
    for output_dir in [run_a, run_b]:
        for frame_index in [0, 2]:
            Image.new("L", (8, 8), 64 + 10 * frame_index).save(
                output_dir / f"raw_frame_{frame_index:06d}.png"
            )
            Image.new("L", (8, 8), 96 + 10 * frame_index).save(
                output_dir / f"residual_fixed_frame_{frame_index:06d}.png"
            )

    artifacts = generate_comparison_report(
        [RunSpec("T4.0", run_a), RunSpec("T3.5", run_b)],
        report_dir=tmp_path / "comparison",
        frames=[0, 2],
    )

    assert "raw_detection_contact_sheet" in artifacts.images
    assert "fixed_scale_detection_contact_sheet" in artifacts.images
    assert artifacts.images["raw_detection_contact_sheet"].is_file()
    assert artifacts.images["fixed_scale_detection_contact_sheet"].is_file()
    report = artifacts.report.read_text(encoding="utf-8")
    assert "Raw crops use one shared display scale" in report
    assert "Fixed residual previews use a fixed normalized-residual display range" in report


def test_generate_comparison_report_can_skip_contact_sheets(tmp_path):
    run_a = tmp_path / "T4p0"
    run_b = tmp_path / "T3p5"
    make_run(run_a, threshold=4.0, count_offset=0)
    make_run(run_b, threshold=3.5, count_offset=2)

    artifacts = generate_comparison_report(
        [RunSpec("T4.0", run_a), RunSpec("T3.5", run_b)],
        report_dir=tmp_path / "comparison",
        frames=[0, 2],
        make_contact_sheets=False,
    )

    assert set(artifacts.plots) == {"detections_per_frame", "velocity_ratio_histogram"}
    assert artifacts.images == {}
    for path in artifacts.plots.values():
        assert path.is_file()
    assert not list((tmp_path / "comparison").glob("*contact_sheet.png"))
    report = artifacts.report.read_text(encoding="utf-8")
    assert "## Detection counts" in report
    assert "Detection contact sheet" not in report


def test_generate_comparison_report_scores_labeled_detection_target(tmp_path):
    run_a = tmp_path / "T4p0"
    run_b = tmp_path / "T3p5"
    make_run(run_a, threshold=4.0, count_offset=0)
    make_run(run_b, threshold=3.5, count_offset=1)
    with (run_b / "detections.csv").open(newline="", encoding="utf-8") as handle:
        run_b_detections = list(csv.DictReader(handle))
    run_b_detections.append(
        {
            "frame_index": 2,
            "label": 2,
            "y": 7.0,
            "x": 7.0,
            "area_px": 9,
            "peak_signal": 4.0,
            "bbox_top": 6,
            "bbox_left": 6,
            "bbox_bottom": 8,
            "bbox_right": 8,
        }
    )
    write_csv(run_b / "detections.csv", run_b_detections)
    truth_path = tmp_path / "labels.csv"
    write_csv(
        truth_path,
        [
            {
                "frame_index": 0,
                "bbox_top": 1,
                "bbox_left": 2,
                "bbox_bottom": 4,
                "bbox_right": 5,
            },
            {"frame_index": 2, "bbox_top": "", "bbox_left": "", "bbox_bottom": "", "bbox_right": ""},
        ],
    )

    artifacts = generate_comparison_report(
        [RunSpec("T4.0", run_a), RunSpec("T3.5", run_b)],
        report_dir=tmp_path / "comparison",
        frames=[0, 2],
        truth_path=truth_path,
        truth_iou_threshold=0.25,
    )

    report = artifacts.report.read_text(encoding="utf-8")
    assert "labeled_detection_froc" in artifacts.plots
    assert artifacts.plots["labeled_detection_froc"].is_file()
    assert "## Labeled real-data target" in report
    assert "### Labeled detection FROC" in report
    assert "labeled F1" in report
    assert "recall @0.1 FP/frame" in report
    rows = list(csv.DictReader(artifacts.summary_csv.open(newline="", encoding="utf-8")))
    assert rows[0]["labeled_detection_available"] == "True"
    assert rows[0]["labeled_scored_frames"] == "2"
    assert rows[0]["labeled_truth_boxes"] == "1"
    assert rows[0]["labeled_predicted_boxes"] == "1"
    assert rows[0]["labeled_true_positives"] == "1"
    assert rows[0]["labeled_false_positives"] == "0"
    assert rows[0]["labeled_false_negatives"] == "0"
    assert rows[0]["labeled_false_positives_per_frame"] == "0.0"
    assert rows[0]["labeled_empty_scored_frames"] == "1"
    assert rows[0]["labeled_empty_frame_false_positives"] == "0"
    assert rows[0]["labeled_empty_frame_fp_per_frame"] == "0.0"
    assert rows[0]["labeled_f1"] == "1.0"
    assert rows[1]["labeled_froc_score_field"] == "peak_signal"
    assert rows[1]["labeled_froc_points"] == "3"
    assert float(rows[1]["labeled_froc_recall_at_0_1_fp_per_frame"]) == pytest.approx(1.0)
    assert float(rows[1]["labeled_froc_recall_at_0_5_fp_per_frame"]) == pytest.approx(1.0)


def test_labeled_truth_rejects_unreviewed_json_scaffold(tmp_path):
    truth_path = tmp_path / "labels.json"
    truth_path.write_text(
        json.dumps(
            {
                "status": "template_not_ground_truth_do_not_use_for_metrics_until_filled",
                "requires_manual_review": True,
                "scored_frames": [0],
                "particles": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reviewed_ground_truth"):
        load_labeled_detection_truth(truth_path)


def test_generate_comparison_report_scores_reviewed_frame_box_json(tmp_path):
    run_a = tmp_path / "T4p0"
    run_b = tmp_path / "T3p5"
    make_run(run_a, threshold=4.0, count_offset=0)
    make_run(run_b, threshold=3.5, count_offset=1)
    with (run_b / "detections.csv").open(newline="", encoding="utf-8") as handle:
        run_b_detections = list(csv.DictReader(handle))
    run_b_detections.append(
        {
            "frame_index": 2,
            "label": 2,
            "y": 7.0,
            "x": 7.0,
            "area_px": 9,
            "peak_signal": 4.0,
            "bbox_top": 6,
            "bbox_left": 6,
            "bbox_bottom": 8,
            "bbox_right": 8,
        }
    )
    write_csv(run_b / "detections.csv", run_b_detections)
    truth_path = tmp_path / "labels.json"
    truth_path.write_text(
        json.dumps(
            {
                "status": "reviewed_ground_truth",
                "requires_manual_review": False,
                "frames": [
                    {
                        "frame_index": 0,
                        "boxes": [
                            {
                                "top": 1,
                                "left": 2,
                                "bottom": 4,
                                "right": 5,
                                "particle_id": "p0",
                            }
                        ],
                    },
                    {"frame_index": 2, "boxes": []},
                ],
            }
        ),
        encoding="utf-8",
    )

    truth = load_labeled_detection_truth(truth_path)
    assert truth["scored_frames"] == [0, 2]
    assert truth["particles"] == [
        {
            "frame_index": 0,
            "top": 1.0,
            "left": 2.0,
            "bottom": 4.0,
            "right": 5.0,
            "particle_id": "p0",
        }
    ]

    artifacts = generate_comparison_report(
        [RunSpec("T4.0", run_a), RunSpec("T3.5", run_b)],
        report_dir=tmp_path / "comparison",
        frames=[0, 2],
        truth_path=truth_path,
        truth_iou_threshold=0.25,
    )
    rows = list(csv.DictReader(artifacts.summary_csv.open(newline="", encoding="utf-8")))
    assert rows[0]["labeled_truth_boxes"] == "1"
    assert rows[0]["labeled_empty_scored_frames"] == "1"
    assert rows[1]["labeled_false_positives"] == "1"
    assert rows[1]["labeled_empty_frame_false_positives"] == "1"


def test_generate_comparison_report_scores_all_empty_reviewed_frames(tmp_path):
    run_a = tmp_path / "T4p0"
    run_b = tmp_path / "T3p5"
    make_run(run_a, threshold=4.0, count_offset=0)
    make_run(run_b, threshold=3.5, count_offset=0)
    truth_path = tmp_path / "labels.csv"
    write_csv(
        truth_path,
        [
            {"frame_index": 2, "bbox_top": "", "bbox_left": "", "bbox_bottom": "", "bbox_right": ""},
        ],
    )

    artifacts = generate_comparison_report(
        [RunSpec("T4.0", run_a), RunSpec("T3.5", run_b)],
        report_dir=tmp_path / "comparison",
        frames=[2],
        truth_path=truth_path,
        truth_iou_threshold=0.25,
        bootstrap_samples=10,
        bootstrap_seed=3,
    )

    rows = list(csv.DictReader(artifacts.summary_csv.open(newline="", encoding="utf-8")))
    assert rows[0]["labeled_detection_available"] == "True"
    assert rows[0]["labeled_scored_frames"] == "1"
    assert rows[0]["labeled_truth_boxes"] == "0"
    assert rows[0]["labeled_predicted_boxes"] == "0"
    assert rows[0]["labeled_false_positives_per_frame"] == "0.0"
    assert rows[0]["labeled_empty_scored_frames"] == "1"
    assert rows[0]["labeled_empty_frame_false_positives"] == "0"
    assert rows[0]["labeled_precision"] == "1.0"
    assert rows[0]["labeled_recall"] == "1.0"
    assert rows[0]["labeled_f1"] == "1.0"
    assert rows[0]["labeled_f1_bootstrap_median"] == "1.0"
    assert rows[0]["labeled_f1_ci_low"] == "1.0"
    assert rows[0]["labeled_f1_ci_high"] == "1.0"


def test_detection_froc_curve_sweeps_peak_signal_with_empty_scored_frames():
    truth = {
        "particles": [
            {
                "frame_index": 0,
                "top": 1,
                "left": 2,
                "bottom": 4,
                "right": 5,
            }
        ]
    }
    detections = [
        {
            "frame_index": "0",
            "bbox_top": "1",
            "bbox_left": "2",
            "bbox_bottom": "4",
            "bbox_right": "5",
            "y": "2.0",
            "x": "3.0",
            "peak_signal": "12.0",
        },
        {
            "frame_index": "2",
            "bbox_top": "20",
            "bbox_left": "20",
            "bbox_bottom": "22",
            "bbox_right": "22",
            "y": "21.0",
            "x": "21.0",
            "peak_signal": "6.0",
        },
    ]

    froc = detection_froc_curve(
        detections,
        truth,
        scored_frames={0, 2},
        iou_threshold=0.25,
    )

    assert froc["available"] is True
    assert froc["score_field"] == "peak_signal"
    assert froc["point_count"] == 3
    assert froc["recall_at_0_1_fp_per_frame"] == pytest.approx(1.0)
    assert froc["recall_at_0_5_fp_per_frame"] == pytest.approx(1.0)
    assert froc["auc_fp_per_frame_le_1"] == pytest.approx(1.0)


def test_detection_froc_curve_restricts_truth_to_scored_frames():
    truth = {
        "particles": [
            {
                "frame_index": 0,
                "top": 1,
                "left": 2,
                "bottom": 4,
                "right": 5,
            },
            {
                "frame_index": 99,
                "top": 1,
                "left": 2,
                "bottom": 4,
                "right": 5,
            },
        ]
    }
    detections = [
        {
            "frame_index": "0",
            "bbox_top": "1",
            "bbox_left": "2",
            "bbox_bottom": "4",
            "bbox_right": "5",
            "y": "2.0",
            "x": "3.0",
            "peak_signal": "12.0",
        }
    ]

    froc = detection_froc_curve(
        detections,
        truth,
        scored_frames={0},
        iou_threshold=0.25,
    )

    assert froc["truth_boxes"] == 1
    assert froc["recall_at_0_1_fp_per_frame"] == pytest.approx(1.0)
    assert froc["auc_fp_per_frame_le_1"] == pytest.approx(1.0)


def test_detection_froc_curve_rejects_partial_score_fields():
    truth = {
        "particles": [
            {
                "frame_index": 0,
                "top": 1,
                "left": 2,
                "bottom": 4,
                "right": 5,
            }
        ]
    }
    detections = [
        {
            "frame_index": "0",
            "bbox_top": "1",
            "bbox_left": "2",
            "bbox_bottom": "4",
            "bbox_right": "5",
            "y": "2.0",
            "x": "3.0",
            "peak_signal": "12.0",
        },
        {
            "frame_index": "2",
            "bbox_top": "20",
            "bbox_left": "20",
            "bbox_bottom": "22",
            "bbox_right": "22",
            "y": "21.0",
            "x": "21.0",
            "peak_signal": "",
        },
    ]

    froc = detection_froc_curve(
        detections,
        truth,
        scored_frames={0, 2},
        iou_threshold=0.25,
    )

    assert froc["available"] is False
    assert froc["score_field"] is None
    assert froc["skipped_score_rows"] == 1
    assert froc["auc_fp_per_frame_le_1"] is None
    assert froc["recall_at_0_1_fp_per_frame"] is None
    assert "partial-score FROC" in froc["reason"]


def test_generate_comparison_report_adds_bootstrap_confidence_intervals(tmp_path):
    run_a = tmp_path / "T4p0"
    run_b = tmp_path / "T3p5"
    make_run(run_a, threshold=4.0, count_offset=0)
    make_run(run_b, threshold=3.5, count_offset=1)
    truth_path = tmp_path / "labels.csv"
    write_csv(
        truth_path,
        [
            {
                "frame_index": 0,
                "bbox_top": 1,
                "bbox_left": 2,
                "bbox_bottom": 4,
                "bbox_right": 5,
            },
            {"frame_index": 2, "bbox_top": "", "bbox_left": "", "bbox_bottom": "", "bbox_right": ""},
        ],
    )

    artifacts = generate_comparison_report(
        [RunSpec("T4.0", run_a), RunSpec("T3.5", run_b)],
        report_dir=tmp_path / "comparison",
        frames=[0, 2],
        truth_path=truth_path,
        truth_iou_threshold=0.25,
        bootstrap_samples=40,
        bootstrap_seed=7,
        bootstrap_block_length_frames=2,
    )

    report = artifacts.report.read_text(encoding="utf-8")
    assert "## Bootstrap confidence intervals" in report
    assert "bootstrap median [low, high]" in report
    rows = list(csv.DictReader(artifacts.summary_csv.open(newline="", encoding="utf-8")))
    assert rows[0]["bootstrap_samples"] == "40"
    assert rows[0]["bootstrap_confidence_level"] == "0.95"
    assert rows[0]["bootstrap_block_length_frames"] == "2"
    assert rows[0]["detections_per_frame_mean_bootstrap_median"] != ""
    assert rows[0]["detections_per_frame_mean_ci_low"] != ""
    assert rows[0]["detections_per_frame_mean_ci_high"] != ""
    assert rows[0]["labeled_f1_bootstrap_median"] != ""
    assert rows[0]["labeled_f1_ci_low"] != ""
    assert rows[0]["long_velocity_tracks_ge_10_ci_high"] != ""


def test_compare_main_prints_artifact_paths(tmp_path, capsys):
    run_a = tmp_path / "T4p0"
    run_b = tmp_path / "T3p5"
    make_run(run_a, threshold=4.0, count_offset=0)
    make_run(run_b, threshold=3.5, count_offset=1)

    exit_code = cli_compare.main(
        [
            "--run",
            f"T4.0={run_a}",
            "--run",
            f"T3.5={run_b}",
            "--report-dir",
            str(tmp_path / "comparison"),
            "--frames",
            "0,2",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert Path(payload["report"]).is_file()
    assert Path(payload["summary_csv"]).is_file()


def test_compare_main_metrics_only_skips_png_outputs(tmp_path, capsys):
    run_a = tmp_path / "T4p0"
    run_b = tmp_path / "T3p5"
    make_run(run_a, threshold=4.0, count_offset=0)
    make_run(run_b, threshold=3.5, count_offset=1)

    exit_code = cli_compare.main(
        [
            "--run",
            f"T4.0={run_a}",
            "--run",
            f"T3.5={run_b}",
            "--report-dir",
            str(tmp_path / "comparison"),
            "--metrics-only",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert Path(payload["report"]).is_file()
    assert Path(payload["summary_csv"]).is_file()
    assert payload["plots"] == {}
    assert payload["images"] == {}
    assert payload["make_metric_plots"] is False
    assert payload["make_contact_sheets"] is False
    assert not list((tmp_path / "comparison").glob("*.png"))
    report = Path(payload["report"]).read_text(encoding="utf-8")
    assert "# BeltMap run comparison" in report
    assert "![" not in report


@pytest.mark.parametrize("value", ["nan", "inf", "-0.1", "1.1"])
def test_compare_cli_rejects_invalid_truth_iou_threshold(value):
    with pytest.raises(SystemExit) as exc_info:
        cli_compare.build_parser().parse_args(
            [
                "--run",
                "a=outputs/a",
                "--run",
                "b=outputs/b",
                "--truth-iou-threshold",
                value,
            ]
        )

    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--bootstrap-samples", "-1"),
        ("--bootstrap-confidence-level", "1.0"),
        ("--bootstrap-confidence-level", "nan"),
        ("--bootstrap-block-length-frames", "0"),
    ],
)
def test_compare_cli_rejects_invalid_bootstrap_options(option, value):
    with pytest.raises(SystemExit) as exc_info:
        cli_compare.build_parser().parse_args(
            [
                "--run",
                "a=outputs/a",
                "--run",
                "b=outputs/b",
                option,
                value,
            ]
        )

    assert exc_info.value.code == 2


def test_compare_rejects_existing_csv_with_missing_required_columns(tmp_path):
    run_a = tmp_path / "T4p0"
    run_b = tmp_path / "T3p5"
    make_run(run_a, threshold=4.0, count_offset=0)
    make_run(run_b, threshold=3.5, count_offset=1)
    write_csv(run_b / "detections.csv", [{"frame_index": 0, "area_px": 4}])

    try:
        generate_comparison_report([RunSpec("a", run_a), RunSpec("b", run_b)], report_dir=tmp_path / "comparison")
    except ValueError as exc:
        assert "bbox_bottom" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected schema validation to reject malformed CSV")


def test_compare_rejects_fractional_metadata_counts(tmp_path):
    run_a = tmp_path / "T4p0"
    run_b = tmp_path / "T3p5"
    make_run(run_a, threshold=4.0, count_offset=0)
    make_run(run_b, threshold=3.5, count_offset=1)
    metadata = json.loads((run_b / "metadata.json").read_text(encoding="utf-8"))
    metadata["n_detections"] = 2.5
    (run_b / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="n_detections"):
        generate_comparison_report(
            [RunSpec("a", run_a), RunSpec("b", run_b)],
            report_dir=tmp_path / "comparison",
        )


def test_compare_rejects_fractional_track_metadata_count(tmp_path):
    run_a = tmp_path / "T4p0"
    run_b = tmp_path / "T3p5"
    make_run(run_a, threshold=4.0, count_offset=0)
    make_run(run_b, threshold=3.5, count_offset=1)
    metadata = json.loads((run_b / "metadata.json").read_text(encoding="utf-8"))
    metadata["n_tracks"] = 2.5
    (run_b / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="n_tracks"):
        generate_comparison_report(
            [RunSpec("a", run_a), RunSpec("b", run_b)],
            report_dir=tmp_path / "comparison",
        )
