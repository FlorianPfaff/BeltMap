import csv
import json

import numpy as np

from beltmap import postrun_improvements as pri


def test_compute_phase_row_counts_wraps_periodically():
    counts = pri.compute_phase_row_counts([0.0, 3.0], map_height=5, crop_height=3)
    assert counts.dtype == np.uint64
    assert counts.tolist() == [2, 1, 1, 1, 1]


def test_uncertainty_from_counts_marks_unobserved_rows():
    uncertainty = pri.uncertainty_from_counts(np.array([0, 1, 4]), scale=2.0)
    assert uncertainty[0] == 2.0
    assert uncertainty[1] == 2.0
    assert uncertainty[2] == 1.0


def test_finite_int_rejects_fractional_values():
    assert pri.finite_int("7") == 7
    assert pri.finite_int("7.0") == 7
    assert pri.finite_int("7.5") is None


def test_detection_count_by_frame_ignores_fractional_frame_indices(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "detections_per_frame.csv").write_text(
        "frame_index,n_detections\n0.5,7\n1,2\n",
        encoding="utf-8",
    )

    assert pri.detection_count_by_frame(out) == {1: 2}


def test_map_uncertainty_creates_explicit_report_dir(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    report = tmp_path / "reports" / "map_uncertainty"
    (out / "metadata.json").write_text(
        json.dumps({"belt_map_height_px": 5, "belt_region": {"height": 3, "width": 2}}),
        encoding="utf-8",
    )
    (out / "phase_estimates.csv").write_text("frame_index,phase_px\n0,0\n1,3\n", encoding="utf-8")

    summary = pri.write_map_uncertainty_outputs(out, report_dir=report)

    assert summary["available"]
    assert (report / "belt_map_row_counts.npy").is_file()


def test_quality_contract_uses_standard_csvs(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "metadata.json").write_text(json.dumps({"registration_search_radius_px": 8.0, "n_detections": 2}), encoding="utf-8")
    (out / "phase_estimates.csv").write_text("frame_index,correction_px,score\n0,0.5,0.8\n1,1.0,0.9\n", encoding="utf-8")
    (out / "detections.csv").write_text("frame_index,area_px\n0,10\n1,12\n", encoding="utf-8")
    (out / "velocities.csv").write_text("track_id,velocity_ratio_y\n0,0.5\n", encoding="utf-8")
    (out / "filtered_velocities.csv").write_text("track_id,velocity_ratio_y\n0,0.5\n", encoding="utf-8")

    results = pri.evaluate_quality_contract(out)

    assert results
    assert all(result.passed for result in results)


def test_quality_contract_uses_metadata_registration_search_radius(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "metadata.json").write_text(
        json.dumps({"registration_search_radius_px": 4.0, "registration_search_step_px": 0.5}),
        encoding="utf-8",
    )
    (out / "phase_estimates.csv").write_text("frame_index,correction_px,score\n0,4.0,0.8\n", encoding="utf-8")
    (out / "detections.csv").write_text("frame_index,area_px\n0,10\n", encoding="utf-8")
    (out / "velocities.csv").write_text("track_id,velocity_ratio_y\n0,0.5\n", encoding="utf-8")
    (out / "filtered_velocities.csv").write_text("track_id,velocity_ratio_y\n0,0.5\n", encoding="utf-8")

    results = pri.evaluate_quality_contract(out, {"max_registration_boundary_share": 0.05})

    assert len(results) == 1
    assert not results[0].passed
    assert results[0].value == 1.0


def test_postrun_quality_flags_preserve_zero_detection_metadata(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "metadata.json").write_text(
        json.dumps({"n_recurrent_artifact_rejected": 5, "n_detections": 0}),
        encoding="utf-8",
    )
    (out / "detections.csv").write_text("frame_index,area_px\n0,10\n1,12\n", encoding="utf-8")

    flags = pri.quality_flags_from_outputs(out)

    flag = next(flag for flag in flags if flag.code == "heavy_recurrent_filtering")
    assert flag.value == 1.0


def test_quality_contract_preserves_zero_detection_metadata(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "metadata.json").write_text(
        json.dumps({"n_recurrent_artifact_rejected": 5, "n_detections": 0}),
        encoding="utf-8",
    )
    (out / "detections.csv").write_text("frame_index,area_px\n0,10\n1,12\n", encoding="utf-8")

    results = pri.evaluate_quality_contract(out, {"max_recurrent_rejection_share": 0.75})

    assert len(results) == 1
    assert not results[0].passed
    assert results[0].value == 1.0


def test_label_plan_combines_failure_buckets(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "detections_per_frame.csv").write_text("frame_index,n_detections\n0,0\n1,10\n2,5\n", encoding="utf-8")
    (out / "phase_estimates.csv").write_text("frame_index,correction_px,score\n0,0,0.9\n1,8,0.1\n2,1,0.8\n", encoding="utf-8")

    rows = pri.suggest_label_frames(out, frame_count=3)

    assert len(rows) == 3
    assert {row["frame_index"] for row in rows}.issubset({0, 1, 2})
    assert all("annotation_role" in row for row in rows)
    assert any(row["annotation_role"] == "empty_check" for row in rows)


def test_write_label_plan_can_write_box_template(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "metadata.json").write_text(json.dumps({"n_images": 3}), encoding="utf-8")
    (out / "detections_per_frame.csv").write_text("frame_index,n_detections\n0,0\n1,12\n2,3\n", encoding="utf-8")
    (out / "phase_estimates.csv").write_text("frame_index,correction_px,score\n0,0,0.9\n1,8,0.1\n2,1,0.8\n", encoding="utf-8")

    plan_path = tmp_path / "label_plan.csv"
    template_path = tmp_path / "validation_boxes.csv"
    plan_rows = pri.write_label_plan(out, output_path=plan_path, frame_count=3, empty_frame_count=1)
    template_rows = pri.write_label_template(plan_rows, output_path=template_path)

    assert plan_path.is_file()
    assert template_path.is_file()
    assert len(template_rows) == len(plan_rows)
    loaded = list(csv.DictReader(template_path.open(newline="", encoding="utf-8")))
    assert loaded
    assert {"frame_index", "bbox_top", "bbox_left", "bbox_bottom", "bbox_right"}.issubset(loaded[0])
    assert any(row["annotation_role"] == "empty_check" for row in loaded)


def test_color_residual_score_returns_spatial_score():
    observed = np.zeros((2, 3, 3), dtype=float)
    expected = np.zeros((2, 3, 3), dtype=float)
    observed[0, 0, 0] = 5.0

    score = pri.color_residual_score(observed, expected)

    assert score.shape == (2, 3)
    assert score[0, 0] > score[1, 1]


def test_belt_edge_ignore_mask_excludes_edges():
    mask = pri.belt_edge_ignore_mask((5, 6), margin_px=1)
    assert not mask[0, 0]
    assert mask[2, 2]


def test_seam_discontinuity_window_changes_current_seam_score():
    belt = np.array([[0.0], [100.0], [0.0], [0.0]])

    narrow = pri.seam_discontinuity_profile(belt, window_px=1)
    wider = pri.seam_discontinuity_profile(belt, window_px=2)

    assert narrow["current_mean_abs_jump_gray"] != wider["current_mean_abs_jump_gray"]


def test_warp_perspective_identity():
    image = np.arange(9, dtype=float).reshape(3, 3)
    warped = pri.warp_perspective(image, np.eye(3), (3, 3))
    np.testing.assert_allclose(warped, image)
