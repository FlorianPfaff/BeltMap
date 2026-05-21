import json

import numpy as np

from beltmap import postrun_improvements as pri


def test_compute_phase_row_counts_wraps_periodically():
    counts = pri.compute_phase_row_counts([0.0, 3.0], map_height=5, crop_height=3)
    assert counts.tolist() == [2, 1, 1, 1, 1]


def test_uncertainty_from_counts_marks_unobserved_rows():
    uncertainty = pri.uncertainty_from_counts(np.array([0, 1, 4]), scale=2.0)
    assert uncertainty[0] == 2.0
    assert uncertainty[1] == 2.0
    assert uncertainty[2] == 1.0


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


def test_label_plan_combines_failure_buckets(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "detections_per_frame.csv").write_text("frame_index,n_detections\n0,0\n1,10\n2,5\n", encoding="utf-8")
    (out / "phase_estimates.csv").write_text("frame_index,correction_px,score\n0,0,0.9\n1,8,0.1\n2,1,0.8\n", encoding="utf-8")

    rows = pri.suggest_label_frames(out, frame_count=3)

    assert rows
    assert {row["frame_index"] for row in rows}.issubset({0, 1, 2})


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
