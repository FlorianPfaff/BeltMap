import json

import numpy as np
import pytest

from beltmap.advanced_quality import (
    GainOffsetFit,
    apply_gain_offset,
    bbox_iou,
    evaluate_real_detections,
    estimate_integer_xy_shift,
    finite_int,
    quality_flags,
    quadratic_subpixel_minimum,
    robust_gain_offset,
    map_uncertainty_from_counts,
    seam_discontinuity_profile,
    smooth_phase_velocity,
    theil_sen_slope,
    track_confidence_score,
    unwrap_periodic,
)


def test_robust_gain_offset_recovers_linear_photometric_change():
    expected = np.arange(100, dtype=float).reshape(10, 10)
    observed = 1.5 * expected + 7.0
    observed[0, 0] += 1000.0

    fit = robust_gain_offset(observed, expected, trim_fraction=0.05, max_iterations=3, min_pixels=20)

    np.testing.assert_allclose(fit.gain, 1.5, rtol=1e-6)
    np.testing.assert_allclose(fit.offset, 7.0, rtol=1e-6)
    assert fit.n_pixels < expected.size


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"trim_fraction": True}, "trim_fraction"),
        ({"trim_fraction": "0.1"}, "trim_fraction"),
        ({"max_iterations": True}, "max_iterations"),
        ({"max_iterations": 1.5}, "max_iterations"),
        ({"min_pixels": True}, "min_pixels"),
        ({"min_pixels": 0}, "min_pixels"),
    ],
)
def test_robust_gain_offset_rejects_coerced_config(kwargs, message):
    expected = np.arange(100, dtype=float).reshape(10, 10)
    observed = expected.copy()

    with pytest.raises(ValueError, match=message):
        robust_gain_offset(observed, expected, **kwargs)


def test_robust_gain_offset_rejects_unidentifiable_constant_expected():
    expected = np.ones((10, 10), dtype=float)
    observed = expected + 2.0

    with pytest.raises(ValueError, match="distinct"):
        robust_gain_offset(observed, expected, min_pixels=20)


def test_apply_gain_offset_rejects_invalid_fit_values():
    expected = np.arange(4, dtype=float).reshape(2, 2)
    fit = GainOffsetFit(
        gain=float("nan"),
        offset=0.0,
        n_pixels=4,
        rmse_gray=0.0,
        trimmed_fraction=0.0,
    )

    with pytest.raises(ValueError, match="fit.gain"):
        apply_gain_offset(expected, fit)


def test_quadratic_subpixel_minimum_fits_best_neighbor_triplet():
    offsets = [-1.0, 0.0, 1.0]
    losses = [(x - 0.25) ** 2 for x in offsets]

    assert abs(quadratic_subpixel_minimum(offsets, losses) - 0.25) < 1e-9


def test_quadratic_subpixel_minimum_rejects_duplicate_offsets():
    with pytest.raises(ValueError, match="distinct"):
        quadratic_subpixel_minimum([0.0, 0.0, 1.0], [1.0, 0.5, 2.0])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_shift_y_px": True}, "max_shift_y_px"),
        ({"max_shift_x_px": 1.5}, "max_shift_x_px"),
        ({"trim_fraction": True}, "trim_fraction"),
        ({"trim_fraction": 1.0}, "trim_fraction"),
    ],
)
def test_estimate_integer_xy_shift_rejects_invalid_config(kwargs, message):
    image = np.arange(9, dtype=float).reshape(3, 3)

    with pytest.raises(ValueError, match=message):
        estimate_integer_xy_shift(image, image, **kwargs)


def test_estimate_integer_xy_shift_rejects_no_finite_overlap():
    observed = np.full((3, 3), np.nan)
    expected = np.arange(9, dtype=float).reshape(3, 3)

    with pytest.raises(ValueError, match="no finite overlap"):
        estimate_integer_xy_shift(observed, expected)


@pytest.mark.parametrize(
    ("period", "values", "message"),
    [
        (True, [0.0, 1.0], "period"),
        ("10.0", [0.0, 1.0], "period"),
        (10.0, [0.0, float("nan")], "values"),
    ],
)
def test_unwrap_periodic_rejects_invalid_inputs(period, values, message):
    with pytest.raises(ValueError, match=message):
        unwrap_periodic(values, period)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"process_noise_px": True}, "process_noise_px"),
        ({"measurement_noise_px": "2.0"}, "measurement_noise_px"),
        ({"scores": [float("inf"), 1.0]}, "scores"),
    ],
)
def test_smooth_phase_velocity_rejects_invalid_config(kwargs, message):
    options = {
        "period_px": 10.0,
    }
    options.update(kwargs)

    with pytest.raises(ValueError, match=message):
        smooth_phase_velocity([0.0, 1.0], **options)


def test_theil_sen_slope_ignores_single_bad_point_better_than_mean_slope():
    times = np.arange(6, dtype=float)
    values = 2.0 * times
    values[-1] += 100.0

    assert theil_sen_slope(times, values) == 2.0


def test_bbox_iou_and_track_confidence_are_finite():
    iou = bbox_iou({"top": 0, "left": 0, "bottom": 10, "right": 10}, {"top": 5, "left": 5, "bottom": 15, "right": 15})
    np.testing.assert_allclose(iou, 25 / 175)
    score = track_confidence_score(n_detections=5, min_track_length=5, mean_peak_signal=10.0, velocity_fit_rmse_px=0.5, velocity_ratio_y=0.8)
    assert 0.0 < score <= 1.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_count": True}, "min_count"),
        ({"scale": "1.0"}, "scale"),
        ({"min_count": 0.0}, "positive"),
    ],
)
def test_map_uncertainty_rejects_invalid_config(kwargs, message):
    with pytest.raises(ValueError, match=message):
        map_uncertainty_from_counts(np.ones((2, 2)), **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"seam_row": True}, "seam_row"),
        ({"window_px": 1.5}, "window_px"),
        ({"window_px": 0}, "window_px"),
    ],
)
def test_seam_discontinuity_rejects_invalid_config(kwargs, message):
    belt = np.arange(9, dtype=float).reshape(3, 3)

    with pytest.raises(ValueError, match=message):
        seam_discontinuity_profile(belt, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_detections": True}, "n_detections"),
        ({"n_detections": -1}, "n_detections"),
        ({"min_track_length": "5"}, "min_track_length"),
        ({"velocity_fit_rmse_px": -1.0}, "velocity_fit_rmse_px"),
        ({"velocity_ratio_y": float("nan")}, "velocity_ratio_y"),
    ],
)
def test_track_confidence_score_rejects_invalid_inputs(kwargs, message):
    options = {"n_detections": 5, "min_track_length": 5}
    options.update(kwargs)

    with pytest.raises(ValueError, match=message):
        track_confidence_score(**options)


def test_bbox_iou_rejects_invalid_coordinate_values():
    with pytest.raises(ValueError, match="a.top"):
        bbox_iou(
            {"top": True, "left": 0, "bottom": 10, "right": 10},
            {"top": 0, "left": 0, "bottom": 10, "right": 10},
        )


def test_finite_int_rejects_fractional_values():
    assert finite_int("7") == 7
    assert finite_int("7.0") == 7
    assert finite_int("7.5") is None
    assert finite_int(True) is None


def test_real_label_metrics_count_detections_only_on_labeled_frames(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "detections.csv").write_text(
        "frame_index,bbox_top,bbox_left,bbox_bottom,bbox_right\n"
        "0,0,0,10,10\n"
        "1,0,0,10,10\n",
        encoding="utf-8",
    )
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps({"frames": [{"frame_index": 0, "boxes": [{"top": 0, "left": 0, "bottom": 10, "right": 10}]}]}),
        encoding="utf-8",
    )

    metrics = evaluate_real_detections(out, labels, iou_threshold=0.5)

    assert metrics.detection_boxes == 1
    assert metrics.truth_boxes == 1
    assert metrics.matches == 1
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0


def test_real_label_metrics_ignore_fractional_detection_frame_indices(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "detections.csv").write_text(
        "frame_index,bbox_top,bbox_left,bbox_bottom,bbox_right\n"
        "0.5,0,0,10,10\n",
        encoding="utf-8",
    )
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps({"frames": [{"frame_index": 0, "boxes": [{"top": 0, "left": 0, "bottom": 10, "right": 10}]}]}),
        encoding="utf-8",
    )

    metrics = evaluate_real_detections(out, labels, iou_threshold=0.5)

    assert metrics.detection_boxes == 0
    assert metrics.truth_boxes == 1
    assert metrics.matches == 0
    assert metrics.precision is None
    assert metrics.recall == 0.0
    assert metrics.f1 is None


def test_real_label_metrics_score_clean_empty_labeled_frames(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "detections.csv").write_text(
        "frame_index,bbox_top,bbox_left,bbox_bottom,bbox_right\n",
        encoding="utf-8",
    )
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps({"frames": [{"frame_index": 0, "boxes": []}]}),
        encoding="utf-8",
    )

    metrics = evaluate_real_detections(out, labels, iou_threshold=0.5)

    assert metrics.frames == 1
    assert metrics.detection_boxes == 0
    assert metrics.truth_boxes == 0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0


def test_real_label_metrics_zero_match_f1_is_zero_not_missing(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "detections.csv").write_text(
        "frame_index,bbox_top,bbox_left,bbox_bottom,bbox_right\n"
        "0,20,20,30,30\n",
        encoding="utf-8",
    )
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps({"frames": [{"frame_index": 0, "boxes": [{"top": 0, "left": 0, "bottom": 10, "right": 10}]}]}),
        encoding="utf-8",
    )

    metrics = evaluate_real_detections(out, labels, iou_threshold=0.5)

    assert metrics.matches == 0
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0


def test_real_label_metrics_reject_unreviewed_json_scaffold(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "detections.csv").write_text(
        "frame_index,bbox_top,bbox_left,bbox_bottom,bbox_right\n",
        encoding="utf-8",
    )
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            {
                "status": "template_not_ground_truth_do_not_use_for_metrics_until_filled",
                "requires_manual_review": True,
                "frames": [{"frame_index": 0, "boxes": []}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reviewed ground truth"):
        evaluate_real_detections(out, labels, iou_threshold=0.5)


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        ({"frame_index": 0.5, "boxes": []}, "frame_index"),
        (42, "frames must be objects"),
    ],
)
def test_real_label_metrics_reject_invalid_reviewed_frames(tmp_path, frame, message):
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "detections.csv").write_text(
        "frame_index,bbox_top,bbox_left,bbox_bottom,bbox_right\n",
        encoding="utf-8",
    )
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            {
                "status": "reviewed_ground_truth",
                "requires_manual_review": False,
                "frames": [frame],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        evaluate_real_detections(out, labels, iou_threshold=0.5)


def test_real_label_metrics_reject_boolean_reviewed_frame_index(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "detections.csv").write_text(
        "frame_index,bbox_top,bbox_left,bbox_bottom,bbox_right\n",
        encoding="utf-8",
    )
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            {
                "status": "reviewed_ground_truth",
                "requires_manual_review": False,
                "frames": [{"frame_index": True, "boxes": []}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="frame_index"):
        evaluate_real_detections(out, labels, iou_threshold=0.5)


def test_real_label_metrics_reject_non_list_boxes(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "detections.csv").write_text(
        "frame_index,bbox_top,bbox_left,bbox_bottom,bbox_right\n",
        encoding="utf-8",
    )
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            {
                "status": "reviewed_ground_truth",
                "requires_manual_review": False,
                "frames": [{"frame_index": 0, "boxes": {"top": 0}}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="boxes must be a list"):
        evaluate_real_detections(out, labels, iou_threshold=0.5)


def test_real_label_metrics_reject_duplicate_reviewed_frames(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "detections.csv").write_text(
        "frame_index,bbox_top,bbox_left,bbox_bottom,bbox_right\n",
        encoding="utf-8",
    )
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            {
                "status": "reviewed_ground_truth",
                "requires_manual_review": False,
                "frames": [
                    {
                        "frame_index": 0,
                        "boxes": [{"top": 0, "left": 0, "bottom": 1, "right": 1}],
                    },
                    {
                        "frame_index": 0,
                        "boxes": [{"top": 2, "left": 2, "bottom": 3, "right": 3}],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate label frame_index"):
        evaluate_real_detections(out, labels, iou_threshold=0.5)


@pytest.mark.parametrize(
    ("box", "message"),
    [
        ({"top": 0, "left": 0, "bottom": float("nan"), "right": 10}, "finite"),
        ({"top": 10, "left": 0, "bottom": 10, "right": 10}, "positive"),
    ],
)
def test_real_label_metrics_reject_invalid_reviewed_boxes(tmp_path, box, message):
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "detections.csv").write_text(
        "frame_index,bbox_top,bbox_left,bbox_bottom,bbox_right\n",
        encoding="utf-8",
    )
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            {
                "status": "reviewed_ground_truth",
                "requires_manual_review": False,
                "frames": [{"frame_index": 0, "boxes": [box]}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        evaluate_real_detections(out, labels, iou_threshold=0.5)


def test_real_label_metrics_reject_boolean_box_coordinate(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "detections.csv").write_text(
        "frame_index,bbox_top,bbox_left,bbox_bottom,bbox_right\n",
        encoding="utf-8",
    )
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            {
                "status": "reviewed_ground_truth",
                "requires_manual_review": False,
                "frames": [
                    {
                        "frame_index": 0,
                        "boxes": [{"top": True, "left": 0, "bottom": 1, "right": 1}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="top"):
        evaluate_real_detections(out, labels, iou_threshold=0.5)


@pytest.mark.parametrize("iou_threshold", [True, "0.5", float("nan")])
def test_real_label_metrics_reject_invalid_iou_threshold(tmp_path, iou_threshold):
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "detections.csv").write_text(
        "frame_index,bbox_top,bbox_left,bbox_bottom,bbox_right\n",
        encoding="utf-8",
    )
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps({"frames": [{"frame_index": 0, "boxes": []}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="iou_threshold"):
        evaluate_real_detections(out, labels, iou_threshold=iou_threshold)


def test_real_label_metrics_ignore_degenerate_detection_boxes(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "detections.csv").write_text(
        "frame_index,bbox_top,bbox_left,bbox_bottom,bbox_right\n"
        "0,10,0,10,10\n",
        encoding="utf-8",
    )
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps({"frames": [{"frame_index": 0, "boxes": []}]}),
        encoding="utf-8",
    )

    metrics = evaluate_real_detections(out, labels, iou_threshold=0.5)

    assert metrics.detection_boxes == 0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0


def test_quality_flags_preserve_zero_detection_metadata_for_recurrent_filtering(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "metadata.json").write_text(
        json.dumps({"n_recurrent_artifact_rejected": 5, "n_detections": 0}),
        encoding="utf-8",
    )
    (out / "detections.csv").write_text(
        "frame_index,area_px\n0,10\n1,12\n",
        encoding="utf-8",
    )

    payload = quality_flags(out)

    flag = next(flag for flag in payload["flags"] if flag["code"] == "heavy_recurrent_filtering")
    assert flag["rejected"] == 5
    assert flag["share"] == 1.0


def test_quality_flags_rejects_zero_registration_search_step_metadata(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "metadata.json").write_text(
        json.dumps({"registration_search_radius_px": 4.0, "registration_search_step_px": 0.0}),
        encoding="utf-8",
    )
    (out / "phase_estimates.csv").write_text(
        "frame_index,correction_px,score\n0,4.0,0.8\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="registration_search_step_px must be positive"):
        quality_flags(out)
