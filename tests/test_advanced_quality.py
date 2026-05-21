import json

import numpy as np

from beltmap.advanced_quality import (
    bbox_iou,
    evaluate_real_detections,
    quadratic_subpixel_minimum,
    robust_gain_offset,
    theil_sen_slope,
    track_confidence_score,
)


def test_robust_gain_offset_recovers_linear_photometric_change():
    expected = np.arange(100, dtype=float).reshape(10, 10)
    observed = 1.5 * expected + 7.0
    observed[0, 0] += 1000.0

    fit = robust_gain_offset(observed, expected, trim_fraction=0.05, max_iterations=3, min_pixels=20)

    np.testing.assert_allclose(fit.gain, 1.5, rtol=1e-6)
    np.testing.assert_allclose(fit.offset, 7.0, rtol=1e-6)
    assert fit.n_pixels < expected.size


def test_quadratic_subpixel_minimum_fits_best_neighbor_triplet():
    offsets = [-1.0, 0.0, 1.0]
    losses = [(x - 0.25) ** 2 for x in offsets]

    assert abs(quadratic_subpixel_minimum(offsets, losses) - 0.25) < 1e-9


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
