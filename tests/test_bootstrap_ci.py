import pytest
import numpy as np

from beltmap.bootstrap_ci import (
    aggregate_labeled_outcomes,
    bootstrap_run_summary,
    ci_summary,
    finite_values,
    labeled_frame_outcomes,
    resample_indices,
)


def test_labeled_frame_outcomes_restricts_truth_to_scored_frames():
    truth = {
        "particles": [
            {"frame_index": 0, "top": 0, "left": 0, "bottom": 10, "right": 10},
            {"frame_index": 99, "top": 0, "left": 0, "bottom": 10, "right": 10},
        ]
    }
    detections = [
        {
            "frame_index": "0",
            "bbox_top": "0",
            "bbox_left": "0",
            "bbox_bottom": "10",
            "bbox_right": "10",
            "y": "5",
            "x": "5",
        },
        {
            "frame_index": "99",
            "bbox_top": "20",
            "bbox_left": "20",
            "bbox_bottom": "30",
            "bbox_right": "30",
            "y": "25",
            "x": "25",
        },
    ]

    outcomes = labeled_frame_outcomes(
        detections,
        truth,
        scored_frames={0},
        iou_threshold=0.5,
    )
    metrics = aggregate_labeled_outcomes(outcomes)

    assert [outcome.frame_index for outcome in outcomes] == [0]
    assert metrics["labeled_truth_boxes"] == 1
    assert metrics["labeled_predicted_boxes"] == 1
    assert metrics["labeled_false_negatives"] == 0
    assert metrics["labeled_recall"] == pytest.approx(1.0)


def test_labeled_frame_outcomes_ignores_boolean_frame_indices():
    truth = {
        "particles": [
            {"frame_index": 1, "top": 0, "left": 0, "bottom": 10, "right": 10},
        ]
    }
    detections = [
        {
            "frame_index": True,
            "bbox_top": "0",
            "bbox_left": "0",
            "bbox_bottom": "10",
            "bbox_right": "10",
            "y": "5",
            "x": "5",
        },
    ]

    outcomes = labeled_frame_outcomes(
        detections,
        truth,
        scored_frames={1},
        iou_threshold=0.5,
    )
    metrics = aggregate_labeled_outcomes(outcomes)

    assert metrics["labeled_predicted_boxes"] == 0
    assert metrics["labeled_false_negatives"] == 1


def test_finite_values_ignores_boolean_samples():
    assert finite_values([{"value": True}, {"value": "2.5"}], "value") == [2.5]


def test_ci_summary_ignores_boolean_estimates():
    assert ci_summary([True, 2.0, 4.0], confidence_level=0.5)[0] == pytest.approx(3.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"samples": True}, "bootstrap samples"),
        ({"confidence_level": True}, "bootstrap confidence level"),
        ({"block_length_frames": True}, "bootstrap block length"),
        ({"seed": True}, "bootstrap seed"),
        ({"truth_iou_threshold": True}, "truth_iou_threshold"),
    ],
)
def test_bootstrap_run_summary_rejects_boolean_numeric_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        bootstrap_run_summary(
            detections_per_frame=[],
            detections=[],
            velocities=[],
            filtered_velocities=[],
            **kwargs,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_units": True}, "n_units"),
        ({"block_length": True}, "block_length"),
    ],
)
def test_resample_indices_rejects_boolean_numeric_options(kwargs, message):
    options = {"n_units": 3, "block_length": 1}
    options.update(kwargs)

    with pytest.raises(ValueError, match=message):
        resample_indices(
            options["n_units"],
            rng=np.random.default_rng(0),
            block_length=options["block_length"],
        )
