import pytest
import numpy as np

from beltmap.bootstrap_ci import (
    LabeledFrameOutcome,
    aggregate_labeled_outcomes,
    bootstrap_labeled_metrics,
    bootstrap_numeric_metrics,
    bootstrap_run_summary,
    ci_summary,
    count_ge,
    finite_values,
    labeled_frame_outcomes,
    mean_value,
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
        ({"samples": True}, "samples"),
        ({"confidence_level": True}, "confidence_level"),
        ({"block_length_frames": True}, "block_length_frames"),
        ({"seed": True}, "seed"),
        ({"truth_iou_threshold": True}, "iou_threshold"),
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


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_units": 2.5}, "n_units"),
        ({"n_units": 2, "block_length": 1.5}, "block_length"),
    ],
)
def test_resample_indices_rejects_fractional_integer_config(kwargs, message):
    rng = np.random.default_rng(0)

    with pytest.raises(ValueError, match=message):
        resample_indices(rng=rng, **kwargs)


@pytest.mark.parametrize("confidence_level", [0.0, 1.0, float("nan")])
def test_ci_summary_rejects_invalid_confidence_level(confidence_level):
    with pytest.raises(ValueError, match="confidence_level"):
        ci_summary([1.0, 2.0], confidence_level=confidence_level)


def test_bootstrap_numeric_metrics_rejects_invalid_sampling_config():
    rng = np.random.default_rng(0)

    with pytest.raises(ValueError, match="samples"):
        bootstrap_numeric_metrics(
            [1.0, 2.0],
            {"mean": mean_value},
            samples=-1,
            confidence_level=0.95,
            rng=rng,
        )


def test_count_ge_rejects_nonfinite_threshold():
    with pytest.raises(ValueError, match="threshold"):
        count_ge(float("nan"))


def test_labeled_frame_outcomes_rejects_invalid_iou_threshold():
    with pytest.raises(ValueError, match="iou_threshold"):
        labeled_frame_outcomes(
            [], {"particles": []}, scored_frames={0}, iou_threshold=float("nan")
        )


def test_labeled_frame_outcomes_rejects_fractional_scored_frame():
    with pytest.raises(ValueError, match="scored_frames"):
        labeled_frame_outcomes(
            [], {"particles": []}, scored_frames={0.5}, iou_threshold=0.5
        )


def test_aggregate_labeled_outcomes_rejects_inconsistent_counts():
    with pytest.raises(ValueError, match="false_positives"):
        aggregate_labeled_outcomes(
            [
                LabeledFrameOutcome(
                    frame_index=0,
                    truth_boxes=1,
                    predicted_boxes=1,
                    true_positives=1,
                    false_positives=1,
                    false_negatives=0,
                )
            ]
        )


def test_bootstrap_labeled_metrics_rejects_fractional_block_length():
    rng = np.random.default_rng(0)

    with pytest.raises(ValueError, match="block_length_frames"):
        bootstrap_labeled_metrics(
            [],
            samples=1,
            confidence_level=0.95,
            rng=rng,
            block_length_frames=1.5,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"samples": 1.5}, "samples"),
        ({"truth_iou_threshold": 0.0}, "iou_threshold"),
        ({"block_length_frames": 1.5}, "block_length_frames"),
    ],
)
def test_bootstrap_run_summary_rejects_invalid_config(kwargs, message):
    base_kwargs = {
        "detections_per_frame": [],
        "detections": [],
        "velocities": [],
        "filtered_velocities": [],
        "samples": 1,
        "confidence_level": 0.95,
        "block_length_frames": 1,
    }
    base_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        bootstrap_run_summary(**base_kwargs)
