from dataclasses import dataclass

import pytest

from beltmap._driver_map import select_map_sample_indices
from beltmap.revolution_split import (
    build_revolution_split,
    parse_revolution_indices,
    revolution_split_detection_summary_rows,
    revolution_split_frame_rows,
    revolution_split_score_summary_rows,
)


@dataclass(frozen=True)
class DummyDetection:
    area_px: float
    peak_signal: float | None = None


@dataclass(frozen=True)
class DummyScore:
    rejected: bool
    overlap_fraction: float
    artifact_probability: float


def test_parse_revolution_indices_accepts_ranges_and_deduplicates():
    assert parse_revolution_indices("1, 3-5, 3") == (1, 3, 4, 5)


def test_parse_revolution_indices_rejects_decreasing_ranges():
    with pytest.raises(ValueError, match="ranges"):
        parse_revolution_indices("5-3")


def test_build_revolution_split_holds_out_modulo_class():
    split = build_revolution_split(
        [0, 0, 1, 1, 2, 2, 3, 3],
        eval_every=2,
        eval_offset=1,
    )

    assert split.train_revolutions == (0, 2)
    assert split.eval_revolutions == (1, 3)
    assert split.train_frame_indices == (0, 1, 4, 5)
    assert split.eval_frame_indices == (2, 3, 6, 7)
    assert split.frame_split == (
        "train",
        "train",
        "eval",
        "eval",
        "train",
        "train",
        "eval",
        "eval",
    )


def test_build_revolution_split_explicit_eval_revolutions():
    split = build_revolution_split(
        [0, 0, 1, 1, 2, 2],
        eval_revolutions=(2,),
    )

    assert split.train_revolutions == (0, 1)
    assert split.eval_revolutions == (2,)


def test_build_revolution_split_rejects_missing_explicit_revolutions():
    with pytest.raises(ValueError, match="not present"):
        build_revolution_split([0, 0, 1, 1], eval_revolutions=(3,))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"revolution_by_frame": [0, 0.5, 1]}, "revolution indices"),
        ({"revolution_by_frame": [0, 1], "eval_every": 1.5}, "eval_every"),
        ({"revolution_by_frame": [0, 1], "eval_offset": 0.5}, "eval_offset"),
        (
            {"revolution_by_frame": [0, 1], "eval_revolutions": (0.5,)},
            "eval_revolutions",
        ),
        (
            {"revolution_by_frame": [0, 1], "min_train_revolutions": 1.5},
            "min_train_revolutions",
        ),
    ],
)
def test_build_revolution_split_rejects_fractional_integer_inputs(kwargs, message):
    revolution_by_frame = kwargs.pop("revolution_by_frame")

    with pytest.raises(ValueError, match=message):
        build_revolution_split(revolution_by_frame, **kwargs)


def test_revolution_split_frame_rows_marks_map_training_samples():
    split = build_revolution_split([0, 0, 1, 1], eval_revolutions=(1,))
    rows = revolution_split_frame_rows(
        split,
        image_names=["a.png", "b.png", "c.png", "d.png"],
        selected_train_frame_indices=(1,),
    )

    assert rows[1]["selected_for_map_training"] is True
    assert rows[2]["split"] == "eval"
    assert rows[2]["selected_for_map_training"] is False


def test_revolution_split_frame_rows_rejects_fractional_selected_training_indices():
    split = build_revolution_split([0, 0, 1, 1], eval_revolutions=(1,))

    with pytest.raises(ValueError, match="selected_train_frame_indices"):
        revolution_split_frame_rows(
            split,
            image_names=["a.png", "b.png", "c.png", "d.png"],
            selected_train_frame_indices=(1.5,),
        )


def test_revolution_split_detection_summary_counts_train_and_eval():
    split = build_revolution_split([0, 0, 1, 1], eval_revolutions=(1,))
    detections = [
        [DummyDetection(area_px=4, peak_signal=6.0)],
        [],
        [DummyDetection(area_px=8, peak_signal=10.0)],
        [DummyDetection(area_px=12, peak_signal=None)],
    ]

    rows = revolution_split_detection_summary_rows(
        split,
        detections,
        stage="pre",
    )
    train = next(
        row for row in rows if row["group"] == "split" and row["split"] == "train"
    )
    eval_row = next(
        row for row in rows if row["group"] == "split" and row["split"] == "eval"
    )

    assert train["n_detections"] == 1
    assert train["detections_per_frame"] == 0.5
    assert eval_row["n_detections"] == 2
    assert eval_row["mean_area_px"] == 10.0
    assert eval_row["mean_peak_signal"] == 10.0


def test_revolution_split_score_summary_counts_rejections():
    split = build_revolution_split([0, 0, 1, 1], eval_revolutions=(1,))
    scores = [[DummyScore(False, 0.1, 0.2)], [], [DummyScore(True, 1.0, 0.8)], []]

    rows = revolution_split_score_summary_rows(split, scores, stage="ghost")
    eval_row = next(
        row for row in rows if row["group"] == "split" and row["split"] == "eval"
    )

    assert eval_row["n_detections"] == 1
    assert eval_row["n_rejected"] == 1
    assert eval_row["rejected_fraction"] == 1.0


def test_map_sampling_can_be_restricted_to_train_frame_pool():
    selected = select_map_sample_indices(
        frame_count=10,
        sample_count=2,
        velocity=1.0,
        reference_phase=0.0,
        model_period=10.0,
        map_height=10,
        crop_height=3,
        sampling_strategy="uniform",
        allowed_indices=[2, 4, 8],
    )

    assert selected == [2, 8]
