import pytest

from beltmap.revolution_split import (
    build_revolution_split,
    revolution_split_frame_rows,
    revolution_split_revolution_rows,
)


def _split():
    return build_revolution_split([0, 0, 1, 1], eval_revolutions=(1,))


@pytest.mark.parametrize(
    "row_builder,kwargs",
    [
        (
            revolution_split_frame_rows,
            {"image_names": ["a.png", "b.png", "c.png", "d.png"]},
        ),
        (revolution_split_revolution_rows, {}),
    ],
)
def test_revolution_split_rows_reject_selected_evaluation_frames(row_builder, kwargs):
    with pytest.raises(ValueError, match="held-out evaluation frames"):
        row_builder(
            _split(),
            selected_train_frame_indices=(2,),
            **kwargs,
        )


@pytest.mark.parametrize(
    "row_builder,kwargs",
    [
        (
            revolution_split_frame_rows,
            {"image_names": ["a.png", "b.png", "c.png", "d.png"]},
        ),
        (revolution_split_revolution_rows, {}),
    ],
)
def test_revolution_split_rows_reject_out_of_range_training_frames(row_builder, kwargs):
    with pytest.raises(ValueError, match="outside the split"):
        row_builder(
            _split(),
            selected_train_frame_indices=(4,),
            **kwargs,
        )


def test_revolution_split_rows_keep_valid_training_selection():
    split = _split()
    frame_rows = revolution_split_frame_rows(
        split,
        image_names=["a.png", "b.png", "c.png", "d.png"],
        selected_train_frame_indices=(1,),
    )
    revolution_rows = revolution_split_revolution_rows(
        split,
        selected_train_frame_indices=(1,),
    )

    assert frame_rows[1]["selected_for_map_training"] is True
    assert frame_rows[2]["selected_for_map_training"] is False
    assert revolution_rows == [
        {
            "revolution_index": 0,
            "split": "train",
            "n_frames": 2,
            "n_selected_for_map_training": 1,
        },
        {
            "revolution_index": 1,
            "split": "eval",
            "n_frames": 2,
            "n_selected_for_map_training": 0,
        },
    ]
