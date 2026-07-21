from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import beltmap  # noqa: F401 - imports side-effect patches
import beltmap.revolution_split as revolution_split


def test_revolution_split_boolean_patch_is_autoloaded() -> None:
    assert getattr(
        revolution_split._nonnegative_integer_value,
        "_beltmap_revolution_split_boolean_patched",
        False,
    )


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (
            lambda: revolution_split.build_revolution_split(
                [False, 0, 1],
                eval_revolutions=(1,),
            ),
            "revolution indices",
        ),
        (
            lambda: revolution_split.build_revolution_split(
                [0, 1],
                eval_every=True,
            ),
            "eval_every",
        ),
        (
            lambda: revolution_split.build_revolution_split(
                [0, 1],
                eval_offset=np.bool_(False),
            ),
            "eval_offset",
        ),
        (
            lambda: revolution_split.build_revolution_split(
                [0, 1],
                eval_revolutions=(np.bool_(True),),
            ),
            "eval_revolutions",
        ),
    ],
)
def test_revolution_split_rejects_boolean_integer_inputs(
    operation: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        operation()


def test_revolution_split_rows_reject_boolean_training_frame_indices() -> None:
    split = revolution_split.build_revolution_split(
        [0, 0, 1, 1],
        eval_revolutions=(1,),
    )

    with pytest.raises(ValueError, match="selected_train_frame_indices"):
        revolution_split.revolution_split_frame_rows(
            split,
            image_names=["a.png", "b.png", "c.png", "d.png"],
            selected_train_frame_indices=(True,),
        )


def test_revolution_split_still_accepts_integer_like_numeric_inputs() -> None:
    split = revolution_split.build_revolution_split(
        [0.0, 0, 1.0, 1],
        eval_every=2.0,
        eval_offset=1.0,
    )

    assert split.train_revolutions == (0,)
    assert split.eval_revolutions == (1,)
