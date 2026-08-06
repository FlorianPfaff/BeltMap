from __future__ import annotations

import beltmap
from beltmap import revolution_split


def test_revolution_split_observed_order_patch_is_autoloaded() -> None:
    assert getattr(
        revolution_split.build_revolution_split,
        "_beltmap_observed_order_revolution_split_patched",
        False,
    )
    assert beltmap.build_revolution_split is revolution_split.build_revolution_split


def test_automatic_split_uses_observed_order_for_sparse_revolution_labels() -> None:
    split = revolution_split.build_revolution_split(
        [10, 10, 20, 20, 30, 30, 40, 40],
        eval_every=2,
        eval_offset=0,
    )

    assert split.eval_revolutions == (10, 30)
    assert split.train_revolutions == (20, 40)
    assert split.eval_frame_indices == (0, 1, 4, 5)
    assert split.train_frame_indices == (2, 3, 6, 7)


def test_explicit_eval_revolutions_keep_label_based_semantics() -> None:
    split = revolution_split.build_revolution_split(
        [10, 10, 20, 20, 30, 30, 40, 40],
        eval_revolutions=(40,),
    )

    assert split.eval_revolutions == (40,)
    assert split.train_revolutions == (10, 20, 30)
    assert split.eval_frame_indices == (6, 7)
