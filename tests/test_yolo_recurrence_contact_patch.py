from __future__ import annotations

import beltmap.yolo_recurrence as yolo_recurrence
from beltmap.yolo_recurrence_contact_patch import (
    _clip_patch_to_crop,
    _patch_box_from_feature,
    contact_sheet_rows,
)


def test_contact_sheet_selection_prefers_shape_supported_recurrence() -> None:
    high_ratio_low_shape = {
        "frame_index": "10",
        "label": "1",
        "raw_match_role": "TP",
        "hard_reject": "False",
        "max_recurrence_ratio": "2.0",
        "belt_fixedness_score": "0.0",
        "valid_revisits": "2",
    }
    lower_ratio_shape_supported = {
        "frame_index": "11",
        "label": "2",
        "raw_match_role": "TP",
        "hard_reject": "False",
        "max_recurrence_ratio": "0.7",
        "belt_fixedness_score": "0.6",
        "valid_revisits": "2",
    }

    selected = contact_sheet_rows(
        [high_ratio_low_shape, lower_ratio_shape_supported],
        limit=2,
    )

    assert selected[0] is lower_ratio_shape_supported


def test_contact_sheet_selection_treats_blank_valid_revisits_as_zero() -> None:
    blank_revisits = {
        "frame_index": "12",
        "label": "3",
        "raw_match_role": "TP",
        "hard_reject": "False",
        "max_recurrence_ratio": "0.4",
        "belt_fixedness_score": "0.5",
        "valid_revisits": "",
    }

    selected = contact_sheet_rows([blank_revisits], limit=1)

    assert selected == [blank_revisits]


def test_contact_sheet_revisit_patch_clips_near_crop_boundary() -> None:
    row = {
        "patch_top": "0",
        "patch_left": "5",
        "patch_bottom": "20",
        "patch_right": "15",
        "revisit_y_prev": "3",
    }

    patch = _patch_box_from_feature(row, "previous")

    assert patch is not None
    assert patch.top < 0
    clipped = _clip_patch_to_crop(patch, (12, 30))
    assert clipped.top == 0
    assert clipped.bottom == 12
    assert clipped.height > 0


def test_yolo_recurrence_contact_sheet_selector_is_patched() -> None:
    assert yolo_recurrence.select_contact_rows is contact_sheet_rows
