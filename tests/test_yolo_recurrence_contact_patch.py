from __future__ import annotations

import beltmap.yolo_recurrence as yolo_recurrence
from beltmap.yolo_recurrence_contact_patch import contact_sheet_rows


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


def test_contact_sheet_selection_tolerates_blank_valid_revisits() -> None:
    row = {
        "frame_index": "12",
        "label": "3",
        "raw_match_role": "FP",
        "hard_reject": "False",
        "max_recurrence_ratio": "0.5",
        "belt_fixedness_score": "0.4",
        "valid_revisits": "",
    }

    assert contact_sheet_rows([row], limit=1) == [row]


def test_yolo_recurrence_contact_sheet_selector_is_patched() -> None:
    assert yolo_recurrence.select_contact_rows is contact_sheet_rows
