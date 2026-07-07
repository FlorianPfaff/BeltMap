from __future__ import annotations

import numpy as np

import beltmap  # noqa: F401
from beltmap.ghost_repair import build_ghost_defect_maps


def test_ghost_repair_defect_mask_clips_bottom_margin_to_visible_crop_height() -> None:
    assert getattr(build_ghost_defect_maps, "_beltmap_ghost_repair_crop_clipped", False)

    mask, counts, _probability, track_rows = build_ghost_defect_maps(
        belt_map_shape=(20, 6),
        tracks_by_id={
            1: [
                {
                    "frame_index": "0",
                    "bbox_top": "8",
                    "bbox_left": "1",
                    "bbox_bottom": "10",
                    "bbox_right": "3",
                    "peak_signal": "12",
                }
            ]
        },
        selected_track_ids={1},
        phase_by_frame={0.0: 0.0},
        metrics={"detection_config": {"crop_height_px": 10}},
        margin_px=4,
    )

    marked_y = set(np.nonzero(counts)[0].tolist())
    assert marked_y
    assert max(marked_y) == 10
    assert not any(y > 10 for y in marked_y)
    assert mask.any()
    assert track_rows[0]["track_id"] == 1
