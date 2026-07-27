import importlib

import numpy as np

from beltmap import ghost_repair
from beltmap import ghost_repair_crop_clip_patch


def test_finite_int_requires_exact_nonnegative_integer():
    assert ghost_repair.finite_int("7") == 7
    assert ghost_repair.finite_int(7.0) == 7
    assert ghost_repair.finite_int(np.int64(7)) == 7

    for value in ("7.4", 7.5, -1, True, np.bool_(False), float("nan")):
        assert ghost_repair.finite_int(value) is None


def test_fractional_track_ids_are_not_rounded_into_valid_tracks():
    grouped = ghost_repair.track_rows_by_id(
        [
            {"track_id": "4", "frame_index": "0"},
            {"track_id": "4.4", "frame_index": "1"},
            {"track_id": "3.6", "frame_index": "2"},
        ]
    )

    assert grouped == {4: [{"track_id": "4", "frame_index": "0"}]}


def test_fractional_and_boolean_counts_are_not_invented(tmp_path):
    row = ghost_repair.map_only_metric_row(
        "candidate",
        {
            "detections": {"false_detections": 0.6},
            "tracks": {"false_tracks": True, "false_long_tracks": 1.4},
            "velocities": {"false_accepted_tracks": -1},
        },
        tmp_path / "belt_map.npy",
    )

    assert row["map_only_false_detections"] == 0
    assert row["map_only_false_tracks"] == 0
    assert row["map_only_false_long_tracks"] == 0
    assert row["map_only_false_accepted_tracks"] == 0
    assert row["map_only_proxy_ghost_penalty"] == 0.0


def test_integer_parser_patch_is_reload_safe():
    importlib.reload(ghost_repair_crop_clip_patch)

    assert ghost_repair.finite_int("2") == 2
    assert ghost_repair.finite_int("2.5") is None
