from beltmap import texture_stress
from beltmap import texture_stress_velocity_assignment_patch as patch


def test_texture_stress_velocity_assignment_patch_is_autoloaded():
    assert texture_stress.velocity_rows_in_frames is patch.velocity_rows_in_frames
    assert patch.PATCH_MARKER == "texture-stress-velocity-single-anchor-v1"


def test_velocity_row_is_assigned_only_to_midpoint_subset():
    row = {
        "track_id": 7,
        "frame_start": 0,
        "frame_end": 8,
        "velocity_ratio_y": 0.8,
    }

    assignments = [
        texture_stress.velocity_rows_in_frames([row], frames)
        for frames in ({0}, {4}, {8})
    ]

    assert assignments == [[], [row], []]
    assert sum(len(items) for items in assignments) == 1


def test_velocity_row_falls_back_to_end_frame_when_start_is_missing():
    row = {
        "track_id": 8,
        "frame_start": "",
        "frame_end": 8,
        "velocity_ratio_y": 0.9,
    }

    assert texture_stress.velocity_rows_in_frames([row], {8}) == [row]
    assert texture_stress.velocity_rows_in_frames([row], {7}) == []
