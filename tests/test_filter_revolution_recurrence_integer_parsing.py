from __future__ import annotations

import pytest

from beltmap.cli.filter_revolution_recurrence import accepted_track_ids
from beltmap.cli.filter_revolution_recurrence import filter_rows_by_track_id
from beltmap.cli.filter_revolution_recurrence import parse_detection
from beltmap.cli.filter_revolution_recurrence import parse_tracks
from beltmap.cli.filter_revolution_recurrence import write_csv


def track_row(**overrides: object) -> dict[str, str]:
    row = {
        "track_id": "1",
        "track_detection_index": "0",
        "frame_index": "0",
        "image": "frame_000.png",
        "label": "1",
        "y": "10.0",
        "x": "5.0",
        "area_px": "25",
        "bbox_top": "8",
        "bbox_left": "3",
        "bbox_bottom": "13",
        "bbox_right": "8",
        "mean_signal": "",
        "peak_signal": "20.0",
        "recurrent_artifact_overlap_fraction": "",
        "recurrent_artifact_probability": "",
        "recurrent_artifact_required_peak_signal": "",
    }
    row.update({key: str(value) for key, value in overrides.items()})
    return row


def test_parse_tracks_rejects_fractional_track_ids_instead_of_merging() -> None:
    rows = [
        track_row(track_id="1.2", frame_index="0"),
        track_row(track_id="1.8", frame_index="1"),
    ]

    with pytest.raises(ValueError, match="track_id must be a finite integer"):
        parse_tracks(rows)


@pytest.mark.parametrize(
    "field",
    [
        "label",
        "area_px",
        "bbox_top",
        "bbox_left",
        "bbox_bottom",
        "bbox_right",
    ],
)
def test_parse_detection_rejects_fractional_discrete_fields(field: str) -> None:
    row = track_row(**{field: "1.5"})

    with pytest.raises(ValueError, match=rf"{field} must be a finite integer"):
        parse_detection(row)


def test_integer_valued_numeric_forms_remain_supported() -> None:
    tracks = parse_tracks(
        [
            track_row(
                track_id="2e0",
                track_detection_index="0.0",
                label="1.0",
                area_px="25.0",
                bbox_top="8.0",
                bbox_left="3e0",
                bbox_bottom="13.0",
                bbox_right="8e0",
            )
        ]
    )

    assert len(tracks) == 1
    assert tracks[0].track_id == 2
    assert tracks[0].detections[0].label == 1
    assert tracks[0].detections[0].area_px == 25


def test_accepted_track_ids_rejects_fractional_ids(tmp_path) -> None:
    write_csv(
        tmp_path / "track_scores.csv",
        [{"track_id": "1.9", "accepted": "True"}],
    )

    with pytest.raises(ValueError, match="track_id must be a finite integer"):
        accepted_track_ids(tmp_path)


def test_filter_rows_rejects_fractional_ids_instead_of_aliasing() -> None:
    with pytest.raises(ValueError, match="track_id must be a finite integer"):
        filter_rows_by_track_id([{"track_id": "1.9"}], {1})
