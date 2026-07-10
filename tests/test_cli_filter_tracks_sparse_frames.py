from __future__ import annotations

import csv
from pathlib import Path

from beltmap.cli import filter_tracks as cli_filter_tracks


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def detection_row(frame_index: int, image: str, y: float) -> dict[str, object]:
    return {
        "frame_index": frame_index,
        "image": image,
        "label": 1,
        "y": y,
        "x": 5.0,
        "area_px": 6,
        "bbox_top": int(y) - 1,
        "bbox_left": 4,
        "bbox_bottom": int(y) + 2,
        "bbox_right": 7,
        "mean_signal": 4.5,
        "peak_signal": 7.5,
    }


def test_reconstruct_track_rows_keeps_sparse_absolute_frames_compact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    write_csv(
        tmp_path / "detections.csv",
        [
            detection_row(10, "frame10.bmp", 10.0),
            detection_row(12, "frame12.bmp", 18.0),
        ],
    )
    captured: dict[str, object] = {}

    def fake_track_particle_detections(
        detections_by_frame,
        *,
        frame_indices,
        config,
    ):
        captured["group_sizes"] = [len(group) for group in detections_by_frame]
        captured["frame_indices"] = frame_indices
        return [
            cli_filter_tracks.ParticleTrack(
                track_id=0,
                detections=tuple(group[0] for group in detections_by_frame),
            )
        ]

    monkeypatch.setattr(
        cli_filter_tracks,
        "track_particle_detections",
        fake_track_particle_detections,
    )

    rows = cli_filter_tracks.reconstruct_track_rows(tmp_path)

    assert getattr(
        cli_filter_tracks.reconstruct_track_rows,
        "_beltmap_filter_tracks_sparse_frame_patched",
        False,
    )
    assert captured["group_sizes"] == [1, 1]
    assert captured["frame_indices"] == [10.0, 12.0]
    assert [row["frame_index"] for row in rows] == [10, 12]
    assert [row["image"] for row in rows] == ["frame10.bmp", "frame12.bmp"]

    with (tmp_path / "tracks.csv").open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert [row["frame_index"] for row in written] == ["10", "12"]
