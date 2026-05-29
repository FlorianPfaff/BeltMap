import csv
import json
from pathlib import Path

from PIL import Image

import beltmap.visual_qc as visual_qc
from beltmap.tracking import ParticleTrack
from beltmap.visual_qc import generate_visual_qc


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_qc_outputs(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "belt_velocity_px_per_frame": 2.0,
        "belt_map_height_px": 16,
        "belt_region": {"top": 0, "left": 0, "height": 8, "width": 8},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    phase_rows = [
        {"frame_index": 0, "phase_px": 0.0},
        {"frame_index": 1, "phase_px": 2.0},
        {"frame_index": 2, "phase_px": 4.0},
    ]
    detection_rows = [
        {
            "frame_index": 0,
            "label": 1,
            "y": 2.0,
            "x": 3.0,
            "bbox_top": 1,
            "bbox_left": 2,
            "bbox_bottom": 4,
            "bbox_right": 5,
        },
        {
            "frame_index": 1,
            "label": 1,
            "y": 4.0,
            "x": 3.0,
            "bbox_top": 3,
            "bbox_left": 2,
            "bbox_bottom": 6,
            "bbox_right": 5,
        },
        {
            "frame_index": 2,
            "label": 1,
            "y": 6.0,
            "x": 3.0,
            "bbox_top": 5,
            "bbox_left": 2,
            "bbox_bottom": 8,
            "bbox_right": 5,
        },
    ]
    write_csv(output_dir / "phase_estimates.csv", phase_rows)
    write_csv(output_dir / "detections.csv", detection_rows)
    for index in range(3):
        image = Image.new("L", (8, 8), 32 + 40 * index)
        image.save(output_dir / f"residual_frame_{index:06d}.png")
    return {
        "metadata": metadata,
        "phase_rows": phase_rows,
        "detections": detection_rows,
    }


def test_generate_visual_qc_writes_histogram_coverage_and_overlays(tmp_path):
    data = make_qc_outputs(tmp_path)

    artifacts = generate_visual_qc(tmp_path, data)

    assert set(artifacts.plots) == {
        "residual_histogram",
        "belt_map_coverage",
        "overlay_contact_sheet",
    }
    assert set(artifacts.images) == {"detections_overlay", "tracks_overlay"}
    for path in artifacts.plots.values():
        assert path.is_file()
        assert path.stat().st_size > 0
    assert len(artifacts.images["detections_overlay"]) == 3
    assert len(artifacts.images["tracks_overlay"]) == 3
    for paths in artifacts.images.values():
        for path in paths:
            assert path.is_file()
            assert path.stat().st_size > 0


def test_visual_qc_tracks_use_pyrecest_tracker(monkeypatch):
    calls = {}
    records = visual_qc.group_detections_by_frame(
        visual_qc.parse_detection_records(
            [
                {
                    "frame_index": 0,
                    "label": 1,
                    "y": 2.0,
                    "x": 3.0,
                    "bbox_top": 1,
                    "bbox_left": 2,
                    "bbox_bottom": 4,
                    "bbox_right": 5,
                },
                {
                    "frame_index": 2,
                    "label": 1,
                    "y": 4.0,
                    "x": 3.0,
                    "bbox_top": 3,
                    "bbox_left": 2,
                    "bbox_bottom": 6,
                    "bbox_right": 5,
                },
            ]
        )
    )

    def fake_track_particle_detections(detections, *, config, frame_indices):
        calls["detections"] = detections
        calls["config"] = config
        calls["frame_indices"] = frame_indices
        return [
            ParticleTrack(
                track_id=0,
                detections=tuple(item for frame in detections for item in frame),
            )
        ]

    monkeypatch.setattr(
        visual_qc,
        "track_particle_detections",
        fake_track_particle_detections,
    )

    tracks = visual_qc.reconstruct_tracks(
        records,
        max_match_distance_px=8.0,
        max_frame_gap=2.0,
    )

    assert len(tracks) == 1
    assert calls["frame_indices"] == [0, 2]
    assert calls["config"].max_match_distance_px == 8.0
    assert calls["config"].max_frame_gap == 2.0
    assert [detection.frame_index for detection in calls["detections"][1]] == [2.0]
