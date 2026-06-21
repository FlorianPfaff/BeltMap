from __future__ import annotations

import csv
import json

from beltmap.cli.filter_revolution_recurrence import filter_revolution_recurrence
from beltmap.revolution_recurrence import BeltRevolutionRecurrenceConfig
from beltmap.revolution_recurrence import score_belt_revolution_track_recurrence
from beltmap.tracking import ParticleDetection
from beltmap.tracking import ParticleTrack


def detection(frame: int, *, y: float, x: float, label: int = 1) -> ParticleDetection:
    return ParticleDetection(
        frame_index=float(frame),
        label=label,
        y=y,
        x=x,
        area_px=25,
        bbox_top=int(y - 2),
        bbox_left=int(x - 2),
        bbox_bottom=int(y + 3),
        bbox_right=int(x + 3),
        peak_signal=20.0,
    )


def track(track_id: int, detections: list[ParticleDetection]) -> ParticleTrack:
    return ParticleTrack(track_id=track_id, detections=tuple(detections))


def write_csv(path, rows):
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_belt_revolution_recurrence_rejects_only_leave_track_out_repeats():
    candidate = track(0, [detection(0, y=10.0, x=5.0)])
    recurring_1 = track(1, [detection(1, y=10.0, x=5.0)])
    recurring_2 = track(2, [detection(2, y=10.0, x=5.0)])
    one_off = track(3, [detection(0, y=16.0, x=30.0)])
    scores = score_belt_revolution_track_recurrence(
        [candidate, recurring_1, recurring_2, one_off],
        phase_px_by_frame=[0.0, 0.0, 0.0],
        revolution_by_frame=[0, 1, 2],
        frame_height_px=20.0,
        map_height_px=100.0,
        config=BeltRevolutionRecurrenceConfig(
            radius_y_px=2.0,
            radius_x_px=2.0,
            min_track_detections=1,
            min_other_revolutions=2,
            min_other_detections=2,
            min_recurrence_fraction=1.0,
        ),
    )

    by_id = {score.track_id: score for score in scores}
    assert by_id[0].runtime_recurrence_rejected is True
    assert by_id[0].other_hit_revolutions == 2
    assert by_id[3].runtime_recurrence_rejected is False
    assert by_id[3].causal_read == "belt-fixed track, but no leave-track-out recurrence"


def test_belt_revolution_recurrence_does_not_penalize_unexposed_cycles():
    candidate = track(0, [detection(0, y=10.0, x=5.0)])
    scores = score_belt_revolution_track_recurrence(
        [candidate],
        phase_px_by_frame=[0.0, 50.0],
        revolution_by_frame=[0, 1],
        frame_height_px=20.0,
        map_height_px=100.0,
        config=BeltRevolutionRecurrenceConfig(min_track_detections=1),
    )

    assert scores[0].other_exposed_revolutions == 0
    assert scores[0].runtime_recurrence_rejected is False
    assert scores[0].causal_read == "no other belt revolution exposed this coordinate"


def test_filter_revolution_recurrence_writes_filtered_outputs(tmp_path):
    input_dir = tmp_path / "run"
    input_dir.mkdir()
    (input_dir / "metadata.json").write_text(
        json.dumps(
            {
                "n_images": 3,
                "belt_region": {"top": 0, "left": 0, "height": 20, "width": 40},
                "belt_velocity_px_per_frame": 100.0,
                "belt_period_px_input": 100.0,
                "belt_map_height_px": 100.0,
                "reference_phase_px": 0.0,
            }
        ),
        encoding="utf-8",
    )
    write_csv(
        input_dir / "phase_estimates.csv",
        [
            {"frame_index": 0, "phase_px": 0},
            {"frame_index": 1, "phase_px": 0},
            {"frame_index": 2, "phase_px": 0},
        ],
    )
    track_rows = [
        {"track_id": 0, "track_detection_index": 0, "frame_index": 0, "image": "", "label": 1, "y": 10, "x": 5, "area_px": 25, "bbox_top": 8, "bbox_left": 3, "bbox_bottom": 13, "bbox_right": 8, "mean_signal": "", "peak_signal": 20},
        {"track_id": 1, "track_detection_index": 0, "frame_index": 1, "image": "", "label": 1, "y": 10, "x": 5, "area_px": 25, "bbox_top": 8, "bbox_left": 3, "bbox_bottom": 13, "bbox_right": 8, "mean_signal": "", "peak_signal": 20},
        {"track_id": 2, "track_detection_index": 0, "frame_index": 2, "image": "", "label": 1, "y": 10, "x": 5, "area_px": 25, "bbox_top": 8, "bbox_left": 3, "bbox_bottom": 13, "bbox_right": 8, "mean_signal": "", "peak_signal": 20},
    ]
    write_csv(input_dir / "tracks.csv", track_rows)
    write_csv(input_dir / "filtered_tracks.csv", [track_rows[0]])
    write_csv(
        input_dir / "velocities.csv",
        [
            {"track_id": 0, "n_detections": 1, "frame_start": 0, "frame_end": 0, "velocity_y_px_per_frame": 0, "velocity_x_px_per_frame": 0, "speed_px_per_frame": 0, "belt_velocity_y_px_per_frame": 100, "velocity_ratio_y": 0, "belt_minus_particle_velocity_y_px_per_frame": 100},
            {"track_id": 1, "n_detections": 1, "frame_start": 1, "frame_end": 1, "velocity_y_px_per_frame": 0, "velocity_x_px_per_frame": 0, "speed_px_per_frame": 0, "belt_velocity_y_px_per_frame": 100, "velocity_ratio_y": 0, "belt_minus_particle_velocity_y_px_per_frame": 100},
            {"track_id": 2, "n_detections": 1, "frame_start": 2, "frame_end": 2, "velocity_y_px_per_frame": 0, "velocity_x_px_per_frame": 0, "speed_px_per_frame": 0, "belt_velocity_y_px_per_frame": 100, "velocity_ratio_y": 0, "belt_minus_particle_velocity_y_px_per_frame": 100},
        ],
    )
    write_csv(
        input_dir / "track_scores.csv",
        [
            {"track_id": 0, "accepted": "True"},
            {"track_id": 1, "accepted": "False"},
            {"track_id": 2, "accepted": "False"},
        ],
    )

    summary = filter_revolution_recurrence(
        input_dir=input_dir,
        output_dir=tmp_path / "filtered",
        config=BeltRevolutionRecurrenceConfig(
            radius_y_px=2.0,
            radius_x_px=2.0,
            min_track_detections=1,
            min_other_revolutions=2,
            min_other_detections=2,
            min_recurrence_fraction=1.0,
        ),
    )

    assert summary["rejected_track_ids"] == [0]
    filtered_tracks = list(csv.DictReader((tmp_path / "filtered" / "filtered_tracks.csv").open()))
    assert filtered_tracks == []
