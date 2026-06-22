import csv
import json

import numpy as np
from PIL import Image

from beltmap.cli import ghost_repair as ghost_repair_cli
from beltmap.ghost_repair import (
    build_ghost_defect_maps,
    local_inpaint_belt_map,
    phase_for_frame,
    rebuild_masked_driver_environment,
    run_rebuild_masked_apply,
    selected_ghost_track_ids,
    write_defect_report,
)
from beltmap.map_only_negative_control import (
    MapOnlyNegativeControlConfig,
    generate_map_only_negative_control_report,
)


def write_phase_estimates(path, *, frames: int, velocity: float, period: float) -> None:
    fieldnames = [
        "frame_index",
        "image",
        "phase_px",
        "predicted_phase_px",
        "correction_px",
        "phase_drift_px",
        "loss",
        "score",
        "method",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(frames):
            phase = (-velocity * index) % period
            writer.writerow(
                {
                    "frame_index": index,
                    "image": f"frame_{index:06d}.png",
                    "phase_px": phase,
                    "predicted_phase_px": phase,
                    "correction_px": 0.0,
                    "phase_drift_px": 0.0,
                    "loss": "",
                    "score": "",
                    "method": "test",
                }
            )


def test_build_ghost_defect_maps_projects_crop_boxes_to_belt_coordinates():
    tracks_by_id = {
        7: [
            {
                "track_id": "7",
                "frame_index": "0",
                "bbox_top": "2",
                "bbox_left": "3",
                "bbox_bottom": "4",
                "bbox_right": "5",
                "peak_signal": "12",
            }
        ]
    }

    mask, counts, probability, rows = build_ghost_defect_maps(
        belt_map_shape=(20, 10),
        tracks_by_id=tracks_by_id,
        selected_track_ids={7},
        phase_by_frame={0.0: 5.0},
        metrics={},
        margin_px=0,
    )

    assert rows[0]["track_id"] == 7
    assert rows[0]["n_detections"] == 1
    assert mask[7, 3]
    assert mask[8, 3]
    assert mask[9, 4]
    assert counts.max() == 2
    assert probability.max() == 1.0


def test_phase_for_frame_preserves_zero_velocity_metadata():
    phase = phase_for_frame(
        5.0,
        phase_by_frame={},
        metrics={
            "phase_source": {
                "period_px": 40.0,
                "belt_velocity_px_per_frame": 0.0,
            },
            "detection_config": {
                "period_px": 40.0,
                "belt_velocity_px_per_frame": 3.0,
            },
        },
        map_height=40,
    )

    assert phase == 0.0


def test_local_inpaint_replaces_masked_pixels_from_neighbors():
    belt_map = np.arange(25, dtype=np.float32).reshape(5, 5)
    belt_map[2, 2] = 1000.0
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True

    repaired = local_inpaint_belt_map(belt_map, mask, radius_px=1)

    assert repaired.shape == belt_map.shape
    assert repaired[2, 2] != belt_map[2, 2]
    np.testing.assert_allclose(repaired[0, 0], belt_map[0, 0])


def test_write_defect_report_allows_missing_peak_signal(tmp_path):
    report_path = tmp_path / "defect_report.md"

    write_defect_report(
        report_path,
        track_rows=[
            {
                "track_id": 7,
                "n_detections": 3,
                "map_y_min": 1.0,
                "map_y_max": 3.0,
                "map_x_min": 4.0,
                "map_x_max": 5.0,
                "max_signal": "",
                "belt_y_rms_px": 0.25,
                "belt_x_std_px": 0.5,
            }
        ],
        defect_pixels=4,
        max_count=2,
        overlay_path=tmp_path / "overlay.png",
    )

    text = report_path.read_text(encoding="utf-8")

    assert "| 7 | 3 | 1.000 | 3.000 | 4.000 | 5.000 |  | 0.250 | 0.500 |" in text


def test_selected_ghost_track_ids_uses_long_or_accepted_tracks_only():
    tracks_by_id = {
        1: [{"track_id": "1"}] * 3,
        2: [{"track_id": "2"}] * 1,
        3: [{"track_id": "3"}] * 1,
    }

    selected = selected_ghost_track_ids(
        tracks_by_id=tracks_by_id,
        track_scores=[
            {"track_id": "2", "accepted": "true"},
            {"track_id": "3", "accepted": "false"},
        ],
        velocities=[
            {"track_id": "2"},
            {"track_id": "3"},
        ],
        long_track_length=3,
    )

    assert selected == {1, 2}


def test_selected_ghost_track_ids_uses_velocity_fallback_without_scores():
    selected = selected_ghost_track_ids(
        tracks_by_id={1: [{"track_id": "1"}]},
        track_scores=[],
        velocities=[{"track_id": "1"}],
        long_track_length=10,
    )

    assert selected == {1}


def test_rebuild_masked_driver_environment_forces_rebuild(tmp_path):
    config_path = tmp_path / "config_resolved.json"
    output_dir = tmp_path / "rebuild"
    mask_path = tmp_path / "ghost_defect_mask.npy"
    config_path.write_text(
        """
{
  "driver_environment": {
    "BELTMAP_IMAGE_DIR": "data/images",
    "BELTMAP_OUTPUT_DIR": "old-output",
    "REUSE_BELT_MAP_PATH": "old-output/belt_map.npy",
    "REUSE_PHASE_ESTIMATES_PATH": "old-output/phase_estimates.csv"
  }
}
""".strip(),
        encoding="utf-8",
    )

    env = rebuild_masked_driver_environment(
        resolved_config_path=config_path,
        output_dir=output_dir,
        mask_path=mask_path,
    )

    assert env["BELTMAP_OUTPUT_DIR"] == str(output_dir)
    assert env["MAP_EXCLUSION_MASK_PATH"] == str(mask_path)
    assert env["BELTMAP_STOP_AFTER_BELT_MAP"] == "1"
    assert "REUSE_BELT_MAP_PATH" not in env
    assert "REUSE_PHASE_ESTIMATES_PATH" not in env
    assert env["BELTMAP_IMAGE_DIR"] == "data/images"


def test_run_rebuild_masked_apply_excludes_defect_coordinate(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    frame = np.asarray(
        [
            [10, 20, 30, 40],
            [50, 60, 70, 80],
            [90, 100, 110, 120],
            [130, 140, 150, 160],
        ],
        dtype=np.uint8,
    )
    for index in range(2):
        Image.fromarray(frame).save(image_dir / f"frame_{index:03d}.png")

    mask_path = tmp_path / "ghost_defect_mask.npy"
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 1] = True
    np.save(mask_path, mask)

    config_path = tmp_path / "config_resolved.json"
    config_path.write_text(
        json.dumps(
            {
                "driver_environment": {
                    "BELTMAP_IMAGE_DIR": str(image_dir),
                    "BELTMAP_OUTPUT_DIR": str(tmp_path / "old-output"),
                    "BELT_REGION": "0,0,4,4",
                    "BELT_VELOCITY_PX_PER_FRAME": "0",
                    "BELT_PERIOD_PX": "4",
                    "MAX_FRAMES": "2",
                    "MAP_SAMPLE_FRAMES": "2",
                    "MAP_MASK_ITERATIONS": "0",
                    "MAP_AGGREGATION": "mean",
                    "PHASE_ESTIMATION_MODE": "motion_model",
                    "DETECTION_THRESHOLD": "999",
                    "MIN_AREA_PX": "1",
                    "TRACK_FILTER_MIN_LENGTH": "1",
                    "DEBUG_RESIDUAL_PREVIEW_FRAMES": "0",
                    "STATIC_NOISE_SAMPLE_FRAMES": "0",
                    "STATIC_BACKGROUND_SAMPLE_FRAMES": "0",
                    "RECURRENT_ARTIFACT_MIN_REVOLUTIONS": "0",
                    "PROGRESS_INTERVAL_FRAMES": "1000",
                }
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "rebuild-output"
    rebuilt_map_path = run_rebuild_masked_apply(
        resolved_config_path=config_path,
        output_dir=output_dir,
        mask_path=mask_path,
    )

    assert rebuilt_map_path == output_dir / "belt_map.npy"
    assert rebuilt_map_path.is_file()
    support = np.load(output_dir / "belt_map_support.npy")
    rebuilt_map = np.load(rebuilt_map_path)
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))

    assert support[0, 1] == 0.0
    assert support[1, 1] > 0.0
    assert rebuilt_map[0, 1] != frame[0, 1]
    assert metadata["map_exclusion_mask_path"] == str(mask_path)
    assert metadata["stop_after_belt_map"] is True


def test_ghost_repair_cli_scores_run_rebuild_masked(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    repair_dir = tmp_path / "repair"

    belt_map = np.zeros((40, 24), dtype=np.float32)
    belt_map[8:12, 6:10] = 100.0
    np.save(input_dir / "belt_map.npy", belt_map)
    write_phase_estimates(input_dir / "phase_estimates.csv", frames=7, velocity=1.0, period=40.0)

    generate_map_only_negative_control_report(
        output_dir=input_dir,
        config=MapOnlyNegativeControlConfig(
            threshold=3.0,
            min_area_px=4,
            highpass_radius_px=5,
            crop_height_px=20,
            belt_velocity_px_per_frame=1.0,
            max_match_distance_px=5.0,
            min_track_length=2,
            track_filter_min_length=2,
            long_track_length=3,
        ),
    )

    raw_frame = np.zeros((20, 24), dtype=np.uint8)
    raw_frame[8:12, 6:10] = 100
    for index in range(2):
        Image.fromarray(raw_frame).save(image_dir / f"frame_{index:03d}.png")
    config_path = input_dir / "config_resolved.json"
    config_path.write_text(
        json.dumps(
            {
                "driver_environment": {
                    "BELTMAP_IMAGE_DIR": str(image_dir),
                    "BELTMAP_OUTPUT_DIR": str(tmp_path / "old-output"),
                    "BELT_REGION": "0,0,20,24",
                    "BELT_VELOCITY_PX_PER_FRAME": "0",
                    "BELT_PERIOD_PX": "40",
                    "MAX_FRAMES": "2",
                    "MAP_SAMPLE_FRAMES": "2",
                    "MAP_MASK_ITERATIONS": "0",
                    "MAP_AGGREGATION": "mean",
                    "PHASE_ESTIMATION_MODE": "motion_model",
                    "DETECTION_THRESHOLD": "999",
                    "MIN_AREA_PX": "1",
                    "TRACK_FILTER_MIN_LENGTH": "1",
                    "DEBUG_RESIDUAL_PREVIEW_FRAMES": "0",
                    "STATIC_NOISE_SAMPLE_FRAMES": "0",
                    "STATIC_BACKGROUND_SAMPLE_FRAMES": "0",
                    "RECURRENT_ARTIFACT_MIN_REVOLUTIONS": "0",
                    "PROGRESS_INTERVAL_FRAMES": "1000",
                }
            }
        ),
        encoding="utf-8",
    )

    exit_code = ghost_repair_cli.main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(repair_dir),
            "--run-rebuild-masked",
            "--quiet",
        ]
    )

    assert exit_code == 0
    with (repair_dir / "ghost_repair_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = {row["map_variant"]: row for row in csv.DictReader(handle)}

    assert int(rows["original"]["map_only_false_detections"]) > 0
    assert rows["local_inpaint"]["map_only_false_detections"] == "0"
    assert rows["rebuild_masked"]["map_only_false_detections"] == "0"
    assert (repair_dir / "rebuild_masked_apply" / "belt_map.npy").is_file()
    manifest = json.loads((repair_dir / "rebuild_masked_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "executed"
