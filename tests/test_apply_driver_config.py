import pytest
import numpy as np
from PIL import Image

from beltmap import BeltRegion, CleanBeltRender, PhaseEstimate, ResidualImage, render_belt_view
from scripts.apply_beltmap_to_images import (
    DATA,
    build_belt_map,
    phase_estimate_row,
    validate_auto_velocity_estimate,
    validate_auto_velocity_region,
)


def test_auto_velocity_rejects_full_frame_region_by_default(monkeypatch):
    monkeypatch.delenv("ALLOW_FULL_FRAME_AUTO_VELOCITY", raising=False)

    with pytest.raises(ValueError, match="full-frame BELT_REGION"):
        validate_auto_velocity_region((0, 0, 1728, 2320), (1728, 2320))


def test_auto_velocity_allows_full_frame_region_when_explicitly_enabled(monkeypatch):
    monkeypatch.setenv("ALLOW_FULL_FRAME_AUTO_VELOCITY", "1")

    validate_auto_velocity_region((0, 0, 1728, 2320), (1728, 2320))


def test_auto_velocity_rejects_near_zero_shift(monkeypatch):
    monkeypatch.setenv("AUTO_VELOCITY_MIN_ABS_PX_PER_FRAME", "0.25")

    with pytest.raises(ValueError, match="below AUTO_VELOCITY_MIN_ABS_PX_PER_FRAME"):
        validate_auto_velocity_estimate(0.002, [0.001, 0.002, 0.003], max_shift=90)


def test_auto_velocity_rejects_search_edge_hits(monkeypatch):
    monkeypatch.setenv("AUTO_VELOCITY_MIN_ABS_PX_PER_FRAME", "0.25")
    monkeypatch.setenv("AUTO_VELOCITY_MAX_EDGE_FRACTION", "0.2")

    with pytest.raises(ValueError, match="search edge"):
        validate_auto_velocity_estimate(89.0, [89.0, 88.0, 1.0, 87.0], max_shift=90)


def test_phase_estimate_row_reports_circular_coordinates():
    phase = PhaseEstimate(
        phase_px=25.0,
        frame_index=3.0,
        predicted_phase_px=24.0,
        correction_px=1.0,
        loss=0.5,
        score=0.75,
        method="registration",
    )
    clean = CleanBeltRender(
        image=np.zeros((4, 5)),
        mask=np.ones((4, 5), dtype=bool),
        phase_estimate=phase,
        belt_region=BeltRegion(top=0, left=0, height=4, width=5),
    )
    residual = ResidualImage(
        raw=np.zeros((4, 5)),
        local_noise=np.ones((4, 5)),
        normalized=np.zeros((4, 5)),
        mask=np.ones((4, 5), dtype=bool),
        expected_background=np.zeros((4, 5)),
        clean_render=clean,
    )

    row = phase_estimate_row(
        3,
        DATA / "example.bmp",
        residual,
        period_px=100.0,
    )

    assert row["frame_index"] == 3
    assert row["image"] == "example.bmp"
    assert row["phase_px"] == 25.0
    assert row["phase_fraction"] == 0.25
    assert row["phase_rad"] == pytest.approx(0.5 * np.pi)
    assert row["predicted_phase_px"] == 24.0
    assert row["correction_px"] == 1.0
    assert row["loss"] == 0.5
    assert row["score"] == 0.75
    assert row["method"] == "registration"


def test_build_belt_map_masks_particle_contaminated_observations(tmp_path, monkeypatch):
    monkeypatch.setenv("MAP_SAMPLE_FRAMES", "40")
    monkeypatch.setenv("PROGRESS_INTERVAL_FRAMES", "1000")

    period = 40
    crop_height = 20
    width = 16
    velocity = 4.0
    y = np.arange(period, dtype=float)[:, None]
    x = np.arange(width, dtype=float)[None, :]
    true_belt = np.round(70 + 0.35 * y + 8 * np.sin(2 * np.pi * y / 11) + 3 * np.cos(2 * np.pi * x / 5))
    particle_rows = np.arange(12, 17)
    particle_cols = np.arange(5, 10)
    particle_frames = set(range(10))
    paths = []

    for frame_index in range(40):
        phase = (-velocity * frame_index) % period
        frame = render_belt_view(true_belt, phase_px=phase, height=crop_height)
        rows = ((np.arange(crop_height) + phase) % period).astype(int)
        if frame_index in particle_frames:
            for image_y, belt_y in enumerate(rows):
                if belt_y in particle_rows:
                    frame[image_y, particle_cols] += 80
        path = tmp_path / f"frame_{frame_index:03d}.bmp"
        Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8)).save(path)
        paths.append(path)

    unmasked, _reference_phase, _map_height = build_belt_map(
        paths,
        (0, 0, crop_height, width),
        velocity,
        period,
        mask_iterations=0,
    )
    masked, _reference_phase, _map_height = build_belt_map(
        paths,
        (0, 0, crop_height, width),
        velocity,
        period,
        mask_iterations=1,
        mask_threshold=3.0,
        mask_margin_px=1,
        mask_min_area_px=2,
    )

    patch = np.ix_(particle_rows, particle_cols)
    unmasked_error = np.mean(np.abs(unmasked[patch] - true_belt[patch]))
    masked_error = np.mean(np.abs(masked[patch] - true_belt[patch]))

    assert masked_error < 0.5 * unmasked_error
