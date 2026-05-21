import numpy as np
from PIL import Image

from beltmap import PhaseRegistrationConfig, render_belt_view
from beltmap._driver_map import (
    PhaseFeedbackConfig,
    accumulate_belt_map,
    build_belt_map,
    build_belt_map_result,
    select_map_sample_indices,
    smooth_phase_corrections,
)


def test_smooth_phase_corrections_interpolates_and_median_smooths():
    corrections = np.full(7, np.nan)
    corrections[0] = 0.0
    corrections[3] = 3.0
    corrections[6] = 6.0

    smoothed = smooth_phase_corrections(
        frame_count=7,
        correction_by_frame=corrections,
        smoothing_window_frames=1,
    )

    assert np.allclose(smoothed, np.arange(7, dtype=float))


def test_accumulate_belt_map_splats_fractional_phase_linearly(tmp_path, monkeypatch):
    monkeypatch.setenv("PROGRESS_INTERVAL_FRAMES", "1000")

    frame = np.asarray([[10], [110]], dtype=np.uint8)
    path = tmp_path / "frame_000.bmp"
    Image.fromarray(frame).save(path)

    belt_map, coverage = accumulate_belt_map(
        paths=[path],
        samples=[0],
        region=(0, 0, 2, 1),
        velocity=0.0,
        reference_phase=0.25,
        model_period=4.0,
        map_height=4,
        previous_belt_map=None,
        mask_threshold=5.0,
        mask_mode="positive",
        mask_grow_threshold=2.0,
        mask_dilation_px=0,
        mask_margin_px=0,
        mask_min_area_px=1,
        pass_label="test",
    )

    assert coverage["contributed_pixels"] == 2
    assert coverage["observed_pixels"] == 3
    assert np.allclose(belt_map[:, 0], [10.0, 85.0, 110.0, 60.0])


def test_adaptive_map_sampling_returns_unique_phase_coverage_indices():
    samples = select_map_sample_indices(
        frame_count=30,
        sample_count=8,
        velocity=7.0,
        reference_phase=0.0,
        model_period=31.0,
        map_height=31,
        crop_height=5,
        sampling_strategy="adaptive_phase_coverage",
    )

    assert len(samples) == 8
    assert samples == sorted(samples)
    assert len(set(samples)) == len(samples)
    assert all(0 <= index < 30 for index in samples)


def test_phase_feedback_map_refinement_reduces_speed_jitter_blur(tmp_path, monkeypatch):
    monkeypatch.setenv("MAP_SAMPLE_FRAMES", "60")
    monkeypatch.setenv("PROGRESS_INTERVAL_FRAMES", "1000")

    period = 96
    crop_height = 40
    width = 18
    frame_count = 60
    velocity = 2.0

    y = np.arange(period, dtype=np.float64)[:, None]
    x = np.arange(width, dtype=np.float64)[None, :]
    true_belt = np.round(
        110
        + 28 * np.sin(2 * np.pi * y / 9)
        + 14 * np.cos(2 * np.pi * y / 17)
        + 6 * np.cos(2 * np.pi * x / 5)
    )
    true_belt = np.clip(true_belt, 0, 255).astype(np.float64)
    paths = []
    true_corrections = 2.0 * np.sin(2 * np.pi * np.arange(frame_count) / frame_count)

    for frame_index, correction in enumerate(true_corrections):
        model_phase = (-velocity * frame_index) % period
        true_phase = (model_phase + correction) % period
        frame = render_belt_view(true_belt, phase_px=true_phase, height=crop_height)
        path = tmp_path / f"frame_{frame_index:03d}.bmp"
        Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8)).save(path)
        paths.append(path)

    unrefined, _reference_phase, _map_height = build_belt_map(
        paths,
        (0, 0, crop_height, width),
        velocity,
        period,
        mask_iterations=0,
    )
    refined_result = build_belt_map_result(
        paths=paths,
        region=(0, 0, crop_height, width),
        velocity=velocity,
        supplied_period=period,
        mask_iterations=0,
        phase_feedback_config=PhaseFeedbackConfig(
            iterations=2,
            min_score=0.0,
            max_abs_correction_px=3.5,
            smoothing_window_frames=1,
            registration_config=PhaseRegistrationConfig(
                search_radius_px=4.0,
                search_step_px=0.5,
                trim_fraction=0.0,
                highpass_radius_px=0,
            ),
        ),
    )

    refined = refined_result.belt_map
    unrefined_error = float(np.mean(np.abs(unrefined - true_belt)))
    refined_error = float(np.mean(np.abs(refined - true_belt)))

    assert refined_error < unrefined_error
    assert refined_result.phase_by_frame is not None
    assert any(row["used_for_refinement"] for row in refined_result.phase_refinement_rows)
