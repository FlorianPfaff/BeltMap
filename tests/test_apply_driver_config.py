import pytest
import numpy as np
from PIL import Image

from beltmap import (
    BeltMotionModel,
    BeltRegion,
    CleanBeltRender,
    PhaseEstimate,
    PhaseRegistrationConfig,
    ResidualConfig,
    ResidualImage,
    render_belt_view,
)
from beltmap import _driver_runtime as rt
from beltmap._driver_map import (
    build_belt_map,
    map_sampling_strategy_from_env,
)
from beltmap._driver_motion import (
    validate_auto_velocity_estimate,
    validate_auto_velocity_region,
)
from beltmap.driver import (
    apply_static_noise_floor,
    learn_static_residual_noise_map,
    load_phase_estimates,
    load_recurrent_artifact_map,
    phase_estimate_row,
    subtract_static_background,
    validate_reused_phase_estimates,
)


@pytest.mark.parametrize("value", ["nan", "inf", "+inf", "-inf"])
def test_env_float_rejects_non_finite_values(monkeypatch, value):
    monkeypatch.setenv("TEST_FLOAT", value)

    with pytest.raises(ValueError, match="TEST_FLOAT must be finite"):
        rt.env_float("TEST_FLOAT", 1.0)


def test_env_float_rejects_non_finite_defaults(monkeypatch):
    monkeypatch.delenv("TEST_FLOAT", raising=False)

    with pytest.raises(ValueError, match="TEST_FLOAT must be finite"):
        rt.env_float("TEST_FLOAT", float("nan"))


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


def test_map_sampling_strategy_env_prefers_canonical_alias(monkeypatch):
    monkeypatch.setenv("MAP_SAMPLE_STRATEGY", "uniform")
    monkeypatch.setenv("MAP_SAMPLING_STRATEGY", "adaptive_phase_coverage")

    assert map_sampling_strategy_from_env() == "adaptive_phase_coverage"


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
        rt.DATA / "example.bmp",
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


def test_load_phase_estimates_for_reuse_mode(tmp_path):
    path = tmp_path / "phase_estimates.csv"
    path.write_text(
        "\n".join(
            [
                "frame_index,image,phase_px,phase_fraction,phase_rad,predicted_phase_px,correction_px,loss,score,method",
                "0,frame0.bmp,1.5,0.15,0.94,1.0,0.5,0.2,0.8,registration",
                "1,frame1.bmp,9.5,0.95,5.97,10.0,-0.5,,,motion_model",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    estimates = load_phase_estimates(path)

    assert estimates[0].phase_px == 1.5
    assert estimates[0].loss == 0.2
    assert estimates[0].score == 0.8
    assert estimates[1].correction_px == -0.5
    assert estimates[1].loss is None
    assert estimates[1].score is None


def test_load_phase_estimates_validates_reused_image_sequence(tmp_path):
    path = tmp_path / "phase_estimates.csv"
    path.write_text(
        "\n".join(
            [
                "frame_index,image,phase_px,phase_fraction,phase_rad,predicted_phase_px,correction_px,loss,score,method",
                "0,frames/frame0.bmp,1.5,0.15,0.94,1.0,0.5,0.2,0.8,registration",
                "1,frames/frame1.bmp,9.5,0.95,5.97,10.0,-0.5,,,motion_model",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    estimates = load_phase_estimates(
        path,
        expected_image_paths=[
            tmp_path / "frames" / "frame0.bmp",
            tmp_path / "frames" / "frame1.bmp",
        ],
        data_dir=tmp_path,
    )

    assert estimates[0].phase_px == 1.5
    assert estimates[1].phase_px == 9.5


def test_load_phase_estimates_rejects_reordered_or_stale_images(tmp_path):
    path = tmp_path / "phase_estimates.csv"
    path.write_text(
        "\n".join(
            [
                "frame_index,image,phase_px,phase_fraction,phase_rad,predicted_phase_px,correction_px,loss,score,method",
                "0,other_sequence/frame0.bmp,1.5,0.15,0.94,1.0,0.5,0.2,0.8,registration",
                "1,frames/frame1.bmp,9.5,0.95,5.97,10.0,-0.5,,,motion_model",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="image column does not match"):
        load_phase_estimates(
            path,
            expected_image_paths=[
                tmp_path / "frames" / "frame0.bmp",
                tmp_path / "frames" / "frame1.bmp",
            ],
            data_dir=tmp_path,
        )


def test_validate_reused_phase_estimates_reports_missing_frames():
    estimates = {
        0: PhaseEstimate(
            phase_px=0.0,
            frame_index=0.0,
            predicted_phase_px=0.0,
        ),
        2: PhaseEstimate(
            phase_px=2.0,
            frame_index=2.0,
            predicted_phase_px=2.0,
        ),
    }

    with pytest.raises(ValueError, match="missing 1 selected frames"):
        validate_reused_phase_estimates(estimates, frame_count=3)


def test_load_recurrent_artifact_map_for_reuse_mode(tmp_path):
    path = tmp_path / "recurrent_artifact_map.npy"
    np.save(path, np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8))

    artifact_map = load_recurrent_artifact_map(path, map_shape=(2, 3))

    assert artifact_map.dtype == bool
    np.testing.assert_array_equal(
        artifact_map,
        [[False, True, False], [True, False, True]],
    )


def test_load_recurrent_artifact_map_rejects_shape_mismatch(tmp_path):
    path = tmp_path / "recurrent_artifact_map.npy"
    np.save(path, np.zeros((2, 3), dtype=bool))

    with pytest.raises(ValueError, match="shape does not match"):
        load_recurrent_artifact_map(path, map_shape=(3, 2))


def test_static_noise_floor_renormalizes_residual_without_changing_raw_values():
    residual = ResidualImage(
        raw=np.array([[2.0, 8.0], [3.0, np.nan]]),
        local_noise=np.array([[1.0, 2.0], [5.0, 1.0]]),
        normalized=np.array([[2.0, 4.0], [0.6, np.nan]]),
        mask=np.array([[True, True], [True, False]]),
        expected_background=np.zeros((2, 2)),
    )

    adjusted = apply_static_noise_floor(
        residual,
        np.array([[4.0, 1.0], [np.nan, 2.0]]),
    )

    assert adjusted.raw is residual.raw
    np.testing.assert_array_equal(adjusted.mask, residual.mask)
    np.testing.assert_allclose(adjusted.local_noise, [[4.0, 2.0], [5.0, 2.0]])
    np.testing.assert_allclose(
        adjusted.normalized,
        [[0.5, 4.0], [0.6, np.nan]],
        equal_nan=True,
    )


def test_static_noise_floor_marks_pixels_without_valid_normalization_invalid():
    residual = ResidualImage(
        raw=np.array([[4.0, 6.0]]),
        local_noise=np.array([[np.nan, 2.0]]),
        normalized=np.array([[np.nan, 3.0]]),
        mask=np.array([[True, True]]),
        expected_background=np.zeros((1, 2)),
    )

    adjusted = apply_static_noise_floor(
        residual,
        np.array([[np.nan, 1.0]]),
    )

    np.testing.assert_array_equal(adjusted.mask, [[False, True]])
    assert np.isnan(adjusted.normalized[0, 0])
    assert adjusted.normalized[0, 1] == pytest.approx(3.0)


def test_static_background_correction_marks_invalid_raw_pixels_invalid():
    residual = ResidualImage(
        raw=np.array(
            [
                [2.0, np.nan],
                [3.0, 4.0],
            ]
        ),
        local_noise=np.ones((2, 2)),
        normalized=np.array(
            [
                [2.0, np.nan],
                [3.0, 4.0],
            ]
        ),
        mask=np.ones((2, 2), dtype=bool),
        expected_background=np.zeros((2, 2)),
    )

    adjusted = subtract_static_background(
        residual,
        np.zeros((2, 2)),
        residual_config=ResidualConfig(noise_radius_px=0),
    )

    np.testing.assert_array_equal(
        adjusted.mask,
        [
            [True, False],
            [True, True],
        ],
    )
    assert np.isnan(adjusted.raw[0, 1])
    assert np.isnan(adjusted.normalized[0, 1])


def test_learn_static_residual_noise_map_estimates_per_pixel_mad(tmp_path, monkeypatch):
    monkeypatch.setenv("PROGRESS_INTERVAL_FRAMES", "1000")
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    monkeypatch.setattr(rt, "OUT", output_dir)

    belt_map = np.full((8, 4), 50.0, dtype=np.float32)
    paths = []
    variable_residuals = [0, 10, 20, 30, 40]
    for frame_index, value in enumerate(variable_residuals):
        frame = np.full((4, 4), 50, dtype=np.uint8)
        frame[1, 2] = 50 + value
        path = tmp_path / f"frame_{frame_index:03d}.bmp"
        Image.fromarray(frame).save(path)
        paths.append(path)

    phases = {
        frame_index: PhaseEstimate(
            phase_px=0.0,
            frame_index=float(frame_index),
            predicted_phase_px=0.0,
        )
        for frame_index in range(len(paths))
    }

    static_noise = learn_static_residual_noise_map(
        paths=paths,
        belt_map=belt_map,
        motion_model=BeltMotionModel(
            image_velocity_px_per_frame=0.0,
            period_px=float(belt_map.shape[0]),
        ),
        region=(0, 0, 4, 4),
        phase_estimates=phases,
        registration_config=PhaseRegistrationConfig(),
        residual_config=ResidualConfig(),
        sample_frames=len(paths),
        min_scale=0.0,
        chunk_rows=2,
    )

    assert static_noise.shape == (4, 4)
    assert static_noise[1, 2] == pytest.approx(14.826, rel=1e-3)
    assert np.median(np.delete(static_noise.ravel(), 1 * 4 + 2)) == pytest.approx(0.0)
    assert not any(output_dir.glob("static_noise_*"))


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
