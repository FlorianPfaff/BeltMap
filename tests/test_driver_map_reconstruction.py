import numpy as np
import pytest
from PIL import Image

from beltmap import _driver_runtime as rt
from beltmap._driver_map import (
    _accumulate_frame_linear,
    _accumulate_frame_nearest,
    accumulate_belt_map,
    estimate_local_illumination_field,
    validate_map_aggregation,
    validate_map_trim_fraction,
)


def _write_gray(path, value: int) -> None:
    Image.fromarray(np.full((1, 1), value, dtype=np.uint8)).save(path)


def _write_gray_array(path, values: np.ndarray) -> None:
    Image.fromarray(np.asarray(values, dtype=np.uint8)).save(path)


def test_trimmed_map_reconstruction_rejects_single_bright_outlier(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "OUT", tmp_path / "out")
    monkeypatch.setenv("PROGRESS_INTERVAL_FRAMES", "1000")

    paths = []
    for index, value in enumerate([10, 10, 10, 250]):
        path = tmp_path / f"frame_{index:03d}.png"
        _write_gray(path, value)
        paths.append(path)

    common_kwargs = dict(
        paths=paths,
        samples=[0, 1, 2, 3],
        region=(0, 0, 1, 1),
        velocity=0.0,
        reference_phase=0.0,
        model_period=1.0,
        map_height=1,
        previous_belt_map=None,
        mask_threshold=5.0,
        mask_mode="positive",
        mask_grow_threshold=2.0,
        mask_dilation_px=0,
        mask_margin_px=0,
        mask_min_area_px=1,
        pass_label="test",
    )

    mean_map, mean_coverage = accumulate_belt_map(
        **common_kwargs,
        map_trim_fraction=0.0,
    )
    trimmed_map, trimmed_coverage = accumulate_belt_map(
        **common_kwargs,
        map_trim_fraction=0.25,
    )

    assert mean_coverage["contributed_pixels"] == 4
    assert trimmed_coverage == mean_coverage
    assert mean_map[0, 0] == pytest.approx(70.0)
    assert trimmed_map[0, 0] == pytest.approx(10.0)


def test_winsorized_map_reconstruction_clips_single_bright_outlier(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "OUT", tmp_path / "out")
    monkeypatch.setenv("PROGRESS_INTERVAL_FRAMES", "1000")

    paths = []
    for index, value in enumerate([10, 10, 12, 250]):
        path = tmp_path / f"frame_{index:03d}.png"
        _write_gray(path, value)
        paths.append(path)

    common_kwargs = dict(
        paths=paths,
        samples=[0, 1, 2, 3],
        region=(0, 0, 1, 1),
        velocity=0.0,
        reference_phase=0.0,
        model_period=1.0,
        map_height=1,
        previous_belt_map=None,
        mask_threshold=5.0,
        mask_mode="positive",
        mask_grow_threshold=2.0,
        mask_dilation_px=0,
        mask_margin_px=0,
        mask_min_area_px=1,
        pass_label="test",
    )

    mean_map, mean_coverage = accumulate_belt_map(
        **common_kwargs,
        map_trim_fraction=0.0,
    )
    winsorized_map, winsorized_coverage = accumulate_belt_map(
        **common_kwargs,
        map_trim_fraction=0.25,
        robust_pixel_estimator="winsorized_mean",
    )

    assert mean_coverage["contributed_pixels"] == 4
    assert winsorized_coverage == mean_coverage
    assert mean_map[0, 0] == pytest.approx(70.5)
    # With one sample trimmed from each tail, the 250 outlier is clipped to 12
    # and the low tail remains at 10: mean([10, 10, 12, 12]) = 11.
    assert winsorized_map[0, 0] == pytest.approx(11.0)


def test_frame_median_offset_correction_aligns_additive_illumination(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "OUT", tmp_path / "out")
    monkeypatch.setenv("PROGRESS_INTERVAL_FRAMES", "1000")

    base = np.asarray([[50, 60], [70, 80]], dtype=np.uint8)
    paths = []
    for index, offset in enumerate([-10, 0, 20]):
        path = tmp_path / f"frame_{index:03d}.png"
        _write_gray_array(path, base.astype(np.int16) + offset)
        paths.append(path)

    common_kwargs = dict(
        paths=paths,
        samples=[0, 1, 2],
        region=(0, 0, 2, 2),
        velocity=0.0,
        reference_phase=0.0,
        model_period=2.0,
        map_height=2,
        previous_belt_map=None,
        mask_threshold=5.0,
        mask_mode="positive",
        mask_grow_threshold=2.0,
        mask_dilation_px=0,
        mask_margin_px=0,
        mask_min_area_px=1,
        pass_label="test",
    )

    mean_map, _mean_coverage = accumulate_belt_map(**common_kwargs)
    corrected_map, corrected_coverage = accumulate_belt_map(
        **common_kwargs,
        frame_median_offset_correction=True,
    )

    np.testing.assert_allclose(corrected_map, base.astype(np.float32))
    assert corrected_coverage["contributed_pixels"] == 12
    assert np.mean(np.abs(corrected_map - base)) < np.mean(np.abs(mean_map - base))


def test_frame_median_offset_correction_uses_residual_median_with_reference_map(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(rt, "OUT", tmp_path / "out")
    monkeypatch.setenv("PROGRESS_INTERVAL_FRAMES", "1000")

    base = np.asarray([[50, 65], [70, 90]], dtype=np.uint8)
    paths = []
    for index, offset in enumerate([-10, 5, 25]):
        path = tmp_path / f"frame_{index:03d}.png"
        _write_gray_array(path, base.astype(np.int16) + offset)
        paths.append(path)

    corrected_map, coverage = accumulate_belt_map(
        paths=paths,
        samples=[0, 1, 2],
        region=(0, 0, 2, 2),
        velocity=0.0,
        reference_phase=0.0,
        model_period=2.0,
        map_height=2,
        previous_belt_map=base.astype(np.float32),
        mask_threshold=5.0,
        mask_mode="positive",
        mask_grow_threshold=2.0,
        mask_dilation_px=0,
        mask_margin_px=0,
        mask_min_area_px=1,
        pass_label="test",
        frame_median_offset_correction=True,
    )

    np.testing.assert_allclose(corrected_map, base.astype(np.float32))
    assert coverage["masked_pixels"] == 0
    assert coverage["contributed_pixels"] == 12


def test_local_illumination_field_interpolates_tile_medians():
    residual = np.tile(np.asarray([-20.0, -10.0, 10.0, 20.0]), (4, 1))
    valid = np.ones(residual.shape, dtype=bool)

    field = estimate_local_illumination_field(residual, valid, tile_px=2)

    assert field.shape == residual.shape
    assert field[0, 0] == pytest.approx(-15.0)
    assert field[0, -1] == pytest.approx(15.0)
    assert np.all(np.diff(field[0]) >= 0)


def test_local_illumination_correction_reduces_spatial_additive_field(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(rt, "OUT", tmp_path / "out")
    monkeypatch.setenv("PROGRESS_INTERVAL_FRAMES", "1000")

    base = np.full((4, 4), 100.0, dtype=np.float32)
    local_field = np.tile(np.asarray([-20.0, -10.0, 10.0, 20.0]), (4, 1))
    frame = base + local_field
    path = tmp_path / "frame_000.png"
    _write_gray_array(path, frame)

    common_kwargs = dict(
        paths=[path],
        samples=[0],
        region=(0, 0, 4, 4),
        velocity=0.0,
        reference_phase=0.0,
        model_period=4.0,
        map_height=4,
        previous_belt_map=base,
        mask_threshold=5.0,
        mask_mode="positive",
        mask_grow_threshold=2.0,
        mask_dilation_px=0,
        mask_margin_px=0,
        mask_min_area_px=1,
        pass_label="test",
    )

    uncorrected_map, _uncorrected_coverage = accumulate_belt_map(**common_kwargs)
    corrected_map, corrected_coverage = accumulate_belt_map(
        **common_kwargs,
        local_illumination_correction=True,
        local_illumination_tile_px=2,
    )

    assert corrected_coverage["contributed_pixels"] == 16
    assert np.mean(np.abs(corrected_map - base)) < np.mean(
        np.abs(uncorrected_map - base)
    )


def test_trimmed_map_reconstruction_fails_before_large_memory_allocation(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "OUT", tmp_path / "out")
    monkeypatch.setenv("PROGRESS_INTERVAL_FRAMES", "1000")
    monkeypatch.setenv("MAP_RECONSTRUCTION_TRIM_MAX_MEMORY_GB", "0.000000001")

    paths = []
    for index, value in enumerate([10, 250]):
        path = tmp_path / f"frame_{index:03d}.png"
        Image.fromarray(np.full((8, 8), value, dtype=np.uint8)).save(path)
        paths.append(path)

    with pytest.raises(MemoryError, match="robust per-pixel belt-map reconstruction"):
        accumulate_belt_map(
            paths=paths,
            samples=[0, 1],
            region=(0, 0, 8, 8),
            velocity=0.0,
            reference_phase=0.0,
            model_period=8.0,
            map_height=8,
            previous_belt_map=None,
            mask_threshold=5.0,
            mask_mode="positive",
            mask_grow_threshold=2.0,
            mask_dilation_px=0,
            mask_margin_px=0,
            mask_min_area_px=1,
            pass_label="test",
            map_trim_fraction=0.25,
        )


def test_huber_map_reconstruction_downweights_single_bright_outlier(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "OUT", tmp_path / "out")
    monkeypatch.setenv("PROGRESS_INTERVAL_FRAMES", "1000")

    paths = []
    for index in range(20):
        values = np.full((3, 3), 10, dtype=np.uint8)
        if index == 19:
            values[1, 1] = 250
        path = tmp_path / f"frame_{index:03d}.png"
        _write_gray_array(path, values)
        paths.append(path)

    common_kwargs = dict(
        paths=paths,
        samples=list(range(20)),
        region=(0, 0, 3, 3),
        velocity=0.0,
        reference_phase=0.0,
        model_period=3.0,
        map_height=3,
        mask_threshold=5.0,
        mask_mode="positive",
        mask_grow_threshold=2.0,
        mask_dilation_px=0,
        mask_margin_px=0,
        mask_min_area_px=1,
    )

    mean_map, _mean_coverage = accumulate_belt_map(
        **common_kwargs,
        previous_belt_map=None,
        pass_label="mean",
    )
    huber_map, huber_coverage = accumulate_belt_map(
        **common_kwargs,
        previous_belt_map=None,
        robust_reference_belt_map=mean_map,
        robust_huber_delta=3.0,
        robust_min_scale=1.0,
        pass_label="huber",
    )

    assert mean_map[1, 1] == pytest.approx(22.0)
    assert huber_coverage["contributed_pixels"] == 20 * 9
    assert huber_map[1, 1] < 12.0


def test_non_fractional_map_accumulation_uses_nearest_row_assignment():
    frame = np.asarray([[10.0], [100.0]], dtype=np.float64)
    valid = np.ones(frame.shape, dtype=bool)
    sums_linear = np.zeros((4, 1), dtype=np.float64)
    weights_linear = np.zeros_like(sums_linear)
    sums_nearest = np.zeros((4, 1), dtype=np.float64)
    weights_nearest = np.zeros_like(sums_nearest)

    common_kwargs = dict(
        frame=frame,
        valid=valid,
        phase=0.4,
        map_height=4,
        model_period=None,
    )
    assert _accumulate_frame_linear(
        sums=sums_linear,
        weights=weights_linear,
        **common_kwargs,
    ) == 2
    assert _accumulate_frame_nearest(
        sums=sums_nearest,
        weights=weights_nearest,
        **common_kwargs,
    ) == 2

    np.testing.assert_allclose(weights_linear[:, 0], [0.6, 1.0, 0.4, 0.0])
    np.testing.assert_allclose(sums_linear[:, 0], [6.0, 64.0, 40.0, 0.0])
    np.testing.assert_allclose(weights_nearest[:, 0], [1.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(sums_nearest[:, 0], [10.0, 100.0, 0.0, 0.0])


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("mean", "mean"),
        ("huber", "huber"),
        ("trimmed", "trimmed_mean"),
        ("winsor", "winsorized_mean"),
        ("winsorised-mean", "winsorized_mean"),
    ],
)
def test_validate_map_aggregation_accepts_robust_estimators(value, expected):
    assert validate_map_aggregation(value) == expected


@pytest.mark.parametrize("trim_fraction", [0.0, 0.1, 0.49])
def test_validate_map_trim_fraction_accepts_valid_fractions(trim_fraction):
    assert validate_map_trim_fraction(trim_fraction) == trim_fraction


@pytest.mark.parametrize("trim_fraction", [-0.01, 0.5, 1.0, np.nan])
def test_validate_map_trim_fraction_rejects_invalid_fractions(trim_fraction):
    with pytest.raises(ValueError, match="MAP_RECONSTRUCTION_TRIM_FRACTION"):
        validate_map_trim_fraction(trim_fraction)
