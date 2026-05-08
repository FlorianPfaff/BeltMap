import numpy as np
import pytest

from beltmap._driver_map import detect_map_particle_mask, validate_map_particle_mask_mode
from beltmap.residual import ResidualImage


def make_residual(normalized: np.ndarray) -> ResidualImage:
    normalized = np.asarray(normalized, dtype=np.float64)
    return ResidualImage(
        raw=normalized.copy(),
        local_noise=np.ones_like(normalized),
        normalized=normalized,
        mask=np.ones(normalized.shape, dtype=bool),
        expected_background=np.zeros_like(normalized),
    )


def test_hysteresis_abs_map_mask_grows_from_bright_seed_into_dark_particle_body():
    z = np.zeros((25, 25), dtype=np.float64)
    z[8:17, 8:17] = -2.0
    z[11:14, 11:14] = 5.0
    residual = make_residual(z)

    positive = detect_map_particle_mask(
        residual,
        mode="positive",
        threshold=4.0,
        grow_threshold=1.5,
        dilation_px=0,
        margin_px=0,
        min_area_px=1,
    )
    hysteresis = detect_map_particle_mask(
        residual,
        mode="hysteresis_abs",
        threshold=4.0,
        grow_threshold=1.5,
        dilation_px=0,
        margin_px=0,
        min_area_px=1,
    )

    assert positive.sum() == 9
    assert hysteresis.sum() == 81
    assert hysteresis[8, 8]
    assert hysteresis[16, 16]


def test_hysteresis_abs_map_mask_can_dilate_grown_components():
    z = np.zeros((15, 15), dtype=np.float64)
    z[7, 7] = 5.0
    residual = make_residual(z)

    undilated = detect_map_particle_mask(
        residual,
        mode="hysteresis_abs",
        threshold=4.0,
        grow_threshold=1.5,
        dilation_px=0,
        margin_px=0,
        min_area_px=1,
    )
    dilated = detect_map_particle_mask(
        residual,
        mode="hysteresis_abs",
        threshold=4.0,
        grow_threshold=1.5,
        dilation_px=2,
        margin_px=0,
        min_area_px=1,
    )

    assert undilated.sum() == 1
    assert dilated.sum() > undilated.sum()
    assert dilated[7, 9]


def test_hysteresis_abs_map_mask_fills_small_internal_holes():
    z = np.zeros((9, 9), dtype=np.float64)
    z[2:7, 2:7] = 5.0
    z[4, 4] = 0.0
    residual = make_residual(z)

    unfilled = detect_map_particle_mask(
        residual,
        mode="hysteresis_abs",
        threshold=4.0,
        grow_threshold=1.5,
        dilation_px=0,
        margin_px=0,
        min_area_px=1,
    )
    filled = detect_map_particle_mask(
        residual,
        mode="hysteresis_abs",
        threshold=4.0,
        grow_threshold=1.5,
        dilation_px=0,
        margin_px=0,
        min_area_px=2,
    )

    assert not unfilled[4, 4]
    assert filled[2:7, 2:7].all()


def test_absolute_map_mask_catches_dark_components_that_positive_mode_misses():
    z = np.zeros((12, 12), dtype=np.float64)
    z[4:8, 4:8] = -5.0
    residual = make_residual(z)

    positive = detect_map_particle_mask(
        residual,
        mode="positive",
        threshold=4.0,
        grow_threshold=1.5,
        dilation_px=0,
        margin_px=0,
        min_area_px=1,
    )
    absolute = detect_map_particle_mask(
        residual,
        mode="absolute",
        threshold=4.0,
        grow_threshold=1.5,
        dilation_px=0,
        margin_px=0,
        min_area_px=1,
    )

    assert not positive.any()
    assert absolute[4:8, 4:8].all()


def test_invalid_map_particle_mask_mode_raises_useful_error():
    with pytest.raises(ValueError, match="MAP_PARTICLE_MASK_MODE"):
        validate_map_particle_mask_mode("unknown")
