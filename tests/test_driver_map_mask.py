import numpy as np
import pytest

from beltmap._driver_map import (
    detect_map_particle_mask,
    map_geometry,
    validate_map_particle_mask_mode,
)
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


@pytest.mark.parametrize(
    ("mode", "signal"),
    [("positive", 5.0), ("negative", -5.0), ("absolute", -5.0)],
)
def test_non_hysteresis_map_masks_can_dilate_components(mode, signal):
    z = np.zeros((15, 15), dtype=np.float64)
    z[7, 7] = signal
    residual = make_residual(z)

    undilated = detect_map_particle_mask(
        residual,
        mode=mode,
        threshold=4.0,
        grow_threshold=1.5,
        dilation_px=0,
        margin_px=0,
        min_area_px=1,
    )
    dilated = detect_map_particle_mask(
        residual,
        mode=mode,
        threshold=4.0,
        grow_threshold=1.5,
        dilation_px=2,
        margin_px=0,
        min_area_px=1,
    )

    assert undilated.sum() == 1
    assert dilated.sum() > undilated.sum()
    assert dilated[7, 9]


def test_hysteresis_abs_map_mask_fills_small_internal_holes_with_optional_morphology():
    pytest.importorskip("scipy.ndimage")
    z = np.zeros((9, 9), dtype=np.float64)
    z[2:7, 2:7] = 5.0
    z[4, 4] = 0.0
    residual = make_residual(z)

    filled = detect_map_particle_mask(
        residual,
        mode="hysteresis_abs",
        threshold=4.0,
        grow_threshold=1.5,
        dilation_px=0,
        margin_px=0,
        min_area_px=2,
    )

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


def test_negative_map_mask_catches_dark_components_without_absolute_mode():
    z = np.zeros((12, 12), dtype=np.float64)
    z[4:8, 4:8] = -5.0
    residual = make_residual(z)

    negative = detect_map_particle_mask(
        residual,
        mode="negative",
        threshold=4.0,
        grow_threshold=1.5,
        dilation_px=0,
        margin_px=0,
        min_area_px=1,
    )

    assert negative[4:8, 4:8].all()
    assert not negative[0, 0]


def test_invalid_map_particle_mask_mode_raises_useful_error():
    with pytest.raises(ValueError, match="MAP_PARTICLE_MASK_MODE"):
        validate_map_particle_mask_mode("unknown")


def test_hysteresis_abs_rejects_grow_threshold_above_seed_threshold():
    residual = make_residual(np.zeros((8, 8), dtype=np.float64))

    with pytest.raises(ValueError, match="grow_threshold"):
        detect_map_particle_mask(
            residual,
            mode="hysteresis_abs",
            threshold=4.0,
            grow_threshold=5.0,
            dilation_px=0,
            margin_px=0,
            min_area_px=1,
        )


def test_map_geometry_rejects_non_positive_supplied_period():
    with pytest.raises(ValueError, match="supplied_period"):
        map_geometry(
            frame_count=10,
            crop_height=4,
            velocity=1.0,
            supplied_period=-5,
        )
