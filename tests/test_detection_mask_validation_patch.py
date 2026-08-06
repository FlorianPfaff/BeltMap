import importlib

import numpy as np
import pytest

import beltmap
import beltmap.advanced_quality as advanced_quality
import beltmap.detection as detection
import beltmap.detection_mask_validation_patch as detection_mask_validation_patch
import beltmap.tracking as tracking
from beltmap import ResidualImage


AMBIGUOUS_MASKS = [
    np.asarray([[1.0, np.nan]], dtype=np.float64),
    np.asarray([[1.0, np.inf]], dtype=np.float64),
    np.asarray([[0, -1]], dtype=np.int64),
    np.asarray([[0, 2]], dtype=np.int64),
    np.asarray([["0", "1"]], dtype=object),
]


@pytest.mark.parametrize("mask", AMBIGUOUS_MASKS)
def test_detect_particles_rejects_ambiguous_optional_masks(mask):
    residual = np.asarray([[8.0, 8.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="boolean or binary 0/1 array"):
        beltmap.detect_particles_from_residual(
            residual,
            threshold=5.0,
            mask=mask,
        )


def test_detection_signal_rejects_nonfinite_residual_image_mask():
    normalized = np.asarray([[8.0, 8.0]], dtype=np.float64)
    residual = ResidualImage(
        raw=normalized,
        local_noise=np.ones_like(normalized),
        normalized=normalized,
        mask=np.asarray([[1.0, np.nan]], dtype=np.float64),
        expected_background=np.zeros_like(normalized),
    )

    with pytest.raises(ValueError, match="ResidualImage mask.*boolean or binary"):
        detection.detection_signal_from_residual(residual)


def test_detect_particles_preserves_legacy_binary_numeric_masks():
    residual = np.full((2, 2), 8.0, dtype=np.float64)
    mask = np.asarray([[1, 0], [0, 1]], dtype=np.uint8)

    detected = beltmap.detect_particles_from_residual(
        residual,
        threshold=5.0,
        mask=mask,
    )

    np.testing.assert_array_equal(detected, mask.astype(bool))


@pytest.mark.parametrize("mask", AMBIGUOUS_MASKS)
def test_extract_particle_detections_rejects_ambiguous_particle_masks(mask):
    with pytest.raises(ValueError, match="particle_mask.*boolean or binary 0/1 array"):
        beltmap.extract_particle_detections(mask)


def test_extract_particle_detections_preserves_binary_numeric_masks():
    mask = np.asarray([[0, 1], [0, 1]], dtype=np.uint8)

    detections = beltmap.extract_particle_detections(mask)

    assert len(detections) == 1
    assert detections[0].area_px == 2
    np.testing.assert_allclose([detections[0].y, detections[0].x], [0.5, 1.0])


@pytest.mark.parametrize("mask", AMBIGUOUS_MASKS)
def test_robust_gain_offset_rejects_ambiguous_masks(mask):
    expected = np.arange(4, dtype=np.float64).reshape(2, 2)
    observed = 2.0 * expected + 1.0

    with pytest.raises(ValueError, match="boolean or binary 0/1 array"):
        advanced_quality.robust_gain_offset(
            observed,
            expected,
            mask=mask,
            trim_fraction=0.0,
            min_pixels=2,
        )


def test_robust_gain_offset_preserves_binary_numeric_masks():
    expected = np.arange(4, dtype=np.float64).reshape(2, 2)
    observed = 2.0 * expected + 1.0
    mask = np.asarray([[1, 1], [1, 0]], dtype=np.uint8)

    fit = advanced_quality.robust_gain_offset(
        observed,
        expected,
        mask=mask,
        trim_fraction=0.0,
        min_pixels=3,
    )

    np.testing.assert_allclose([fit.gain, fit.offset], [2.0, 1.0], rtol=1e-12)


@pytest.mark.parametrize("mask", AMBIGUOUS_MASKS)
def test_integer_shift_rejects_ambiguous_masks(mask):
    image = np.arange(4, dtype=np.float64).reshape(2, 2)

    with pytest.raises(ValueError, match="boolean or binary 0/1 array"):
        advanced_quality.estimate_integer_xy_shift(
            image,
            image,
            mask=mask,
            max_shift_y_px=0,
            max_shift_x_px=0,
        )


def test_integer_shift_preserves_binary_numeric_masks():
    image = np.arange(9, dtype=np.float64).reshape(3, 3)
    mask = np.asarray(
        [
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    estimate = advanced_quality.estimate_integer_xy_shift(
        image,
        image,
        mask=mask,
        max_shift_y_px=0,
        max_shift_x_px=0,
    )

    assert estimate.shift_y_px == 0
    assert estimate.shift_x_px == 0
    assert estimate.loss == 0.0


def test_detection_mask_patch_reload_is_idempotent():
    importlib.reload(detection_mask_validation_patch)
    first_detector = detection.detect_particles_from_residual
    first_extractor = tracking.extract_particle_detections
    first_gain_offset = advanced_quality.robust_gain_offset
    first_shift = advanced_quality.estimate_integer_xy_shift

    importlib.reload(detection_mask_validation_patch)
    second_detector = detection.detect_particles_from_residual
    second_extractor = tracking.extract_particle_detections
    second_gain_offset = advanced_quality.robust_gain_offset
    second_shift = advanced_quality.estimate_integer_xy_shift

    assert getattr(second_detector, "_beltmap_detection_mask_validation_patched", False)
    assert getattr(
        second_detector,
        "_beltmap_original_detect_particles_from_residual",
    ) is getattr(
        first_detector,
        "_beltmap_original_detect_particles_from_residual",
    )
    assert getattr(second_extractor, "_beltmap_detection_mask_validation_patched", False)
    assert getattr(
        second_extractor,
        "_beltmap_original_extract_particle_detections",
    ) is getattr(
        first_extractor,
        "_beltmap_original_extract_particle_detections",
    )
    assert getattr(second_gain_offset, "_beltmap_detection_mask_validation_patched", False)
    assert getattr(
        second_gain_offset,
        "_beltmap_original_robust_gain_offset",
    ) is getattr(
        first_gain_offset,
        "_beltmap_original_robust_gain_offset",
    )
    assert getattr(second_shift, "_beltmap_detection_mask_validation_patched", False)
    assert getattr(
        second_shift,
        "_beltmap_original_estimate_integer_xy_shift_before_mask_validation",
    ) is getattr(
        first_shift,
        "_beltmap_original_estimate_integer_xy_shift_before_mask_validation",
    )
    assert beltmap.detect_particles_from_residual is second_detector
    assert beltmap.extract_particle_detections is second_extractor

    with pytest.raises(ValueError, match="boolean or binary 0/1 array"):
        second_detector(
            np.asarray([[8.0]], dtype=np.float64),
            threshold=5.0,
            mask=np.asarray([[np.nan]], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="particle_mask.*boolean or binary"):
        second_extractor(np.asarray([[np.nan]], dtype=np.float64))
    with pytest.raises(ValueError, match="boolean or binary"):
        second_gain_offset(
            np.asarray([[0.0]], dtype=np.float64),
            np.asarray([[0.0]], dtype=np.float64),
            mask=np.asarray([[np.nan]], dtype=np.float64),
            min_pixels=1,
        )
    with pytest.raises(ValueError, match="boolean or binary"):
        second_shift(
            np.asarray([[0.0]], dtype=np.float64),
            np.asarray([[0.0]], dtype=np.float64),
            mask=np.asarray([[np.nan]], dtype=np.float64),
            max_shift_y_px=0,
            max_shift_x_px=0,
        )
