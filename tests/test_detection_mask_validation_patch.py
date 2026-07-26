import importlib

import numpy as np
import pytest

import beltmap
import beltmap.detection as detection
import beltmap.detection_mask_validation_patch as detection_mask_validation_patch
from beltmap import ResidualImage


@pytest.mark.parametrize(
    "mask",
    [
        np.asarray([[1.0, np.nan]], dtype=np.float64),
        np.asarray([[1.0, np.inf]], dtype=np.float64),
        np.asarray([[1, 2]], dtype=np.int64),
        np.asarray([["1", "0"]], dtype=object),
    ],
)
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


def test_detection_mask_patch_reload_is_idempotent():
    importlib.reload(detection_mask_validation_patch)
    first = detection.detect_particles_from_residual
    importlib.reload(detection_mask_validation_patch)
    second = detection.detect_particles_from_residual

    assert getattr(second, "_beltmap_detection_mask_validation_patched", False)
    assert getattr(
        second,
        "_beltmap_original_detect_particles_from_residual",
    ) is getattr(
        first,
        "_beltmap_original_detect_particles_from_residual",
    )
    assert beltmap.detect_particles_from_residual is second

    with pytest.raises(ValueError, match="boolean or binary 0/1 array"):
        second(
            np.asarray([[8.0]], dtype=np.float64),
            threshold=5.0,
            mask=np.asarray([[np.nan]], dtype=np.float64),
        )
