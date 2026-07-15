from __future__ import annotations

import numpy as np

import beltmap  # noqa: F401 - imports compatibility patches
import beltmap.cross_map_agreement as agreement
from beltmap.residual import ResidualImage
from beltmap.tracking import ParticleDetection


def _detection(label: int) -> ParticleDetection:
    return ParticleDetection(
        frame_index=0.0,
        label=label,
        y=1.0,
        x=1.0,
        area_px=4,
        bbox_top=0,
        bbox_left=0,
        bbox_bottom=2,
        bbox_right=2,
        mean_signal=1.0,
        peak_signal=2.0,
    )


def _balanced_residual() -> ResidualImage:
    raw = np.zeros((4, 4), dtype=np.float64)
    raw[0, 0] = 2.0
    raw[0, 1] = -2.0
    local_noise = np.ones_like(raw)
    return ResidualImage(
        raw=raw,
        local_noise=local_noise,
        normalized=raw / local_noise,
        mask=np.ones_like(raw, dtype=bool),
        expected_background=np.zeros_like(raw),
    )


def test_zero_mean_residual_region_has_no_raw_sign():
    helper = agreement.detection_raw_sign

    assert getattr(helper, "_beltmap_cross_map_zero_sign_patched", False)
    assert helper(_detection(1), _balanced_residual()) is None


def test_signless_regions_do_not_pass_required_sign_consistency():
    scores = agreement.score_cross_map_agreement(
        [_detection(1)],
        [[_detection(2)]],
        primary_residual=_balanced_residual(),
        confirming_residuals=[_balanced_residual()],
        config=agreement.CrossMapAgreementConfig(
            max_centroid_distance_px=1.0,
            min_bbox_iou=0.5,
            min_peak_ratio=0.5,
            require_sign_consistency=True,
            min_confirming_maps=1,
        ),
    )

    assert not scores[0].accepted
    assert scores[0].confirming_maps == 0
    assert scores[0].matches[0].sign_consistent is None
    assert not scores[0].matches[0].accepted
