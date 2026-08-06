import pytest

from beltmap.cross_map_agreement import (
    CrossMapAgreementConfig,
    score_cross_map_agreement,
)
from beltmap.tracking import ParticleDetection


def _detection(*, frame_index: float, label: int = 1) -> ParticleDetection:
    return ParticleDetection(
        frame_index=frame_index,
        label=label,
        y=10.0,
        x=10.0,
        area_px=16,
        bbox_top=8,
        bbox_left=8,
        bbox_bottom=13,
        bbox_right=13,
        mean_signal=6.0,
        peak_signal=12.0,
    )


def _config() -> CrossMapAgreementConfig:
    return CrossMapAgreementConfig(
        max_centroid_distance_px=1.0,
        min_bbox_iou=0.5,
        min_peak_ratio=0.5,
        require_sign_consistency=False,
        min_confirming_maps=1,
    )


def test_cross_map_agreement_rejects_confirmation_from_different_frame():
    primary = [_detection(frame_index=4.0)]
    confirming = [[_detection(frame_index=5.0, label=7)]]

    with pytest.raises(ValueError, match="same frame_index"):
        score_cross_map_agreement(primary, confirming, config=_config())


def test_cross_map_agreement_rejects_mixed_primary_frames():
    primary = [
        _detection(frame_index=4.0),
        _detection(frame_index=5.0, label=2),
    ]
    confirming = [[_detection(frame_index=4.0, label=7)]]

    with pytest.raises(ValueError, match="primary detections.*same frame_index"):
        score_cross_map_agreement(primary, confirming, config=_config())


def test_cross_map_agreement_keeps_same_frame_confirmation():
    primary = [_detection(frame_index=4.0)]
    confirming = [[_detection(frame_index=4.0, label=7)]]

    scores = score_cross_map_agreement(primary, confirming, config=_config())

    assert len(scores) == 1
    assert scores[0].accepted
    assert scores[0].confirming_maps == 1
