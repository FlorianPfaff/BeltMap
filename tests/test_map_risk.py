import numpy as np
import pytest

from beltmap.map_risk import compute_belt_map_risk_maps, score_map_risk_detections
from beltmap.tracking import ParticleDetection


def _detection(**kwargs) -> ParticleDetection:
    values = dict(
        frame_index=0.0,
        label=1,
        y=0.5,
        x=1.0,
        area_px=2,
        bbox_top=0,
        bbox_left=1,
        bbox_bottom=2,
        bbox_right=2,
    )
    values.update(kwargs)
    return ParticleDetection(**values)


def test_compute_belt_map_risk_maps_marks_interpolated_and_low_support_pixels():
    support = np.asarray([[0.0, 0.5, 2.0]], dtype=np.float32)

    maps = compute_belt_map_risk_maps(support, min_support=1.0)

    np.testing.assert_array_equal(maps.observed_mask, [[False, True, True]])
    np.testing.assert_array_equal(maps.interpolated_mask, [[True, False, False]])
    np.testing.assert_array_equal(maps.low_support_mask, [[True, True, False]])
    np.testing.assert_allclose(maps.risk, [[1.0, 0.5, 0.0]])


def test_score_map_risk_detections_adds_bbox_support_stats_and_rejects():
    support = np.full((4, 3), 2.0, dtype=np.float32)
    support[1, 1] = 0.0
    maps = compute_belt_map_risk_maps(support, min_support=1.0)

    scores = score_map_risk_detections(
        [_detection()],
        phase_px=0.0,
        frame_shape=(2, 3),
        maps=maps,
        reject_max_mean_risk=1.0,
        reject_max_interpolated_fraction=0.25,
        reject_max_low_support_fraction=1.0,
    )

    assert len(scores) == 1
    assert scores[0].rejected is True
    scored = scores[0].detection
    assert scored.map_support_min == pytest.approx(0.0)
    assert scored.map_support_mean == pytest.approx(1.0)
    assert scored.map_risk_mean == pytest.approx(0.5)
    assert scored.map_risk_max == pytest.approx(1.0)
    assert scored.map_interpolated_fraction == pytest.approx(0.5)
    assert scored.map_low_support_fraction == pytest.approx(0.5)


def test_score_map_risk_detections_rejects_invalid_thresholds():
    maps = compute_belt_map_risk_maps(np.ones((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="reject_max_mean_risk"):
        score_map_risk_detections(
            [_detection(bbox_left=0, bbox_right=1)],
            phase_px=0.0,
            frame_shape=(1, 2),
            maps=maps,
            reject_max_mean_risk=1.5,
        )


@pytest.mark.parametrize("frame_shape", [(1.5, 2), (1, float("nan")), (0, 2)])
def test_score_map_risk_detections_rejects_invalid_frame_shape(frame_shape):
    maps = compute_belt_map_risk_maps(np.ones((2, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="frame_shape"):
        score_map_risk_detections(
            [_detection(bbox_left=0, bbox_right=1)],
            phase_px=0.0,
            frame_shape=frame_shape,
            maps=maps,
        )


def test_score_map_risk_detections_rejects_fractional_bbox_coordinates():
    maps = compute_belt_map_risk_maps(np.ones((2, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="bbox_top must be a finite integer"):
        score_map_risk_detections(
            [_detection(bbox_top=0.5, bbox_left=0, bbox_bottom=1, bbox_right=1)],
            phase_px=0.0,
            frame_shape=(1, 2),
            maps=maps,
        )
