import numpy as np
import pytest

from beltmap.map_risk import (
    BeltMapRiskMaps,
    compute_belt_map_risk_maps,
    load_belt_map_support,
    score_map_risk_detections,
)
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


@pytest.mark.parametrize("min_support", [True, "1.0", float("nan"), -1.0])
def test_compute_belt_map_risk_maps_rejects_invalid_min_support(min_support):
    with pytest.raises(ValueError, match="min_support"):
        compute_belt_map_risk_maps(
            np.ones((2, 2), dtype=np.float32),
            min_support=min_support,
        )


@pytest.mark.parametrize(
    "support",
    [
        np.ones((2, 2), dtype=bool),
        np.asarray([[1.0, -0.1]], dtype=np.float32),
        np.asarray([["1.0", "2.0"]], dtype=object),
    ],
)
def test_compute_belt_map_risk_maps_rejects_invalid_support_arrays(support):
    with pytest.raises(ValueError, match="support"):
        compute_belt_map_risk_maps(support)


def test_load_belt_map_support_rejects_invalid_support_file(tmp_path):
    path = tmp_path / "belt_map_support.npy"
    np.save(path, np.asarray([[1.0, -0.1]], dtype=np.float32))

    with pytest.raises(ValueError, match="REUSE_MAP_SUPPORT_PATH"):
        load_belt_map_support(path, map_shape=(1, 2))


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


@pytest.mark.parametrize("reject_max_mean_risk", [1.5, True, "0.5", float("nan")])
def test_score_map_risk_detections_rejects_invalid_thresholds(reject_max_mean_risk):
    maps = compute_belt_map_risk_maps(np.ones((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="reject_max_mean_risk"):
        score_map_risk_detections(
            [_detection(bbox_left=0, bbox_right=1)],
            phase_px=0.0,
            frame_shape=(1, 2),
            maps=maps,
            reject_max_mean_risk=reject_max_mean_risk,
        )


@pytest.mark.parametrize(
    "frame_shape",
    [(1.5, 2), (1, float("nan")), (0, 2), (True, 2), ("1", 2)],
)
def test_score_map_risk_detections_rejects_invalid_frame_shape(frame_shape):
    maps = compute_belt_map_risk_maps(np.ones((2, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="frame_shape"):
        score_map_risk_detections(
            [_detection(bbox_left=0, bbox_right=1)],
            phase_px=0.0,
            frame_shape=frame_shape,
            maps=maps,
        )


@pytest.mark.parametrize("phase_px", [True, "0.0", float("nan")])
def test_score_map_risk_detections_rejects_invalid_phase(phase_px):
    maps = compute_belt_map_risk_maps(np.ones((2, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="phase_px"):
        score_map_risk_detections(
            [_detection(bbox_left=0, bbox_right=1)],
            phase_px=phase_px,
            frame_shape=(1, 2),
            maps=maps,
        )


def test_score_map_risk_detections_rejects_malformed_risk_maps():
    maps = BeltMapRiskMaps(
        support=np.ones((2, 2), dtype=np.float32),
        observed_mask=np.ones((2, 2), dtype=bool),
        interpolated_mask=np.ones((2, 2), dtype=np.float32),
        low_support_mask=np.zeros((2, 2), dtype=bool),
        risk=np.ones((2, 2), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="interpolated_mask"):
        score_map_risk_detections(
            [_detection(bbox_left=0, bbox_right=1)],
            phase_px=0.0,
            frame_shape=(1, 2),
            maps=maps,
        )


def test_score_map_risk_detections_rejects_fractional_detection_bbox():
    maps = compute_belt_map_risk_maps(np.ones((2, 2), dtype=np.float32))

    with pytest.raises(ValueError, match=r"detection\.bbox_top"):
        score_map_risk_detections(
            [_detection(bbox_top=0.5, bbox_left=0, bbox_bottom=1, bbox_right=1)],
            phase_px=0.0,
            frame_shape=(1, 2),
            maps=maps,
        )
