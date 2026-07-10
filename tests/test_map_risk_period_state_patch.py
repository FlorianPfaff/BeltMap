from __future__ import annotations

import numpy as np
import pytest

import beltmap  # noqa: F401 - imports side-effect patches
import beltmap.driver_period_state_patch as driver_period_state
import beltmap.map_risk as map_risk
from beltmap.tracking import ParticleDetection


@pytest.fixture(autouse=True)
def reset_driver_period_context():
    previous = driver_period_state._DRIVER_MODEL_PERIOD_PX[0]
    driver_period_state._DRIVER_MODEL_PERIOD_PX[0] = (
        driver_period_state._DRIVER_MODEL_PERIOD_UNKNOWN
    )
    try:
        yield
    finally:
        driver_period_state._DRIVER_MODEL_PERIOD_PX[0] = previous


def boundary_detection() -> ParticleDetection:
    return ParticleDetection(
        frame_index=0.0,
        label=1,
        y=0.5,
        x=0.0,
        area_px=2,
        bbox_top=0,
        bbox_left=0,
        bbox_bottom=2,
        bbox_right=1,
    )


def test_map_risk_period_state_patch_is_autoloaded() -> None:
    assert getattr(
        map_risk.render_belt_view,
        "_beltmap_map_risk_period_state_patched",
        False,
    )


def test_finite_strip_map_risk_does_not_wrap_opposite_edge_support() -> None:
    maps = map_risk.compute_belt_map_risk_maps(
        np.asarray([[1.0], [1.0], [1.0], [0.0]], dtype=np.float32),
        min_support=1.0,
    )
    detection = boundary_detection()

    driver_period_state._DRIVER_MODEL_PERIOD_PX[0] = None
    finite_strip_score = map_risk.score_map_risk_detections(
        [detection],
        phase_px=3.0,
        frame_shape=(2, 1),
        maps=maps,
        reject_max_mean_risk=0.75,
    )[0]

    driver_period_state._DRIVER_MODEL_PERIOD_PX[0] = 4.0
    periodic_score = map_risk.score_map_risk_detections(
        [detection],
        phase_px=3.0,
        frame_shape=(2, 1),
        maps=maps,
        reject_max_mean_risk=0.75,
    )[0]

    assert finite_strip_score.detection.map_risk_mean == pytest.approx(1.0)
    assert finite_strip_score.detection.map_interpolated_fraction == pytest.approx(1.0)
    assert finite_strip_score.rejected is True

    assert periodic_score.detection.map_risk_mean == pytest.approx(0.5)
    assert periodic_score.detection.map_interpolated_fraction == pytest.approx(0.5)
    assert periodic_score.rejected is False


def test_map_risk_rendering_remains_periodic_outside_driver_context() -> None:
    belt_map = np.arange(4, dtype=np.float32).reshape(4, 1)

    rendered = map_risk.render_belt_view(belt_map, 3.0, 2)

    np.testing.assert_allclose(rendered[:, 0], [3.0, 0.0])
