import numpy as np
import pytest
from PIL import Image

from beltmap._driver_map import build_belt_map_result
from beltmap.cross_map_agreement import (
    CrossMapAgreementConfig,
    bbox_iou,
    filter_detections_by_agreement,
    peak_ratio,
    score_cross_map_agreement,
)
from beltmap.residual import ResidualImage
from beltmap.tracking import ParticleDetection


def _detection(label=1, y=10.0, x=10.0, peak=12.0, bbox=(8, 8, 13, 13)):
    top, left, bottom, right = bbox
    return ParticleDetection(
        frame_index=0.0,
        label=label,
        y=y,
        x=x,
        area_px=16,
        bbox_top=top,
        bbox_left=left,
        bbox_bottom=bottom,
        bbox_right=right,
        mean_signal=None if peak is None else 0.5 * peak,
        peak_signal=peak,
    )


def _residual(sign=1.0, points=((10, 10),)):
    raw = np.zeros((24, 24), dtype=np.float64)
    for y, x in points:
        raw[y, x] = sign * 12.0
    local_noise = np.ones_like(raw)
    return ResidualImage(
        raw=raw,
        local_noise=local_noise,
        normalized=raw / local_noise,
        mask=np.ones(raw.shape, dtype=bool),
        expected_background=np.zeros_like(raw),
    )


def test_cross_map_agreement_accepts_detection_confirmed_by_both_maps():
    primary = [_detection()]
    confirming = [
        [_detection(label=7, y=10.3, x=10.2, peak=10.0)],
        [_detection(label=8, y=9.8, x=10.1, peak=11.0)],
    ]

    scores = score_cross_map_agreement(
        primary,
        confirming,
        primary_residual=_residual(points=((10, 10),)),
        confirming_residuals=[
            _residual(points=((10, 10),)),
            _residual(points=((10, 10),)),
        ],
        config=CrossMapAgreementConfig(
            max_centroid_distance_px=2.0,
            min_bbox_iou=0.2,
            min_peak_ratio=0.5,
            require_sign_consistency=True,
            min_confirming_maps=2,
        ),
    )

    assert scores[0].accepted
    assert scores[0].confirming_maps == 2
    assert filter_detections_by_agreement(scores)[0] is primary[0]


def test_cross_map_agreement_rejects_when_one_map_does_not_reproduce_component():
    primary = [_detection()]
    confirming = [
        [_detection(label=7, y=10.3, x=10.2, peak=10.0)],
        [_detection(label=8, y=18.0, x=18.0, peak=11.0, bbox=(16, 16, 21, 21))],
    ]

    scores = score_cross_map_agreement(
        primary,
        confirming,
        primary_residual=_residual(points=((10, 10),)),
        confirming_residuals=[
            _residual(points=((10, 10),)),
            _residual(points=((18, 18),)),
        ],
        config=CrossMapAgreementConfig(
            max_centroid_distance_px=2.0,
            min_bbox_iou=0.2,
            min_peak_ratio=0.5,
            require_sign_consistency=True,
            min_confirming_maps=2,
        ),
    )

    assert not scores[0].accepted
    assert scores[0].confirming_maps == 1
    assert filter_detections_by_agreement(scores) == []


def test_cross_map_agreement_rejects_fractional_bbox_coordinates():
    primary = [_detection(bbox=(8.5, 8, 13, 13))]
    confirming = [[_detection(label=7)]]

    with pytest.raises(ValueError, match="first bbox_top"):
        score_cross_map_agreement(
            primary,
            confirming,
            config=CrossMapAgreementConfig(
                require_sign_consistency=False,
                min_confirming_maps=1,
            ),
        )


def test_cross_map_agreement_rejects_degenerate_bbox_coordinates():
    primary = [_detection(bbox=(8, 8, 8, 13))]
    confirming = [[_detection(label=7)]]

    with pytest.raises(ValueError, match="first bbox must have positive"):
        score_cross_map_agreement(
            primary,
            confirming,
            config=CrossMapAgreementConfig(
                require_sign_consistency=False,
                min_confirming_maps=1,
            ),
        )


@pytest.mark.parametrize("min_confirming_maps", [float("nan"), 1.5, 0])
def test_cross_map_agreement_rejects_invalid_min_confirming_maps(min_confirming_maps):
    with pytest.raises(ValueError, match="min_confirming_maps"):
        CrossMapAgreementConfig(min_confirming_maps=min_confirming_maps)


def test_cross_map_agreement_normalizes_integral_min_confirming_maps():
    config = CrossMapAgreementConfig(min_confirming_maps=1.0)

    assert config.min_confirming_maps == 1
    assert isinstance(config.min_confirming_maps, int)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_centroid_distance_px": True}, "max_centroid_distance_px"),
        ({"max_centroid_distance_px": "4.0"}, "max_centroid_distance_px"),
        ({"min_bbox_iou": True}, "min_bbox_iou"),
        ({"min_bbox_iou": 1.1}, "min_bbox_iou"),
        ({"min_peak_ratio": True}, "min_peak_ratio"),
        ({"min_peak_ratio": -0.1}, "min_peak_ratio"),
        ({"min_confirming_maps": True}, "min_confirming_maps"),
        ({"require_sign_consistency": "false"}, "require_sign_consistency"),
        ({"filter_detections": "false"}, "filter_detections"),
    ],
)
def test_cross_map_agreement_rejects_invalid_config_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        CrossMapAgreementConfig(**kwargs)


def test_cross_map_agreement_normalizes_numpy_scalar_config_values():
    config = CrossMapAgreementConfig(
        max_centroid_distance_px=np.float64(4),
        min_bbox_iou=np.float64(0.25),
        min_peak_ratio=np.float64(0.5),
        min_confirming_maps=np.int64(1),
        require_sign_consistency=np.bool_(True),
        filter_detections=np.bool_(False),
    )

    assert config.max_centroid_distance_px == 4.0
    assert isinstance(config.max_centroid_distance_px, float)
    assert config.min_bbox_iou == 0.25
    assert isinstance(config.min_bbox_iou, float)
    assert config.min_peak_ratio == 0.5
    assert isinstance(config.min_peak_ratio, float)
    assert config.min_confirming_maps == 1
    assert isinstance(config.min_confirming_maps, int)
    assert config.require_sign_consistency is True
    assert config.filter_detections is False


def test_cross_map_agreement_rejects_invalid_primary_detection():
    with pytest.raises(ValueError, match=r"primary detection\.y"):
        score_cross_map_agreement(
            [_detection(y=float("nan"))],
            [[_detection()]],
            config=CrossMapAgreementConfig(min_confirming_maps=1),
        )


def test_cross_map_agreement_rejects_invalid_confirming_detection():
    with pytest.raises(ValueError, match=r"confirming detection\.bbox_top"):
        score_cross_map_agreement(
            [_detection()],
            [[_detection(bbox=(8.5, 8, 13, 13))]],
            config=CrossMapAgreementConfig(min_confirming_maps=1),
        )


def test_bbox_iou_rejects_fractional_bbox_edges():
    with pytest.raises(ValueError, match=r"b\.bbox_top"):
        bbox_iou(_detection(), _detection(bbox=(8.5, 8, 13, 13)))


def test_peak_ratio_rejects_bool_and_nonfinite_values():
    with pytest.raises(ValueError, match="peak a"):
        peak_ratio(True, 1.0)
    with pytest.raises(ValueError, match="peak b"):
        peak_ratio(1.0, float("nan"))


def test_build_belt_map_result_records_overridden_sample_indices(tmp_path, monkeypatch):
    monkeypatch.setenv("PROGRESS_INTERVAL_FRAMES", "1000")
    paths = []
    for index in range(6):
        frame = np.full((4, 3), 80 + index, dtype=np.uint8)
        path = tmp_path / f"frame_{index:03d}.bmp"
        Image.fromarray(frame).save(path)
        paths.append(path)

    result = build_belt_map_result(
        paths=paths,
        region=(0, 0, 4, 3),
        velocity=1.0,
        supplied_period=8,
        mask_iterations=0,
        sample_indices_override=[4, 0, 2, 2],
    )

    assert result.sample_indices == (0, 2, 4)
