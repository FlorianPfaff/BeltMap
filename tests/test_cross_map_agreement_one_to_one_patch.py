from beltmap import (
    CrossMapAgreementConfig,
    ParticleDetection,
    score_cross_map_agreement,
)
from beltmap.cross_map_agreement import (
    score_cross_map_agreement as module_score_cross_map_agreement,
)


def _detection(
    label: int,
    y: float,
    x: float,
    bbox: tuple[int, int, int, int],
    peak: float = 10.0,
) -> ParticleDetection:
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
        mean_signal=0.5 * peak,
        peak_signal=peak,
    )


def _config() -> CrossMapAgreementConfig:
    return CrossMapAgreementConfig(
        max_centroid_distance_px=2.0,
        min_bbox_iou=0.2,
        min_peak_ratio=0.5,
        require_sign_consistency=False,
        min_confirming_maps=1,
    )


def test_cross_map_agreement_does_not_reuse_confirming_component():
    primary_exact = _detection(1, 10.0, 10.0, (8, 8, 13, 13))
    primary_duplicate = _detection(2, 11.0, 10.0, (9, 8, 14, 13))
    confirming = _detection(7, 10.0, 10.0, (8, 8, 13, 13))

    scores = score_cross_map_agreement(
        [primary_exact, primary_duplicate],
        [[confirming]],
        config=_config(),
    )

    assert [score.accepted for score in scores] == [True, False]
    assert [score.confirming_maps for score in scores] == [1, 0]
    assert scores[1].matches[0].matched_label == confirming.label
    assert scores[1].matches[0].accepted is False


def test_cross_map_agreement_maximizes_one_to_one_match_cardinality():
    primary_flexible = _detection(1, 10.0, 10.0, (8, 8, 13, 13))
    primary_constrained = _detection(2, 10.0, 11.0, (8, 9, 13, 14))
    confirming_shared = _detection(7, 10.0, 10.8, (8, 9, 13, 14))
    confirming_alternative = _detection(8, 10.0, 8.5, (8, 7, 13, 12))

    scores = score_cross_map_agreement(
        [primary_flexible, primary_constrained],
        [[confirming_shared, confirming_alternative]],
        config=_config(),
    )

    assert [score.accepted for score in scores] == [True, True]
    assert {score.matches[0].matched_label for score in scores} == {7, 8}


def test_package_export_uses_one_to_one_cross_map_scorer():
    assert score_cross_map_agreement is module_score_cross_map_agreement
