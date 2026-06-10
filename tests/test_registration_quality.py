import numpy as np
import pytest

from beltmap.phase import PhaseEstimate
from beltmap.registration_quality import (
    RegistrationQualityGateConfig,
    evaluate_registration_quality,
    registration_quality_failure_reasons,
    residual_with_inflated_noise,
)
from beltmap.rendering import BeltRegion, CleanBeltRender
from beltmap.residual import ResidualImage


def make_residual(estimate: PhaseEstimate) -> ResidualImage:
    raw = np.array([[2.0, 4.0], [6.0, 8.0]])
    noise = np.ones_like(raw)
    mask = np.ones(raw.shape, dtype=bool)
    clean = CleanBeltRender(
        image=np.zeros_like(raw),
        mask=mask,
        phase_estimate=estimate,
        belt_region=BeltRegion(0, 0, raw.shape[0], raw.shape[1]),
    )
    return ResidualImage(
        raw=raw,
        local_noise=noise,
        normalized=raw / noise,
        mask=mask,
        expected_background=np.zeros_like(raw),
        clean_render=clean,
    )


def test_registration_quality_reasons_cover_score_gap_uncertainty_and_correction():
    estimate = PhaseEstimate(
        phase_px=0.0,
        frame_index=0.0,
        predicted_phase_px=0.0,
        correction_px=4.5,
        loss=1.0,
        score=0.05,
        loss_gap_ratio=0.01,
        uncertainty_px=3.0,
        method="registration",
    )
    config = RegistrationQualityGateConfig(
        enabled=True,
        action="report",
        min_score=0.1,
        min_loss_gap_ratio=0.05,
        max_uncertainty_px=1.5,
        max_abs_correction_px=4.0,
    )

    assert registration_quality_failure_reasons(estimate, config) == [
        "score",
        "loss_gap_ratio",
        "uncertainty_px",
        "correction_px",
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_score": float("nan")},
        {"min_loss_gap_ratio": float("nan")},
        {"max_uncertainty_px": float("nan")},
        {"max_abs_correction_px": float("nan")},
        {"noise_inflation_factor": float("nan")},
        {"uncertainty_inflation_scale": float("nan")},
    ],
)
def test_registration_quality_config_rejects_nonfinite_thresholds(kwargs):
    config = RegistrationQualityGateConfig(**kwargs)

    with pytest.raises(ValueError, match="must be finite"):
        config.validate()


def test_registration_quality_inflates_low_quality_residual_noise():
    estimate = PhaseEstimate(
        phase_px=0.0,
        frame_index=0.0,
        predicted_phase_px=0.0,
        score=0.0,
        uncertainty_px=4.0,
        method="registration",
    )
    residual = make_residual(estimate)
    config = RegistrationQualityGateConfig(
        enabled=True,
        action="inflate",
        min_score=0.1,
        noise_inflation_factor=2.0,
        uncertainty_inflation_scale=0.75,
    )

    inflated, row, skip = evaluate_registration_quality(
        residual,
        frame_index=0,
        image="frame.png",
        config=config,
    )

    assert not skip
    assert row is not None
    assert row["accepted"] is False
    assert row["action"] == "inflate"
    assert row["inflation_factor"] == 4.0
    np.testing.assert_allclose(inflated.local_noise, 4.0 * residual.local_noise)
    np.testing.assert_allclose(inflated.normalized, residual.normalized / 4.0)


def test_registration_quality_skip_marks_frame_without_changing_residual():
    estimate = PhaseEstimate(
        phase_px=0.0,
        frame_index=0.0,
        predicted_phase_px=0.0,
        score=0.0,
        method="registration",
    )
    residual = make_residual(estimate)

    adjusted, row, skip = evaluate_registration_quality(
        residual,
        frame_index=5,
        image="frame.png",
        config=RegistrationQualityGateConfig(
            enabled=True,
            action="skip",
            min_score=0.1,
        ),
    )

    assert adjusted is residual
    assert skip
    assert row is not None
    assert row["action"] == "skip"
    assert row["reasons"] == "score"


def test_residual_with_inflated_noise_is_noop_for_factor_one():
    estimate = PhaseEstimate(phase_px=0.0, frame_index=0.0, predicted_phase_px=0.0)
    residual = make_residual(estimate)

    assert residual_with_inflated_noise(residual, 1.0) is residual
