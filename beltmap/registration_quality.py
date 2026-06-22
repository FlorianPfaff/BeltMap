"""Registration-quality gates for BeltMap residual detection."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np

from .phase import PhaseEstimate
from .residual import ResidualImage

REGISTRATION_QUALITY_ACTIONS = {"report", "inflate", "skip"}
REGISTRATION_QUALITY_FIELDS = [
    "frame_index",
    "image",
    "method",
    "accepted",
    "action",
    "reasons",
    "inflation_factor",
    "score",
    "correction_px",
    "phase_drift_px",
    "loss",
    "second_best_loss",
    "loss_gap",
    "loss_gap_ratio",
    "loss_curvature",
    "uncertainty_px",
]


@dataclass(frozen=True)
class RegistrationQualityGateConfig:
    """Frame-level registration-quality response settings.

    The gates are deliberately frame-level rather than detection-level because
    poor phase registration contaminates the whole residual image. ``report``
    records diagnostics only, ``inflate`` reduces detection confidence by
    increasing the residual noise scale on suspect frames, and ``skip`` writes no
    detections for suspect frames.
    """

    enabled: bool = False
    action: str = "report"
    min_score: float | None = None
    min_loss_gap_ratio: float | None = None
    max_uncertainty_px: float | None = None
    max_abs_correction_px: float | None = None
    noise_inflation_factor: float = 2.0
    uncertainty_inflation_scale: float = 0.0

    def validate(self) -> None:
        _validate_bool_value(self.enabled, "enabled")
        if normalize_registration_quality_action(self.action) != self.action:
            raise ValueError(
                "RegistrationQualityGateConfig.action must already be normalized"
            )
        _validate_optional_non_negative(self.min_score, "min_score")
        _validate_optional_non_negative(
            self.min_loss_gap_ratio,
            "min_loss_gap_ratio",
        )
        _validate_optional_non_negative(
            self.max_uncertainty_px,
            "max_uncertainty_px",
        )
        _validate_optional_non_negative(
            self.max_abs_correction_px,
            "max_abs_correction_px",
        )
        _require_finite(self.noise_inflation_factor, "noise_inflation_factor")
        if _require_finite(self.noise_inflation_factor, "noise_inflation_factor") < 1.0:
            raise ValueError("noise_inflation_factor must be at least 1")
        _require_finite(
            self.uncertainty_inflation_scale,
            "uncertainty_inflation_scale",
        )
        if _require_finite(self.uncertainty_inflation_scale, "uncertainty_inflation_scale") < 0.0:
            raise ValueError("uncertainty_inflation_scale must be non-negative")


def normalize_registration_quality_action(value: str) -> str:
    """Normalize registration-quality response aliases."""

    if not isinstance(value, str):
        raise ValueError("REGISTRATION_QUALITY_ACTION must be a string")
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "diagnostic": "report",
        "diagnostics": "report",
        "none": "report",
        "report": "report",
        "inflate": "inflate",
        "noise_inflate": "inflate",
        "noise_inflation": "inflate",
        "skip": "skip",
        "drop": "skip",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        choices = ", ".join(sorted(REGISTRATION_QUALITY_ACTIONS))
        raise ValueError(
            f"REGISTRATION_QUALITY_ACTION must be one of {choices}; got {value!r}"
        ) from exc


def evaluate_registration_quality(
    residual: ResidualImage,
    *,
    frame_index: int,
    image: str,
    config: RegistrationQualityGateConfig,
) -> tuple[ResidualImage, dict | None, bool]:
    """Apply registration-quality policy to one residual image.

    Returns ``(residual, row, skip_detections)``. ``row`` is ``None`` when the
    feature is disabled; otherwise it contains one CSV-ready diagnostic row.
    """

    _validate_bool_value(config.enabled, "enabled")
    if not config.enabled:
        return residual, None, False
    config.validate()
    if residual.clean_render is None:
        return residual, None, False

    estimate = residual.clean_render.phase_estimate
    reasons = registration_quality_failure_reasons(estimate, config)
    accepted = not reasons
    action = "accept" if accepted else config.action
    skip_detections = False
    inflation_factor = 1.0
    adjusted = residual

    if not accepted:
        if config.action == "skip":
            skip_detections = True
        elif config.action == "inflate":
            inflation_factor = registration_quality_inflation_factor(
                estimate,
                config=config,
            )
            adjusted = residual_with_inflated_noise(residual, inflation_factor)

    return (
        adjusted,
        registration_quality_row(
            estimate,
            frame_index=frame_index,
            image=image,
            accepted=accepted,
            action=action,
            reasons=reasons,
            inflation_factor=inflation_factor,
        ),
        skip_detections,
    )


def registration_quality_failure_reasons(
    estimate: PhaseEstimate,
    config: RegistrationQualityGateConfig,
) -> list[str]:
    """Return failed quality-gate names for a phase estimate."""

    if "registration" not in estimate.method:
        return []

    reasons: list[str] = []
    if config.min_score is not None and not _at_least(estimate.score, config.min_score):
        reasons.append("score")
    if config.min_loss_gap_ratio is not None and not _at_least(
        estimate.loss_gap_ratio,
        config.min_loss_gap_ratio,
    ):
        reasons.append("loss_gap_ratio")
    if config.max_uncertainty_px is not None and not _at_most(
        estimate.uncertainty_px,
        config.max_uncertainty_px,
    ):
        reasons.append("uncertainty_px")
    if config.max_abs_correction_px is not None:
        correction = abs(float(estimate.correction_px))
        if not np.isfinite(correction) or correction > config.max_abs_correction_px:
            reasons.append("correction_px")
    return reasons


def registration_quality_inflation_factor(
    estimate: PhaseEstimate,
    *,
    config: RegistrationQualityGateConfig,
) -> float:
    """Return the residual-noise multiplier for a suspect frame."""

    factor = max(
        1.0,
        _require_finite(config.noise_inflation_factor, "noise_inflation_factor"),
    )
    uncertainty = estimate.uncertainty_px
    if (
        uncertainty is not None
        and np.isfinite(uncertainty)
        and _require_finite(
            config.uncertainty_inflation_scale,
            "uncertainty_inflation_scale",
        ) > 0.0
    ):
        factor = max(
            factor,
            1.0
            + _require_finite(
                config.uncertainty_inflation_scale,
                "uncertainty_inflation_scale",
            )
            * max(0.0, float(uncertainty)),
        )
    return float(factor)


def residual_with_inflated_noise(residual: ResidualImage, factor: float) -> ResidualImage:
    """Return a residual whose normalized signal is divided by ``factor``."""

    try:
        factor_value = _require_finite(factor, "factor")
    except ValueError as exc:
        raise ValueError("noise inflation factor must be finite and positive") from exc
    if factor_value < 1.0:
        raise ValueError("noise inflation factor must be at least 1")
    if factor_value == 1.0:
        return residual
    if (
        residual.raw.shape != residual.local_noise.shape
        or residual.raw.shape != residual.normalized.shape
        or residual.raw.shape != residual.mask.shape
    ):
        raise ValueError("residual arrays must have matching shapes")
    local_noise = np.asarray(residual.local_noise, dtype=np.float64) * factor_value
    valid = (
        residual.mask
        & np.isfinite(residual.raw)
        & np.isfinite(local_noise)
        & (local_noise > 0)
    )
    normalized = np.full(residual.normalized.shape, np.nan, dtype=np.float64)
    normalized[valid] = residual.raw[valid] / local_noise[valid]
    return ResidualImage(
        raw=residual.raw,
        local_noise=local_noise,
        normalized=normalized,
        mask=valid,
        expected_background=residual.expected_background,
        clean_render=residual.clean_render,
    )


def registration_quality_row(
    estimate: PhaseEstimate,
    *,
    frame_index: int,
    image: str,
    accepted: bool,
    action: str,
    reasons: list[str],
    inflation_factor: float,
) -> dict:
    """Return a CSV-ready diagnostic row for one phase estimate."""

    return {
        "frame_index": frame_index,
        "image": image,
        "method": estimate.method,
        "accepted": accepted,
        "action": action,
        "reasons": ";".join(reasons),
        "inflation_factor": inflation_factor,
        "score": _csv_float(estimate.score),
        "correction_px": estimate.correction_px,
        "phase_drift_px": estimate.drift_px,
        "loss": _csv_float(estimate.loss),
        "second_best_loss": _csv_float(estimate.second_best_loss),
        "loss_gap": _csv_float(estimate.loss_gap),
        "loss_gap_ratio": _csv_float(estimate.loss_gap_ratio),
        "loss_curvature": _csv_float(estimate.loss_curvature),
        "uncertainty_px": _csv_float(estimate.uncertainty_px),
    }


def _at_least(value: float | None, threshold: float) -> bool:
    return value is not None and np.isfinite(value) and float(value) >= threshold


def _at_most(value: float | None, threshold: float) -> bool:
    return value is not None and np.isfinite(value) and float(value) <= threshold


def _csv_float(value: float | None) -> float | str:
    return "" if value is None else float(value)


def _validate_optional_non_negative(value: float | None, name: str) -> None:
    if value is None:
        return
    parsed = _require_finite(value, name)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative when set")


def _require_finite(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric, not boolean")
    if not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _validate_bool_value(value: bool, name: str) -> None:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be boolean")
