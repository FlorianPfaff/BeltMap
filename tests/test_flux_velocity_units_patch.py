from __future__ import annotations

import math

import pytest

import beltmap  # noqa: F401 - imports side-effect patches
import beltmap.operational_improvements as operational


def test_flux_velocity_units_patch_is_autoloaded() -> None:
    assert getattr(
        operational.summarize_flux,
        "_beltmap_flux_velocity_units_patched",
        False,
    )


def test_flux_summary_does_not_relabel_per_frame_velocity_without_frame_rate() -> None:
    summary = operational.summarize_flux(
        [{"velocity_y_px_per_frame": "2.5", "velocity_ratio_y": "0.5"}],
        duration_s=2.0,
    )

    assert summary.particle_flux_per_s == pytest.approx(0.5)
    assert summary.mean_velocity_y_px_per_s is None
    assert summary.frame_rate_hz is None


def test_flux_summary_converts_velocity_with_positive_frame_rate() -> None:
    summary = operational.summarize_flux(
        [
            {"velocity_y_px_per_frame": "2.0"},
            {"velocity_y_px_per_frame": "3.0"},
        ],
        frame_count=100,
        frame_rate_hz=50.0,
    )

    assert summary.mean_velocity_y_px_per_s == pytest.approx(125.0)
    assert summary.particle_flux_per_s == pytest.approx(1.0)
    assert summary.frame_rate_hz == pytest.approx(50.0)


@pytest.mark.parametrize("frame_rate_hz", [0.0, -10.0])
def test_flux_summary_omits_per_second_velocity_for_unspecified_frame_rates(
    frame_rate_hz,
) -> None:
    summary = operational.summarize_flux(
        [{"velocity_y_px_per_frame": "2.0"}],
        frame_rate_hz=frame_rate_hz,
        duration_s=1.0,
    )

    assert summary.mean_velocity_y_px_per_s is None
    assert summary.frame_rate_hz is None
    assert math.isfinite(summary.particle_flux_per_s)


@pytest.mark.parametrize("frame_rate_hz", [float("nan"), float("inf"), True])
def test_flux_summary_rejects_invalid_frame_rate_metadata(frame_rate_hz) -> None:
    with pytest.raises(ValueError, match="frame_rate_hz must be a finite number"):
        operational.summarize_flux(
            [{"velocity_y_px_per_frame": "2.0"}],
            frame_rate_hz=frame_rate_hz,
            duration_s=1.0,
        )
