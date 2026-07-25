from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

import beltmap  # noqa: F401 - imports side-effect patches
from beltmap import operational_improvements as operational
import beltmap.flux_summary_finite_patch as flux_patch
from beltmap.cli.flux_summary import main


def _velocity_rows() -> list[dict[str, str]]:
    return [
        {
            "track_id": "1",
            "velocity_ratio_y": "0.5",
            "velocity_y_px_per_frame": "2.0",
        }
    ]


def test_flux_summary_finite_patch_is_autoloaded() -> None:
    assert getattr(
        operational.summarize_flux,
        "_beltmap_finite_flux_summary_patched",
        False,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frame_count": float("inf")},
        {"frame_count": 1.5},
        {"frame_rate_hz": float("nan")},
        {"frame_rate_hz": float("inf")},
        {"duration_s": float("nan")},
        {"belt_velocity_px_per_s": float("-inf")},
    ],
)
def test_summarize_flux_rejects_nonfinite_or_fractional_metadata(kwargs) -> None:
    with pytest.raises(ValueError, match="must be"):
        operational.summarize_flux(_velocity_rows(), **kwargs)


def test_summarize_flux_rejects_overflowed_derived_velocity() -> None:
    with np.errstate(over="ignore"):
        with pytest.raises(
            ValueError,
            match="mean_velocity_y_px_per_s must be finite",
        ):
            operational.summarize_flux(
                [
                    {"velocity_y_px_per_frame": "1e308"},
                    {"velocity_y_px_per_frame": "1e308"},
                ],
                frame_rate_hz=1e308,
            )


def test_flux_summary_cli_rejects_nonfinite_frame_rate_before_writing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    velocities_path = tmp_path / "velocities.csv"
    velocities_path.write_text(
        "track_id,velocity_ratio_y,velocity_y_px_per_frame\n"
        "1,0.5,2.0\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "science"

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--velocities",
                str(velocities_path),
                "--output-dir",
                str(output_dir),
                "--frame-rate-hz",
                "nan",
            ]
        )

    assert exc_info.value.code == 2
    assert "frame rate must be a finite non-negative number" in capsys.readouterr().err
    assert not output_dir.exists()


def test_flux_summary_patch_reload_keeps_true_original() -> None:
    before = operational.summarize_flux
    before_original = getattr(
        before,
        "_beltmap_original_summarize_flux",
        before,
    )

    importlib.reload(flux_patch)
    importlib.reload(flux_patch)

    after = operational.summarize_flux
    after_original = getattr(
        after,
        "_beltmap_original_summarize_flux",
        after,
    )
    assert after_original is before_original
