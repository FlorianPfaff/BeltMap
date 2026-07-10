from __future__ import annotations

import json

import numpy as np

import beltmap  # noqa: F401 - imports side-effect patches
from beltmap import postrun_improvements as pri


def test_map_uncertainty_period_state_patch_is_autoloaded() -> None:
    assert getattr(
        pri.compute_phase_row_counts,
        "_beltmap_map_uncertainty_period_state_patched",
        False,
    )
    assert getattr(
        pri.write_map_uncertainty_outputs,
        "_beltmap_map_uncertainty_period_state_patched",
        False,
    )


def test_phase_row_counts_can_preserve_finite_strip_boundaries() -> None:
    periodic = pri.compute_phase_row_counts(
        [0.0, 3.0],
        map_height=5,
        crop_height=3,
    )
    finite = pri.compute_phase_row_counts(
        [0.0, 3.0],
        map_height=5,
        crop_height=3,
        periodic=False,
    )

    assert periodic.tolist() == [2, 1, 1, 1, 1]
    assert finite.tolist() == [1, 1, 1, 1, 1]


def test_map_uncertainty_writer_does_not_wrap_finite_strip_metadata(tmp_path) -> None:
    output_dir = tmp_path / "outputs"
    report_dir = tmp_path / "report"
    output_dir.mkdir()
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "belt_map_height_px": 5,
                "model_period_px": None,
                "belt_period_known": False,
                "belt_map_periodic": False,
                "belt_region": {"height": 3, "width": 2},
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "phase_estimates.csv").write_text(
        "frame_index,phase_px\n0,0\n1,3\n",
        encoding="utf-8",
    )

    summary = pri.write_map_uncertainty_outputs(
        output_dir,
        report_dir=report_dir,
    )

    assert summary["available"] is True
    assert np.load(report_dir / "belt_map_row_counts.npy").tolist() == [1, 1, 1, 1, 1]


def test_map_uncertainty_writer_keeps_known_periodic_metadata(tmp_path) -> None:
    output_dir = tmp_path / "outputs"
    report_dir = tmp_path / "report"
    output_dir.mkdir()
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "belt_map_height_px": 5,
                "model_period_px": 5.0,
                "belt_period_known": True,
                "belt_map_periodic": True,
                "belt_region": {"height": 3, "width": 2},
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "phase_estimates.csv").write_text(
        "frame_index,phase_px\n0,0\n1,3\n",
        encoding="utf-8",
    )

    pri.write_map_uncertainty_outputs(output_dir, report_dir=report_dir)

    assert np.load(report_dir / "belt_map_row_counts.npy").tolist() == [2, 1, 1, 1, 1]
