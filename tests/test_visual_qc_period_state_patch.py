from __future__ import annotations

import numpy as np

import beltmap  # noqa: F401 - imports side-effect patches
from beltmap import visual_qc


def _metadata(*, periodic: bool) -> dict[str, object]:
    return {
        "belt_map_height_px": 4,
        "belt_map_periodic": periodic,
        "belt_region": {"top": 0, "left": 0, "height": 2, "width": 1},
    }


def test_visual_qc_period_state_patch_is_autoloaded() -> None:
    assert getattr(
        visual_qc.estimate_belt_map_coverage,
        "_beltmap_visual_qc_finite_strip_coverage_patched",
        False,
    )


def test_finite_strip_coverage_does_not_wrap_beyond_reconstructed_support() -> None:
    coverage = visual_qc.estimate_belt_map_coverage(
        [{"frame_index": 0, "phase_px": 3.0}],
        _metadata(periodic=False),
    )

    assert coverage is not None
    np.testing.assert_array_equal(coverage, [0.0, 0.0, 0.0, 1.0])
    assert float(np.sum(coverage)) == 1.0


def test_periodic_coverage_still_wraps_across_map_boundary() -> None:
    coverage = visual_qc.estimate_belt_map_coverage(
        [{"frame_index": 0, "phase_px": 3.0}],
        _metadata(periodic=True),
    )

    assert coverage is not None
    np.testing.assert_array_equal(coverage, [1.0, 0.0, 0.0, 1.0])
    assert float(np.sum(coverage)) == 2.0
