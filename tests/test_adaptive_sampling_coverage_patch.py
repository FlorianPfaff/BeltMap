from __future__ import annotations

import importlib

import beltmap  # noqa: F401 - imports side-effect patches
from beltmap import adaptive_sampling_coverage_patch as coverage_patch
from beltmap import operational_improvements as operational


def test_adaptive_sampling_coverage_patch_is_autoloaded() -> None:
    assert getattr(
        operational.select_adaptive_map_frames,
        "_beltmap_adaptive_sampling_coverage_patched",
        False,
    )


def test_crop_taller_than_map_counts_each_phase_bin_once() -> None:
    samples = operational.select_adaptive_map_frames(
        [0.0, 5.0],
        map_height_px=10,
        sample_count=2,
        crop_height_px=25,
        bin_count=10,
    )

    assert [sample.coverage_gain for sample in samples] == [10, 0]
    assert all(sample.coverage_gain <= 10 for sample in samples)


def test_crop_equal_to_map_covers_the_same_unique_bins() -> None:
    one_period = operational.select_adaptive_map_frames(
        [0.0],
        map_height_px=10,
        sample_count=1,
        crop_height_px=10,
        bin_count=10,
    )
    multiple_periods = operational.select_adaptive_map_frames(
        [0.0],
        map_height_px=10,
        sample_count=1,
        crop_height_px=30,
        bin_count=10,
    )

    assert one_period[0].coverage_gain == 10
    assert multiple_periods[0].coverage_gain == one_period[0].coverage_gain


def test_adaptive_sampling_coverage_patch_reload_keeps_true_original() -> None:
    before = operational.select_adaptive_map_frames
    before_original = getattr(
        before,
        "_beltmap_original_select_adaptive_map_frames",
        before,
    )

    importlib.reload(coverage_patch)
    importlib.reload(coverage_patch)

    after = operational.select_adaptive_map_frames
    after_original = getattr(
        after,
        "_beltmap_original_select_adaptive_map_frames",
        after,
    )
    assert after_original is before_original
    assert after(
        [0.0],
        map_height_px=10,
        sample_count=1,
        crop_height_px=20,
        bin_count=10,
    )[0].coverage_gain == 10
