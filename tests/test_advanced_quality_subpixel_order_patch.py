from __future__ import annotations

import pytest

import beltmap  # noqa: F401 - imports side-effect patches
import beltmap.advanced_quality as advanced_quality


def test_subpixel_offset_order_patch_is_autoloaded() -> None:
    assert getattr(
        advanced_quality.quadratic_subpixel_minimum,
        "_beltmap_subpixel_offset_order_patched",
        False,
    )


def test_quadratic_subpixel_minimum_sorts_offsets_before_fitting() -> None:
    offsets = [0.0, 1.0, -1.0]
    losses = [(offset - 0.25) ** 2 for offset in offsets]

    refined = advanced_quality.quadratic_subpixel_minimum(offsets, losses)
    sorted_refined = advanced_quality.quadratic_subpixel_minimum(
        sorted(offsets),
        [(offset - 0.25) ** 2 for offset in sorted(offsets)],
    )

    assert refined == pytest.approx(0.25)
    assert refined == pytest.approx(sorted_refined)
