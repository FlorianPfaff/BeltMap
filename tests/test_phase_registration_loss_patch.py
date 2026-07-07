from __future__ import annotations

import pytest

import beltmap  # noqa: F401 - imports side-effect patches
from beltmap.phase import _loss_to_score, _refine_quadratic_offset


def test_subpixel_quadratic_refinement_never_returns_negative_loss() -> None:
    losses = [(1.0, -1.0), (0.0, 0.0), (2.0, 1.0)]

    refined_loss, refined_offset = _refine_quadratic_offset(losses, best_index=1)

    assert refined_offset == pytest.approx(-1.0 / 6.0)
    assert refined_loss == pytest.approx(0.0)
    assert refined_loss >= 0.0


def test_registration_loss_score_is_bounded_for_negative_refined_loss() -> None:
    score = _loss_to_score(-0.25, [0.0, 0.5, 1.0])

    assert score == pytest.approx(1.0)
