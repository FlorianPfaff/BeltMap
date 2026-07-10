from __future__ import annotations

import numpy as np
import pytest

import beltmap  # noqa: F401 - imports side-effect patches
from beltmap import operational_improvements as operational


def test_fdr_threshold_step_up_patch_is_autoloaded() -> None:
    assert getattr(
        operational.fdr_threshold_from_p_values,
        "_beltmap_benjamini_hochberg_step_up_patched",
        False,
    )


def test_fdr_threshold_uses_largest_passing_rank_cutoff() -> None:
    p_values = np.asarray([0.03, 0.04])
    scores = np.asarray([1.0, 10.0])

    threshold = operational.fdr_threshold_from_p_values(
        p_values,
        scores,
        alpha=0.05,
    )

    # Sorted critical values are [0.025, 0.05].  Rank one fails but rank two
    # passes, so the BH step-up rejection set contains both hypotheses.
    assert threshold == pytest.approx(1.0)
