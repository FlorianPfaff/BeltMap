import numpy as np

from beltmap.advanced_quality import robust_gain_offset


def test_robust_gain_offset_refits_after_last_trim_iteration():
    expected = np.arange(100, dtype=float).reshape(10, 10)
    observed = 2.0 * expected + 5.0
    observed[0, 0] += 10_000.0

    fit = robust_gain_offset(
        observed,
        expected,
        trim_fraction=0.1,
        max_iterations=1,
        min_pixels=20,
    )

    # The single trimming pass removes the extreme outlier.  The coefficients
    # must then be recomputed on those retained pixels rather than left at the
    # fully outlier-biased initial least-squares fit.
    np.testing.assert_allclose(fit.gain, 2.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fit.offset, 5.0, rtol=1e-12, atol=1e-12)
    assert fit.n_pixels == 90
    assert fit.rmse_gray < 1e-10


def test_robust_gain_offset_does_not_adopt_too_small_trimmed_set():
    expected = np.arange(10, dtype=float)
    observed = 1.5 * expected + 3.0
    observed[0] += 100.0

    fit = robust_gain_offset(
        observed,
        expected,
        trim_fraction=0.4,
        max_iterations=1,
        min_pixels=10,
    )

    assert fit.n_pixels == 10
    assert fit.trimmed_fraction == 0.0
