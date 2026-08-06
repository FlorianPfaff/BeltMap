from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import beltmap  # noqa: F401 - imports side-effect patches
import beltmap.benchmark as benchmark


def test_benchmark_map_overlap_patch_is_autoloaded() -> None:
    assert getattr(
        benchmark.map_metrics,
        "_beltmap_map_overlap_patched",
        False,
    )


def test_map_metrics_prefers_maximum_finite_overlap_before_rmse(
    tmp_path: Path,
) -> None:
    truth_dir = tmp_path / "truth"
    output_dir = tmp_path / "output"
    truth_dir.mkdir()
    output_dir.mkdir()

    # At shift 0, both finite rows agree within one gray level.  At shift 1,
    # only one finite pixel remains, but it matches exactly.  Minimizing RMSE
    # alone therefore selected shift 1 and reported a misleading zero error.
    target = np.asarray([[0.0], [1.0], [np.nan], [np.nan]])
    reconstructed = np.asarray([[1.0], [2.0], [np.nan], [np.nan]])
    np.save(truth_dir / "true_belt_map.npy", target)
    np.save(output_dir / "belt_map.npy", reconstructed)

    metrics = benchmark.map_metrics(
        output_dir,
        truth_dir / "synthetic_metadata.json",
        {"true_belt_map_npy": "true_belt_map.npy"},
    )

    assert metrics["available"] is True
    assert metrics["best_cyclic_shift_px"] == 0
    assert metrics["finite_pixels"] == 2
    assert metrics["rmse_gray"] == pytest.approx(1.0)
    assert metrics["mean_abs_error_gray"] == pytest.approx(1.0)
