import numpy as np

from beltmap import _driver_runtime as rt
from beltmap.driver import apply_local_illumination_correction
from beltmap.residual import ResidualConfig, generate_residual_image


def test_detection_local_illumination_correction_returns_ablation_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "DATA", tmp_path)

    expected = np.full((4, 4), 100.0, dtype=np.float64)
    additive_field = np.tile(np.asarray([-20.0, -10.0, 10.0, 20.0]), (4, 1))
    frame = expected + additive_field
    path = tmp_path / "frame_000.png"

    residual_config = ResidualConfig(
        noise_radius_px=0,
        noise_exclusion_sigma=None,
        min_noise=1.0,
    )
    residual = generate_residual_image(frame, expected, config=residual_config)

    corrected, row, illumination_field = apply_local_illumination_correction(
        frame=frame,
        residual=residual,
        residual_config=residual_config,
        frame_index=0,
        path=path,
        enabled=True,
        tile_px=2,
        min_pixels=4,
        mask_threshold=100.0,
        mask_mode="positive",
        mask_grow_threshold=0.0,
        mask_dilation_px=0,
        mask_margin_px=0,
        mask_min_area_px=1,
    )

    assert row is not None
    assert row["status"] == "ok"
    assert row["frame_index"] == 0
    assert row["image"] == "frame_000.png"
    assert row["tile_px"] == 2
    assert row["fit_pixels"] == 16
    assert row["masked_pixels"] == 0
    assert row["field_max_abs_gray"] > 0
    assert row["residual_rmse_after_gray"] < row["residual_rmse_before_gray"]
    assert illumination_field is not None
    assert illumination_field.shape == frame.shape
    assert np.mean(np.abs(corrected.raw)) < np.mean(np.abs(residual.raw))


def test_detection_local_illumination_correction_skips_when_fit_support_is_too_small(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "DATA", tmp_path)

    expected = np.full((3, 3), 50.0, dtype=np.float64)
    frame = expected + 5.0
    residual_config = ResidualConfig(
        noise_radius_px=0,
        noise_exclusion_sigma=None,
        min_noise=1.0,
    )
    residual = generate_residual_image(frame, expected, config=residual_config)

    corrected, row, illumination_field = apply_local_illumination_correction(
        frame=frame,
        residual=residual,
        residual_config=residual_config,
        frame_index=1,
        path=tmp_path / "frame_001.png",
        enabled=True,
        tile_px=2,
        min_pixels=100,
        mask_threshold=100.0,
        mask_mode="positive",
        mask_grow_threshold=0.0,
        mask_dilation_px=0,
        mask_margin_px=0,
        mask_min_area_px=1,
    )

    assert corrected is residual
    assert illumination_field is None
    assert row is not None
    assert row["status"] == "skipped:insufficient_fit_pixels"
    assert row["fit_pixels"] == 9
    assert row["masked_pixels"] == 0
