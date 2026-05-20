import numpy as np
import pytest
from PIL import Image

from beltmap import _driver_runtime as rt
from beltmap._driver_map import accumulate_belt_map, validate_map_trim_fraction


def _write_gray(path, value: int) -> None:
    Image.fromarray(np.full((1, 1), value, dtype=np.uint8)).save(path)


def test_trimmed_map_reconstruction_rejects_single_bright_outlier(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "OUT", tmp_path / "out")
    monkeypatch.setenv("PROGRESS_INTERVAL_FRAMES", "1000")

    paths = []
    for index, value in enumerate([10, 10, 10, 250]):
        path = tmp_path / f"frame_{index:03d}.png"
        _write_gray(path, value)
        paths.append(path)

    common_kwargs = dict(
        paths=paths,
        samples=[0, 1, 2, 3],
        region=(0, 0, 1, 1),
        velocity=0.0,
        reference_phase=0.0,
        model_period=1.0,
        map_height=1,
        previous_belt_map=None,
        mask_threshold=5.0,
        mask_mode="positive",
        mask_grow_threshold=2.0,
        mask_dilation_px=0,
        mask_margin_px=0,
        mask_min_area_px=1,
        pass_label="test",
    )

    mean_map, mean_coverage = accumulate_belt_map(
        **common_kwargs,
        map_trim_fraction=0.0,
    )
    trimmed_map, trimmed_coverage = accumulate_belt_map(
        **common_kwargs,
        map_trim_fraction=0.25,
    )

    assert mean_coverage["contributed_pixels"] == 4
    assert trimmed_coverage == mean_coverage
    assert mean_map[0, 0] == pytest.approx(70.0)
    assert trimmed_map[0, 0] == pytest.approx(10.0)


@pytest.mark.parametrize("trim_fraction", [0.0, 0.1, 0.49])
def test_validate_map_trim_fraction_accepts_valid_fractions(trim_fraction):
    assert validate_map_trim_fraction(trim_fraction) == trim_fraction


@pytest.mark.parametrize("trim_fraction", [-0.01, 0.5, 1.0, np.nan])
def test_validate_map_trim_fraction_rejects_invalid_fractions(trim_fraction):
    with pytest.raises(ValueError, match="MAP_RECONSTRUCTION_TRIM_FRACTION"):
        validate_map_trim_fraction(trim_fraction)
