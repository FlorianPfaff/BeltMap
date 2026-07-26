import numpy as np
import pytest

from beltmap.cli import suggest_threshold as cli_suggest_threshold


def test_suggest_threshold_rejects_output_aliasing_residual_input(tmp_path):
    residual_path = tmp_path / "residual.npy"
    np.save(residual_path, np.arange(9, dtype=np.float64).reshape(3, 3))
    original_bytes = residual_path.read_bytes()

    with pytest.raises(SystemExit, match="would overwrite it"):
        cli_suggest_threshold.main(
            [
                "--residual-npy",
                str(residual_path),
                "--output",
                str(residual_path),
            ]
        )

    assert residual_path.read_bytes() == original_bytes


def test_suggest_threshold_rejects_hard_link_output_alias(tmp_path):
    residual_path = tmp_path / "residual.npy"
    output_path = tmp_path / "threshold.json"
    np.save(residual_path, np.arange(9, dtype=np.float64).reshape(3, 3))
    output_path.hardlink_to(residual_path)
    original_bytes = residual_path.read_bytes()

    with pytest.raises(SystemExit, match="would overwrite it"):
        cli_suggest_threshold.main(
            [
                "--residual-npy",
                str(residual_path),
                "--output",
                str(output_path),
            ]
        )

    assert residual_path.read_bytes() == original_bytes
    assert output_path.read_bytes() == original_bytes
