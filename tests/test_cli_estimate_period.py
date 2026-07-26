from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from beltmap.cli.estimate_period import main


def _write_belt_map(path: Path) -> bytes:
    belt_map = np.arange(64, dtype=np.float64).reshape(16, 4)
    np.save(path, belt_map)
    return path.read_bytes()


def test_estimate_period_rejects_output_equal_to_belt_map(tmp_path: Path) -> None:
    belt_map_path = tmp_path / "belt_map.npy"
    original_bytes = _write_belt_map(belt_map_path)

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--belt-map",
                str(belt_map_path),
                "--output",
                str(belt_map_path),
            ]
        )

    assert exc_info.value.code == 2
    assert belt_map_path.read_bytes() == original_bytes


def test_estimate_period_rejects_hard_link_output_alias(tmp_path: Path) -> None:
    belt_map_path = tmp_path / "belt_map.npy"
    output_path = tmp_path / "period.json"
    original_bytes = _write_belt_map(belt_map_path)
    try:
        os.link(belt_map_path, output_path)
    except OSError as exc:  # pragma: no cover - platform/filesystem dependent
        pytest.skip(f"hard links are unavailable: {exc}")

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--belt-map",
                str(belt_map_path),
                "--output",
                str(output_path),
            ]
        )

    assert exc_info.value.code == 2
    assert belt_map_path.read_bytes() == original_bytes
    assert output_path.read_bytes() == original_bytes
