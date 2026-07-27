from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import beltmap  # noqa: F401 - imports side-effect patches
from beltmap import operational_improvements as operational
import beltmap.timestamp_csv_validation_patch as timestamp_patch


def _write_timestamps(path: Path, rows: str) -> Path:
    path.write_text(f"frame_index,time_s\n{rows}", encoding="utf-8")
    return path


def test_timestamp_csv_validation_patch_is_autoloaded() -> None:
    assert getattr(
        operational.load_timestamps_csv,
        "_beltmap_timestamp_csv_validation_patched",
        False,
    )


def test_timestamp_csv_loads_finite_irregular_times(tmp_path: Path) -> None:
    path = _write_timestamps(tmp_path / "timestamps.csv", "0,0.0\n2,0.11\n")

    table = operational.load_timestamps_csv(path)

    assert table.time_for_frame(0) == 0.0
    assert table.time_for_frame(2) == 0.11


def test_timestamp_csv_rejects_duplicate_frame_indices(tmp_path: Path) -> None:
    path = _write_timestamps(tmp_path / "timestamps.csv", "1,0.1\n01,0.2\n")

    with pytest.raises(ValueError, match="duplicates frame index 1"):
        operational.load_timestamps_csv(path)


@pytest.mark.parametrize("timestamp", ["nan", "inf", "-inf"])
def test_timestamp_csv_rejects_nonfinite_times(
    tmp_path: Path,
    timestamp: str,
) -> None:
    path = _write_timestamps(tmp_path / "timestamps.csv", f"0,{timestamp}\n")

    with pytest.raises(ValueError, match="non-finite"):
        operational.load_timestamps_csv(path)


def test_timestamp_csv_rejects_missing_required_columns(tmp_path: Path) -> None:
    path = tmp_path / "timestamps.csv"
    path.write_text("frame,time\n0,0.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required column"):
        operational.load_timestamps_csv(path)


def test_timestamp_csv_validation_reload_preserves_original() -> None:
    original = getattr(
        operational.load_timestamps_csv,
        "_beltmap_original_load_timestamps_csv",
    )

    importlib.reload(timestamp_patch)
    importlib.reload(timestamp_patch)

    patched = operational.load_timestamps_csv
    assert getattr(patched, "_beltmap_original_load_timestamps_csv") is original
