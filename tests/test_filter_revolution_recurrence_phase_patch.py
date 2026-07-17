from __future__ import annotations

import csv
from pathlib import Path

import pytest

from beltmap.cli import filter_revolution_recurrence as cli_filter


def write_phase_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_index", "phase_px"])
        writer.writeheader()
        writer.writerows(rows)


def test_phase_loader_rejects_duplicate_frame_rows(tmp_path: Path) -> None:
    phase_path = tmp_path / "phase_estimates.csv"
    write_phase_rows(
        phase_path,
        [
            {"frame_index": 0, "phase_px": 1.0},
            {"frame_index": 0, "phase_px": 9.0},
            {"frame_index": 1, "phase_px": 2.0},
        ],
    )

    assert getattr(
        cli_filter.load_phase_px_by_frame,
        "_beltmap_revolution_recurrence_unique_phase_rows_patched",
        False,
    )
    with pytest.raises(ValueError, match="duplicate phase estimate for frame 0"):
        cli_filter.load_phase_px_by_frame(phase_path, frame_count=2)


@pytest.mark.parametrize("frame_index", ["0.5", "nan", "inf", "-inf", "bad"])
def test_phase_loader_rejects_invalid_frame_identifiers(
    tmp_path: Path,
    frame_index: str,
) -> None:
    phase_path = tmp_path / "phase_estimates.csv"
    write_phase_rows(
        phase_path,
        [
            {"frame_index": frame_index, "phase_px": 1.0},
            {"frame_index": 1, "phase_px": 2.0},
        ],
    )

    with pytest.raises(ValueError, match="invalid frame_index"):
        cli_filter.load_phase_px_by_frame(phase_path, frame_count=2)


@pytest.mark.parametrize("phase_px", ["nan", "inf", "-inf", "bad", ""])
def test_phase_loader_rejects_invalid_in_range_phases(
    tmp_path: Path,
    phase_px: str,
) -> None:
    phase_path = tmp_path / "phase_estimates.csv"
    write_phase_rows(
        phase_path,
        [
            {"frame_index": 0, "phase_px": phase_px},
            {"frame_index": 1, "phase_px": 2.0},
        ],
    )

    with pytest.raises(ValueError, match="invalid phase_px"):
        cli_filter.load_phase_px_by_frame(phase_path, frame_count=2)


def test_phase_loader_preserves_integer_like_frame_values(tmp_path: Path) -> None:
    phase_path = tmp_path / "phase_estimates.csv"
    write_phase_rows(
        phase_path,
        [
            {"frame_index": "0.0", "phase_px": 1.5},
            {"frame_index": "1e0", "phase_px": 2.5},
        ],
    )

    assert cli_filter.load_phase_px_by_frame(phase_path, frame_count=2) == [1.5, 2.5]
