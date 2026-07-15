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
