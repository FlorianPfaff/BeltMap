from __future__ import annotations

import csv
from pathlib import Path

from beltmap.advanced_quality import quality_flags


def _write_detection_counts(output_dir: Path, values: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "detections_per_frame.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_index", "n_detections"])
        writer.writeheader()
        for frame_index, value in enumerate(values):
            writer.writerow({"frame_index": frame_index, "n_detections": value})


def _flag_codes(output_dir: Path) -> set[str]:
    return {flag["code"] for flag in quality_flags(output_dir)["flags"]}


def test_quality_flags_ignore_unavailable_detection_counts(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    _write_detection_counts(
        output_dir,
        ["", "bad", "nan", "inf", " ", "missing", "20", "20", "20", "100"],
    )

    assert "detection_spikes" not in _flag_codes(output_dir)


def test_quality_flags_keep_real_zero_detection_frames(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    _write_detection_counts(
        output_dir,
        ["0", "0", "0", "0", "0", "0", "20", "20", "20", "100"],
    )

    assert "detection_spikes" in _flag_codes(output_dir)
