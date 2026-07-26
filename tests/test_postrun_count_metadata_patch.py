from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path

import beltmap  # noqa: F401 - imports side-effect patches
import beltmap.postrun_count_metadata_patch as count_patch
from beltmap import postrun_improvements as postrun


def _write_detection_rows(path: Path, count: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_index", "area_px"])
        writer.writeheader()
        for frame_index in range(count):
            writer.writerow({"frame_index": frame_index, "area_px": 10})


def test_postrun_count_metadata_patch_is_autoloaded() -> None:
    assert getattr(
        postrun.metadata_count_or_rows,
        "_beltmap_nonnegative_metadata_count_patched",
        False,
    )


def test_metadata_count_or_rows_falls_back_for_negative_counts() -> None:
    rows = [{}, {}, {}]

    assert postrun.metadata_count_or_rows(
        {"n_detections": -4},
        "n_detections",
        rows,
    ) == len(rows)


def test_negative_detection_metadata_does_not_create_impossible_rejection_share(
    tmp_path: Path,
) -> None:
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "n_recurrent_artifact_rejected": 8,
                "n_detections": -5,
            }
        ),
        encoding="utf-8",
    )
    _write_detection_rows(tmp_path / "detections.csv", 8)

    flags = postrun.quality_flags_from_outputs(tmp_path)

    assert all(flag.code != "heavy_recurrent_filtering" for flag in flags)


def test_postrun_count_metadata_patch_reload_preserves_original() -> None:
    original = getattr(
        postrun.metadata_count_or_rows,
        "_beltmap_original_metadata_count_or_rows",
    )

    importlib.reload(count_patch)
    importlib.reload(count_patch)

    patched = postrun.metadata_count_or_rows
    assert getattr(
        patched,
        "_beltmap_original_metadata_count_or_rows",
    ) is original
    assert patched({"n_detections": -1}, "n_detections", [{}, {}]) == 2
