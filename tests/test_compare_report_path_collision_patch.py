from __future__ import annotations

import csv
import importlib
import os
from pathlib import Path

import pytest

import beltmap  # noqa: F401 - imports side-effect patches
import beltmap.cli.compare as cli_compare
import beltmap.compare_report_path_collision_patch as collision_patch
import beltmap.compare_runs as compare_runs


def _write_empty_frame_truth(path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_index",
                "bbox_top",
                "bbox_left",
                "bbox_bottom",
                "bbox_right",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "frame_index": 0,
                "bbox_top": "",
                "bbox_left": "",
                "bbox_bottom": "",
                "bbox_right": "",
            }
        )
    return path.read_bytes()


def _empty_run(tmp_path: Path) -> compare_runs.RunSpec:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    return compare_runs.RunSpec("empty", run_dir)


def test_comparison_path_collision_patch_is_autoloaded() -> None:
    assert getattr(
        compare_runs.generate_comparison_report,
        "_beltmap_compare_path_collision_patched",
        False,
    )
    assert cli_compare.generate_comparison_report is compare_runs.generate_comparison_report


def test_comparison_rejects_truth_named_summary_csv_before_writing(
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "comparison"
    truth_path = report_dir / "summary.csv"
    original_bytes = _write_empty_frame_truth(truth_path)

    with pytest.raises(ValueError, match="must not overwrite input truth labels"):
        compare_runs.generate_comparison_report(
            [_empty_run(tmp_path)],
            report_dir=report_dir,
            truth_path=truth_path,
            make_metric_plots=False,
            make_contact_sheets=False,
        )

    assert truth_path.read_bytes() == original_bytes
    assert not (report_dir / "comparison_report.md").exists()


def test_comparison_rejects_hard_link_alias_to_truth(tmp_path: Path) -> None:
    truth_path = tmp_path / "labels.csv"
    original_bytes = _write_empty_frame_truth(truth_path)
    report_dir = tmp_path / "comparison"
    report_dir.mkdir()
    summary_path = report_dir / "summary.csv"
    try:
        os.link(truth_path, summary_path)
    except OSError as exc:  # pragma: no cover - platform/filesystem dependent
        pytest.skip(f"hard links are unavailable: {exc}")

    with pytest.raises(ValueError, match="must not overwrite input truth labels"):
        compare_runs.generate_comparison_report(
            [_empty_run(tmp_path)],
            report_dir=report_dir,
            truth_path=truth_path,
            make_metric_plots=False,
            make_contact_sheets=False,
        )

    assert truth_path.read_bytes() == original_bytes
    assert summary_path.read_bytes() == original_bytes
    assert not (report_dir / "comparison_report.md").exists()


def test_comparison_rejects_aliasing_sibling_outputs(tmp_path: Path) -> None:
    truth_path = tmp_path / "labels.csv"
    _write_empty_frame_truth(truth_path)
    report_dir = tmp_path / "comparison"
    report_dir.mkdir()
    summary_path = report_dir / "summary.csv"
    report_path = report_dir / "comparison_report.md"
    original_bytes = b"preserve existing report artifacts"
    summary_path.write_bytes(original_bytes)
    try:
        os.link(summary_path, report_path)
    except OSError as exc:  # pragma: no cover - platform/filesystem dependent
        pytest.skip(f"hard links are unavailable: {exc}")

    with pytest.raises(ValueError, match="output paths must be distinct"):
        compare_runs.generate_comparison_report(
            [_empty_run(tmp_path)],
            report_dir=report_dir,
            truth_path=truth_path,
            make_metric_plots=False,
            make_contact_sheets=False,
        )

    assert summary_path.read_bytes() == original_bytes
    assert report_path.read_bytes() == original_bytes


def test_comparison_still_writes_distinct_outputs(tmp_path: Path) -> None:
    truth_path = tmp_path / "labels.csv"
    original_bytes = _write_empty_frame_truth(truth_path)
    report_dir = tmp_path / "comparison"

    artifacts = compare_runs.generate_comparison_report(
        [_empty_run(tmp_path)],
        report_dir=report_dir,
        truth_path=truth_path,
        make_metric_plots=False,
        make_contact_sheets=False,
    )

    assert artifacts.summary_csv == report_dir / "summary.csv"
    assert artifacts.report == report_dir / "comparison_report.md"
    assert artifacts.summary_csv.is_file()
    assert artifacts.report.is_file()
    assert truth_path.read_bytes() == original_bytes


def test_comparison_path_collision_patch_reload_preserves_original() -> None:
    original = getattr(
        compare_runs.generate_comparison_report,
        "_beltmap_original_generate_comparison_report",
    )

    importlib.reload(collision_patch)
    importlib.reload(collision_patch)

    patched = compare_runs.generate_comparison_report
    assert getattr(patched, "_beltmap_original_generate_comparison_report") is original
    assert cli_compare.generate_comparison_report is patched
