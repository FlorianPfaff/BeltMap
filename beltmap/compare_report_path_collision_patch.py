"""Prevent comparison reports from overwriting their own inputs."""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from . import compare_runs as _compare_runs

_PATCHED_ATTR = "_beltmap_compare_path_collision_patched"
_ORIGINAL_ATTR = "_beltmap_original_generate_comparison_report"

_STANDARD_RUN_INPUT_FILES = (
    "metadata.json",
    "config_resolved.json",
    "detections.csv",
    "detections_per_frame.csv",
    "velocities.csv",
    "filtered_velocities.csv",
    "filtered_tracks.csv",
)
_PREVIEW_GLOBS = (
    "residual_frame_*.png",
    "residual_fixed_frame_*.png",
    "raw_frame_*.png",
)


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the original report generator behind this compatibility patch."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_generate_comparison_report = _unwrap_patched_callable(
    _compare_runs.generate_comparison_report
)


def _paths_refer_to_same_file(first: Path, second: Path) -> bool:
    """Return whether two path spellings identify the same filesystem object."""

    try:
        if first.resolve(strict=False) == second.resolve(strict=False):
            return True
    except (OSError, RuntimeError):
        pass

    try:
        return first.samefile(second)
    except (FileNotFoundError, OSError, RuntimeError):
        return False


def _has_regular_preview(specs: Iterable[_compare_runs.RunSpec], pattern: str) -> bool:
    """Return whether any compared run contains a regular matching preview file."""

    return any(
        path.is_file()
        for spec in specs
        for path in Path(spec.output_dir).glob(pattern)
    )


def _generated_output_paths(
    specs: list[_compare_runs.RunSpec],
    *,
    report_dir: Path,
    truth_path: Path | None,
    make_metric_plots: bool,
    make_contact_sheets: bool,
) -> dict[str, Path]:
    """Resolve every file that the comparison call can write."""

    outputs = {
        "summary_csv": report_dir / "summary.csv",
        "markdown_report": report_dir / "comparison_report.md",
    }
    if make_metric_plots:
        outputs.update(
            {
                "detection_count_plot": report_dir
                / "detections_per_frame_comparison.png",
                "velocity_histogram": report_dir
                / "velocity_ratio_histogram_comparison.png",
            }
        )
        if truth_path is not None:
            outputs["labeled_froc_plot"] = report_dir / "labeled_detection_froc.png"

    if make_contact_sheets:
        outputs.update(
            {
                "detection_contact_sheet": report_dir
                / "detection_contact_sheet.png",
                "filtered_detection_contact_sheet": report_dir
                / "filtered_detection_contact_sheet.png",
            }
        )
        if _has_regular_preview(specs, "residual_fixed_frame_*.png"):
            outputs.update(
                {
                    "fixed_detection_contact_sheet": report_dir
                    / "fixed_scale_detection_contact_sheet.png",
                    "fixed_filtered_detection_contact_sheet": report_dir
                    / "fixed_scale_filtered_detection_contact_sheet.png",
                }
            )
        if _has_regular_preview(specs, "raw_frame_*.png"):
            outputs.update(
                {
                    "raw_detection_contact_sheet": report_dir
                    / "raw_detection_contact_sheet.png",
                    "raw_filtered_detection_contact_sheet": report_dir
                    / "raw_filtered_detection_contact_sheet.png",
                }
            )
    return outputs


def _comparison_input_paths(
    specs: list[_compare_runs.RunSpec],
    *,
    truth_path: Path | None,
) -> list[tuple[str, Path]]:
    """Collect explicit truth and run artifacts consumed by a comparison."""

    inputs: list[tuple[str, Path]] = []
    if truth_path is not None:
        inputs.append(("truth labels", truth_path))

    for spec in specs:
        run_dir = Path(spec.output_dir)
        for filename in _STANDARD_RUN_INPUT_FILES:
            inputs.append((f"run artifact {spec.label!r}/{filename}", run_dir / filename))
        for pattern in _PREVIEW_GLOBS:
            inputs.extend(
                (f"run preview {spec.label!r}/{path.name}", path)
                for path in run_dir.glob(pattern)
                if path.is_file()
            )
    return inputs


def _validate_comparison_paths(
    specs: list[_compare_runs.RunSpec],
    *,
    report_dir: Path,
    truth_path: Path | None,
    make_metric_plots: bool,
    make_contact_sheets: bool,
) -> None:
    outputs = _generated_output_paths(
        specs,
        report_dir=report_dir,
        truth_path=truth_path,
        make_metric_plots=make_metric_plots,
        make_contact_sheets=make_contact_sheets,
    )

    for (first_name, first_path), (second_name, second_path) in combinations(
        outputs.items(), 2
    ):
        if _paths_refer_to_same_file(first_path, second_path):
            raise ValueError(
                "comparison output paths must be distinct: "
                f"{first_name} and {second_name} refer to the same file"
            )

    for input_name, input_path in _comparison_input_paths(specs, truth_path=truth_path):
        for output_name, output_path in outputs.items():
            if _paths_refer_to_same_file(output_path, input_path):
                raise ValueError(
                    f"comparison output {output_name!r} must not overwrite input "
                    f"{input_name} at {input_path}"
                )


def generate_comparison_report_without_path_collisions(
    specs: list[_compare_runs.RunSpec],
    *,
    report_dir: Path,
    frames: list[int] | None = None,
    truth_path: Path | None = None,
    truth_iou_threshold: float = 0.25,
    froc_max_thresholds: int | None = _compare_runs.DEFAULT_FROC_MAX_THRESHOLDS,
    bootstrap_samples: int = 0,
    bootstrap_confidence_level: float = 0.95,
    bootstrap_seed: int | None = 0,
    bootstrap_block_length_frames: int = 1,
    make_metric_plots: bool = True,
    make_contact_sheets: bool = True,
) -> _compare_runs.ComparisonArtifacts:
    """Generate a comparison only after validating every destination path."""

    materialized_specs = list(specs)
    resolved_report_dir = Path(report_dir)
    resolved_truth_path = None if truth_path is None else Path(truth_path)
    _validate_comparison_paths(
        materialized_specs,
        report_dir=resolved_report_dir,
        truth_path=resolved_truth_path,
        make_metric_plots=make_metric_plots,
        make_contact_sheets=make_contact_sheets,
    )
    return _original_generate_comparison_report(
        materialized_specs,
        report_dir=resolved_report_dir,
        frames=frames,
        truth_path=resolved_truth_path,
        truth_iou_threshold=truth_iou_threshold,
        froc_max_thresholds=froc_max_thresholds,
        bootstrap_samples=bootstrap_samples,
        bootstrap_confidence_level=bootstrap_confidence_level,
        bootstrap_seed=bootstrap_seed,
        bootstrap_block_length_frames=bootstrap_block_length_frames,
        make_metric_plots=make_metric_plots,
        make_contact_sheets=make_contact_sheets,
    )


setattr(generate_comparison_report_without_path_collisions, _PATCHED_ATTR, True)
setattr(
    generate_comparison_report_without_path_collisions,
    _ORIGINAL_ATTR,
    _original_generate_comparison_report,
)
_compare_runs.generate_comparison_report = (
    generate_comparison_report_without_path_collisions
)

_cli_compare = sys.modules.get("beltmap.cli.compare")
if _cli_compare is not None:
    setattr(
        _cli_compare,
        "generate_comparison_report",
        generate_comparison_report_without_path_collisions,
    )
