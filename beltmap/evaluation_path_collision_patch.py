"""Prevent evaluation reports from overwriting inputs or sibling outputs."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from . import evaluation as _evaluation

_PATCHED_ATTR = "_beltmap_evaluation_path_collision_patched"
_ORIGINAL_ATTR = "_beltmap_original_write_evaluation"


def _unwrap_patched_callable(func: Any) -> Any:
    return getattr(func, _ORIGINAL_ATTR, func)


_original_write_evaluation = _unwrap_patched_callable(_evaluation.write_evaluation)


def _paths_refer_to_same_file(first: Path, second: Path) -> bool:
    try:
        if first.resolve(strict=False) == second.resolve(strict=False):
            return True
    except (OSError, RuntimeError):
        pass
    try:
        return first.samefile(second)
    except (FileNotFoundError, OSError, RuntimeError):
        return False


def _resolved_output_paths(*, output_dir: Path, json_path: Path | None, csv_path: Path | None, markdown_path: Path | None) -> dict[str, Path]:
    output_dir = Path(output_dir)
    return {
        "json": Path(json_path) if json_path is not None else output_dir / "evaluation_summary.json",
        "csv": Path(csv_path) if csv_path is not None else output_dir / "evaluation_summary.csv",
        "markdown": Path(markdown_path) if markdown_path is not None else output_dir / "evaluation_summary.md",
    }


def _validate_output_paths(runs: list[_evaluation.RunSpec], output_paths: dict[str, Path]) -> None:
    for (first_name, first_path), (second_name, second_path) in combinations(output_paths.items(), 2):
        if _paths_refer_to_same_file(first_path, second_path):
            raise ValueError(
                "evaluation output paths must be distinct: "
                f"{first_name} and {second_name} refer to the same file"
            )
    for run in runs:
        run_dir = Path(run.output_dir)
        for artifact_name in _evaluation.STANDARD_OUTPUT_FILES:
            artifact_path = run_dir / artifact_name
            for output_name, output_path in output_paths.items():
                if _paths_refer_to_same_file(output_path, artifact_path):
                    raise ValueError(
                        f"evaluation {output_name} output must not overwrite run artifact {artifact_path}"
                    )


def write_evaluation_without_path_collisions(
    runs: Iterable[_evaluation.RunSpec],
    *,
    output_dir: Path,
    json_path: Path | None = None,
    csv_path: Path | None = None,
    markdown_path: Path | None = None,
) -> _evaluation.EvaluationArtifacts:
    run_list = list(runs)
    resolved_paths = _resolved_output_paths(
        output_dir=output_dir,
        json_path=json_path,
        csv_path=csv_path,
        markdown_path=markdown_path,
    )
    _validate_output_paths(run_list, resolved_paths)
    return _original_write_evaluation(
        run_list,
        output_dir=Path(output_dir),
        json_path=resolved_paths["json"],
        csv_path=resolved_paths["csv"],
        markdown_path=resolved_paths["markdown"],
    )


setattr(write_evaluation_without_path_collisions, _PATCHED_ATTR, True)
setattr(write_evaluation_without_path_collisions, _ORIGINAL_ATTR, _original_write_evaluation)
_evaluation.write_evaluation = write_evaluation_without_path_collisions
