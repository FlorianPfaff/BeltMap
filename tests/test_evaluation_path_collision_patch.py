from __future__ import annotations

from pathlib import Path

import pytest

import beltmap  # noqa: F401
import beltmap.evaluation as evaluation


def test_patch_is_autoloaded() -> None:
    assert getattr(evaluation.write_evaluation, "_beltmap_evaluation_path_collision_patched", False)


def test_evaluation_rejects_shared_output_path_before_writing(tmp_path: Path) -> None:
    shared_path = tmp_path / "summary.out"
    shared_path.write_bytes(b"keep")
    with pytest.raises(ValueError, match="output paths must be distinct"):
        evaluation.write_evaluation(
            [],
            output_dir=tmp_path / "evaluation",
            json_path=shared_path,
            csv_path=shared_path,
            markdown_path=tmp_path / "summary.md",
        )
    assert shared_path.read_bytes() == b"keep"


def test_evaluation_rejects_output_aliasing_run_artifact(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    metadata_path = run_dir / "metadata.json"
    metadata_path.write_bytes(b"{}")
    with pytest.raises(ValueError, match="must not overwrite run artifact"):
        evaluation.write_evaluation(
            [evaluation.RunSpec(name="baseline", output_dir=run_dir)],
            output_dir=tmp_path / "evaluation",
            json_path=metadata_path,
            csv_path=tmp_path / "summary.csv",
            markdown_path=tmp_path / "summary.md",
        )
    assert metadata_path.read_bytes() == b"{}"


def test_evaluation_still_writes_distinct_default_outputs(tmp_path: Path) -> None:
    artifacts = evaluation.write_evaluation([], output_dir=tmp_path / "evaluation")
    assert artifacts.json_path.is_file()
    assert artifacts.csv_path.is_file()
    assert artifacts.markdown_path.is_file()
