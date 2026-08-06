from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

import beltmap  # noqa: F401 - imports side-effect patches
import beltmap.evaluation as evaluation
import beltmap.evaluation_path_collision_patch as collision_patch


def test_evaluation_path_collision_patch_is_autoloaded() -> None:
    assert getattr(
        evaluation.write_evaluation,
        "_beltmap_evaluation_path_collision_patched",
        False,
    )


def test_evaluation_rejects_shared_output_path_before_writing(tmp_path: Path) -> None:
    shared_path = tmp_path / "summary.out"
    markdown_path = tmp_path / "summary.md"
    original_bytes = b"keep this artifact"
    shared_path.write_bytes(original_bytes)

    with pytest.raises(ValueError, match="output paths must be distinct"):
        evaluation.write_evaluation(
            [],
            output_dir=tmp_path / "evaluation",
            json_path=shared_path,
            csv_path=shared_path,
            markdown_path=markdown_path,
        )

    assert shared_path.read_bytes() == original_bytes
    assert not markdown_path.exists()
    assert not (tmp_path / "evaluation").exists()


def test_evaluation_rejects_hard_link_output_alias(tmp_path: Path) -> None:
    json_path = tmp_path / "summary.json"
    csv_path = tmp_path / "summary.csv"
    markdown_path = tmp_path / "summary.md"
    original_bytes = b"hard-link sentinel"
    json_path.write_bytes(original_bytes)
    try:
        os.link(json_path, csv_path)
    except OSError as exc:  # pragma: no cover - platform/filesystem dependent
        pytest.skip(f"hard links are unavailable: {exc}")

    with pytest.raises(ValueError, match="output paths must be distinct"):
        evaluation.write_evaluation(
            [],
            output_dir=tmp_path / "evaluation",
            json_path=json_path,
            csv_path=csv_path,
            markdown_path=markdown_path,
        )

    assert json_path.read_bytes() == original_bytes
    assert csv_path.read_bytes() == original_bytes
    assert not markdown_path.exists()


def test_evaluation_rejects_output_aliasing_run_artifact(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    metadata_path = run_dir / "metadata.json"
    original_bytes = b"{}"
    metadata_path.write_bytes(original_bytes)

    with pytest.raises(ValueError, match="must not overwrite run artifact"):
        evaluation.write_evaluation(
            [evaluation.RunSpec(name="baseline", output_dir=run_dir)],
            output_dir=tmp_path / "evaluation",
            json_path=metadata_path,
            csv_path=tmp_path / "summary.csv",
            markdown_path=tmp_path / "summary.md",
        )

    assert metadata_path.read_bytes() == original_bytes
    assert not (tmp_path / "summary.csv").exists()
    assert not (tmp_path / "summary.md").exists()


def test_evaluation_still_writes_distinct_default_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "evaluation"

    artifacts = evaluation.write_evaluation([], output_dir=output_dir)

    assert artifacts.json_path == output_dir / "evaluation_summary.json"
    assert artifacts.csv_path == output_dir / "evaluation_summary.csv"
    assert artifacts.markdown_path == output_dir / "evaluation_summary.md"
    assert artifacts.json_path.is_file()
    assert artifacts.csv_path.is_file()
    assert artifacts.markdown_path.is_file()


def test_evaluation_path_collision_patch_reload_preserves_original(
    tmp_path: Path,
) -> None:
    original = getattr(
        evaluation.write_evaluation,
        "_beltmap_original_write_evaluation",
    )

    importlib.reload(collision_patch)
    importlib.reload(collision_patch)

    patched = evaluation.write_evaluation
    assert getattr(patched, "_beltmap_original_write_evaluation") is original

    shared_path = tmp_path / "shared"
    with pytest.raises(ValueError, match="output paths must be distinct"):
        patched(
            [],
            output_dir=tmp_path / "evaluation",
            json_path=shared_path,
            csv_path=shared_path,
            markdown_path=tmp_path / "summary.md",
        )
