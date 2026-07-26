from __future__ import annotations

import beltmap  # noqa: F401 - imports side-effect patches
from beltmap.cli import validate as cli_validate


def test_validation_progress_jsonl_patch_is_autoloaded() -> None:
    assert getattr(
        cli_validate.read_progress_jsonl,
        "_beltmap_validation_progress_objects_patched",
        False,
    )


def test_validation_ignores_non_object_jsonl_records(tmp_path) -> None:
    progress_path = tmp_path / "progress.jsonl"
    progress_path.write_text(
        "\n".join(
            [
                "null",
                "[]",
                "42",
                '"status"',
                "not valid json",
                '{"stage":"belt_map","observed_pixels":7}',
                "",
            ]
        ),
        encoding="utf-8",
    )

    rows = cli_validate.read_progress_jsonl(progress_path)

    assert rows == [{"stage": "belt_map", "observed_pixels": 7}]
    assert cli_validate.final_belt_map_progress(rows) == rows[0]
