from __future__ import annotations

from pathlib import Path

from beltmap import evaluation
from beltmap.cli.evaluate import main as evaluate_main


def test_evaluation_markdown_escaping_patch_is_autoloaded() -> None:
    assert getattr(
        evaluation.build_markdown,
        "_beltmap_evaluation_markdown_table_escaping_patched",
        False,
    )


def test_cli_evaluation_escapes_table_delimiters_and_line_breaks(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "evaluation"
    run_dir.mkdir()

    status = evaluate_main(
        [
            "--run",
            f"baseline|stress\n<script>={run_dir}",
            "--output-dir",
            str(output_dir),
            "--quiet",
        ]
    )

    assert status == 0
    report = (output_dir / "evaluation_summary.md").read_text(encoding="utf-8")
    row = next(line for line in report.splitlines() if "<code>baseline" in line)
    assert "<code>baseline&#124;stress<br>&lt;script&gt;</code>" in row
    assert row.count("|") == 11
