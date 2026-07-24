from pathlib import Path

import pytest

from beltmap.cli.evaluate import _validate_distinct_output_paths, main


def test_evaluate_rejects_aliasing_output_paths_before_writing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    shared_path = report_dir / "summary.txt"
    shared_path.write_text("sentinel", encoding="utf-8")
    alias_path = report_dir / ".." / report_dir.name / shared_path.name
    markdown_path = report_dir / "summary.md"

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--run",
                str(run_dir),
                "--json-path",
                str(shared_path),
                "--csv-path",
                str(alias_path),
                "--markdown-path",
                str(markdown_path),
                "--quiet",
            ]
        )

    assert exc_info.value.code == 2
    assert "JSON and CSV evaluation outputs must use distinct paths" in capsys.readouterr().err
    assert shared_path.read_text(encoding="utf-8") == "sentinel"
    assert not markdown_path.exists()


def test_evaluate_default_output_paths_are_distinct(tmp_path: Path):
    output_dir = tmp_path / "evaluation"

    _validate_distinct_output_paths(
        output_dir=output_dir,
        json_path=None,
        csv_path=None,
        markdown_path=None,
    )


def test_evaluate_writes_all_default_artifacts(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    output_dir = tmp_path / "evaluation"

    assert main(["--run", str(run_dir), "--output-dir", str(output_dir), "--quiet"]) == 0
    assert (output_dir / "evaluation_summary.json").is_file()
    assert (output_dir / "evaluation_summary.csv").is_file()
    assert (output_dir / "evaluation_summary.md").is_file()
