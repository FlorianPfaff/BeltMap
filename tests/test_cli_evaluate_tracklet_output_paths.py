from __future__ import annotations

from pathlib import Path

import pytest

from beltmap.cli.evaluate_tracklets import main


def test_evaluate_tracklets_refuses_to_overwrite_truth_input(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    truth_path = tmp_path / "truth.json"
    truth_path.write_text("truth sentinel\n", encoding="utf-8")
    prediction_path = tmp_path / "tracks.csv"
    prediction_path.write_text("track_id,frame_index\n", encoding="utf-8")
    alias_path = tmp_path / "reports" / ".." / truth_path.name

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--truth-path",
                str(truth_path),
                "--prediction-path",
                str(prediction_path),
                "--metrics-path",
                str(alias_path),
                "--quiet",
            ]
        )

    assert exc_info.value.code == 2
    assert "metrics output must not overwrite truth input" in capsys.readouterr().err
    assert truth_path.read_text(encoding="utf-8") == "truth sentinel\n"


def test_evaluate_tracklets_rejects_colliding_outputs_before_writing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    truth_path = tmp_path / "truth.json"
    prediction_path = tmp_path / "tracks.csv"
    shared_path = tmp_path / "shared-output.txt"
    shared_path.write_text("output sentinel\n", encoding="utf-8")
    alias_path = tmp_path / "nested" / ".." / shared_path.name

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--truth-path",
                str(truth_path),
                "--prediction-path",
                str(prediction_path),
                "--metrics-path",
                str(shared_path),
                "--report-path",
                str(alias_path),
                "--quiet",
            ]
        )

    assert exc_info.value.code == 2
    assert (
        "metrics output and report output must use distinct paths"
        in capsys.readouterr().err
    )
    assert shared_path.read_text(encoding="utf-8") == "output sentinel\n"
