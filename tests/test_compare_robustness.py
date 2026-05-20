import pytest

from beltmap.cli import compare as cli_compare
from beltmap.compare_runs import paired_values


def test_paired_values_keeps_detection_count_series_aligned():
    rows = [
        {"frame_index": "", "n_detections": "2"},
        {"frame_index": "3", "n_detections": "not-a-number"},
        {"frame_index": "4", "n_detections": "5"},
    ]

    xs, ys = paired_values(rows, x_field="frame_index", y_field="n_detections")

    assert xs == [0.0, 4.0]
    assert ys == [2.0, 5.0]


def test_compare_main_reports_parse_errors_without_traceback(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli_compare.main(
            [
                "--run",
                "=missing-label",
                "--run",
                "other=missing-output-dir",
            ]
        )

    assert exc_info.value.code == 2
    assert "run label must not be empty" in capsys.readouterr().err
