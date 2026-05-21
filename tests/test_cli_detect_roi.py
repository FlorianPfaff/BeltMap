import pytest

from beltmap.cli import detect_roi as cli_detect_roi


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--max-frames", "0"),
        ("--max-frames", "-1"),
        ("--max-frames", "1.5"),
        ("--percentile", "nan"),
        ("--percentile", "100"),
        ("--percentile", "-0.1"),
        ("--margin-px", "-1"),
        ("--margin-px", "1.5"),
    ],
)
def test_detect_roi_rejects_invalid_numeric_arguments(flag, value):
    with pytest.raises(SystemExit) as exc_info:
        cli_detect_roi.build_parser().parse_args(
            ["--image-dir", "images", flag, value]
        )

    assert exc_info.value.code == 2


def test_detect_roi_accepts_valid_numeric_arguments():
    args = cli_detect_roi.build_parser().parse_args(
        [
            "--image-dir",
            "images",
            "--max-frames",
            "3",
            "--percentile",
            "99.5",
            "--margin-px",
            "0",
        ]
    )

    assert args.max_frames == 3
    assert args.percentile == 99.5
    assert args.margin_px == 0
