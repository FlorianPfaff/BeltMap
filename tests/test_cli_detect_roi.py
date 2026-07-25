from PIL import Image
import pytest

from beltmap.cli import detect_roi as cli_detect_roi
from beltmap.cli import stream_snapshot as cli_stream_snapshot


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


def test_detect_roi_rejects_output_aliasing_an_input_image(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "frame_000.png"
    Image.new("L", (8, 8), color=32).save(image_path)
    original_bytes = image_path.read_bytes()

    with pytest.raises(SystemExit, match="would overwrite it"):
        cli_detect_roi.main(
            [
                "--image-dir",
                str(image_dir),
                "--output",
                str(image_path),
            ]
        )

    assert image_path.read_bytes() == original_bytes


@pytest.mark.parametrize("value", ["-1", "1.5", "abc"])
def test_stream_snapshot_rejects_invalid_max_new(value):
    with pytest.raises(SystemExit) as exc_info:
        cli_stream_snapshot.build_parser().parse_args(
            ["--image-dir", "images", "--max-new", value]
        )

    assert exc_info.value.code == 2


@pytest.mark.parametrize("value", ["0", "3"])
def test_stream_snapshot_accepts_nonnegative_max_new(value):
    args = cli_stream_snapshot.build_parser().parse_args(
        ["--image-dir", "images", "--max-new", value]
    )

    assert args.max_new == int(value)
