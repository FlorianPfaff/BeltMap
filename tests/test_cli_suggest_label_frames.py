import pytest

from beltmap.cli import suggest_label_frames as cli_suggest_label_frames


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "abc"])
def test_suggest_label_frames_rejects_invalid_frame_count(value):
    with pytest.raises(SystemExit) as exc_info:
        cli_suggest_label_frames.build_parser().parse_args(["--frames", value])

    assert exc_info.value.code == 2


def test_suggest_label_frames_accepts_positive_frame_count():
    args = cli_suggest_label_frames.build_parser().parse_args(["--frames", "3"])

    assert args.frames == 3
