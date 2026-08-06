import json
import os

import pytest

from beltmap.cli.suggest_label_frames import main


def test_suggest_label_frames_rejects_input_output_collision(tmp_path):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    source = output_dir / "detections_per_frame.csv"
    source.write_text("frame_index,n_detections\n0,1\n", encoding="utf-8")
    original = source.read_bytes()

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--output-dir",
                str(output_dir),
                "--frames",
                "1",
                "--output",
                str(source),
            ]
        )

    assert exc_info.value.code == 2
    assert source.read_bytes() == original


def test_suggest_label_frames_rejects_hard_link_to_input(tmp_path):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    source = output_dir / "phase_estimates.csv"
    source.write_text(
        "frame_index,correction_px,score\n0,0,1\n",
        encoding="utf-8",
    )
    alias = tmp_path / "label_plan.csv"
    try:
        os.link(source, alias)
    except OSError as exc:
        pytest.skip(f"hard links unavailable: {exc}")
    original = source.read_bytes()

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--output-dir",
                str(output_dir),
                "--frames",
                "1",
                "--output",
                str(alias),
            ]
        )

    assert exc_info.value.code == 2
    assert source.read_bytes() == original
    assert alias.read_bytes() == original


def test_suggest_label_frames_rejects_colliding_outputs(tmp_path):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    shared_output = tmp_path / "labels.csv"

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--output-dir",
                str(output_dir),
                "--frames",
                "1",
                "--output",
                str(shared_output),
                "--template-output",
                str(shared_output),
            ]
        )

    assert exc_info.value.code == 2
    assert not shared_output.exists()


def test_suggest_label_frames_writes_distinct_outputs(tmp_path):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    (output_dir / "metadata.json").write_text(
        json.dumps({"n_images": 1}),
        encoding="utf-8",
    )
    plan_output = tmp_path / "label_plan.csv"
    template_output = tmp_path / "label_template.csv"

    result = main(
        [
            "--output-dir",
            str(output_dir),
            "--frames",
            "1",
            "--empty-frames",
            "0",
            "--output",
            str(plan_output),
            "--template-output",
            str(template_output),
        ]
    )

    assert result == 0
    assert plan_output.is_file()
    assert template_output.is_file()
    assert plan_output.read_text(encoding="utf-8") != template_output.read_text(
        encoding="utf-8"
    )
