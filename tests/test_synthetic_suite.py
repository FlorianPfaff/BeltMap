import json
import sys

from beltmap.cli import synthetic_suite
from beltmap.cli.synthetic_suite import main, render_case


def test_render_case_splits_event_ids_at_particle_wraps(tmp_path):
    root = tmp_path / "synthetic"

    render_case("baseline", root, frames=28, height=16, width=24, period=16, seed=4)

    metadata = json.loads((root / "synthetic_metadata.json").read_text(encoding="utf-8"))
    boxes = [box for frame in metadata["frames"] for box in frame["boxes"]]
    event_ids = {box["event_id"] for box in boxes}

    assert metadata["height"] == 16
    assert metadata["width"] == 24
    assert len(event_ids) > 1
    assert all(":" in event_id for event_id in event_ids)


def test_execute_uses_current_python_modules(tmp_path, monkeypatch):
    calls = []

    def fake_run(command, *, check):
        calls.append(command)
        assert check is True

    monkeypatch.setattr(synthetic_suite.subprocess, "run", fake_run)

    result = main(
        [
            "--output-root",
            str(tmp_path / "suite"),
            "--case",
            "baseline",
            "--frames",
            "2",
            "--height",
            "16",
            "--width",
            "24",
            "--period",
            "16",
            "--execute",
        ]
    )

    assert result == 0
    assert calls[0][:3] == [sys.executable, "-m", "beltmap.cli.apply"]
    assert calls[1][:3] == [sys.executable, "-m", "beltmap.cli.benchmark"]
