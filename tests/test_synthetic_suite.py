import json

from beltmap.cli.synthetic_suite import render_case


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
