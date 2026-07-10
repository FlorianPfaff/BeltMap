from __future__ import annotations

import json
from pathlib import Path

from beltmap.cli import stream_snapshot as cli_stream_snapshot


def test_stream_snapshot_creates_state_parent_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    state_path = tmp_path / "nested" / "state" / "stream_state.json"

    monkeypatch.setattr(
        cli_stream_snapshot,
        "discover_new_stream_frames",
        lambda image_dir, state, *, max_new: [],
    )

    result = cli_stream_snapshot.main(
        [
            "--image-dir",
            str(image_dir),
            "--state",
            str(state_path),
        ]
    )

    assert result == 0
    assert state_path.parent.is_dir()
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload == {
        "seen_paths": [],
        "last_scan_unix_s": 0.0,
    }
