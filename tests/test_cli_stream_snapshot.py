from __future__ import annotations

import json
from pathlib import Path

import pytest

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


def test_stream_snapshot_loads_valid_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    state_path = tmp_path / "stream_state.json"
    state_path.write_text(
        json.dumps(
            {
                "seen_paths": ["images/frame_000.png"],
                "last_scan_unix_s": 123.5,
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_discover(image_dir, state, *, max_new):
        captured["seen_paths"] = state.seen_paths.copy()
        captured["last_scan_unix_s"] = state.last_scan_unix_s
        return []

    monkeypatch.setattr(
        cli_stream_snapshot,
        "discover_new_stream_frames",
        fake_discover,
    )

    result = cli_stream_snapshot.main(
        ["--image-dir", str(image_dir), "--state", str(state_path)]
    )

    assert result == 0
    assert captured == {
        "seen_paths": {"images/frame_000.png"},
        "last_scan_unix_s": 123.5,
    }


def test_stream_snapshot_rejects_string_seen_paths(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    state_path = tmp_path / "stream_state.json"
    state_path.write_text(
        json.dumps(
            {
                "seen_paths": "images/frame_000.png",
                "last_scan_unix_s": 0.0,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="seen_paths"):
        cli_stream_snapshot.main(
            ["--image-dir", str(image_dir), "--state", str(state_path)]
        )


def test_stream_snapshot_rejects_nonfinite_last_scan(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    state_path = tmp_path / "stream_state.json"
    state_path.write_text(
        '{"seen_paths": [], "last_scan_unix_s": NaN}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="last_scan_unix_s"):
        cli_stream_snapshot.main(
            ["--image-dir", str(image_dir), "--state", str(state_path)]
        )
