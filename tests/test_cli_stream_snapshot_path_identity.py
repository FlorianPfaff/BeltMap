from __future__ import annotations

import json
from pathlib import Path

from beltmap.cli import stream_snapshot as cli_stream_snapshot


def test_stream_snapshot_preserves_seen_frames_across_equivalent_image_roots(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    image_dir = Path("images")
    image_dir.mkdir()
    image_path = image_dir / "frame_000.png"
    image_path.write_bytes(b"stream frame")
    state_path = Path("stream_state.json")

    assert cli_stream_snapshot.main(
        ["--image-dir", str(image_dir), "--state", str(state_path)]
    ) == 0
    first_payload = json.loads(capsys.readouterr().out)
    assert first_payload["new_frames"] == [str(image_path.resolve())]

    assert cli_stream_snapshot.main(
        ["--image-dir", str(image_dir.resolve()), "--state", str(state_path)]
    ) == 0
    second_payload = json.loads(capsys.readouterr().out)
    assert second_payload["new_frames"] == []

    state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert state_payload["seen_paths"] == [str(image_path.resolve())]
