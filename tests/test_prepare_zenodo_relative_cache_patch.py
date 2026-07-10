from __future__ import annotations

from pathlib import Path

from beltmap.cli import prepare_zenodo_dataset as prep


def test_expose_path_keeps_relative_cache_target_reachable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    target = Path("cache/images_Test")
    target.mkdir(parents=True)
    (target / "frame.bmp").write_bytes(b"frame")
    link = Path("data/images")

    prep.expose_path(
        target=target,
        link=link,
        target_is_directory=True,
    )

    assert (link / "frame.bmp").read_bytes() == b"frame"
    if link.is_symlink():
        assert link.resolve() == target.resolve()
