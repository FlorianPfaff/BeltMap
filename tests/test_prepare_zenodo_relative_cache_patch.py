from __future__ import annotations

from pathlib import Path

import pytest

from beltmap.cli import prepare_zenodo_dataset as prep


def test_expose_path_keeps_relative_cache_target_reachable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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


def test_expose_path_can_refresh_existing_exposure(
    tmp_path: Path,
) -> None:
    target = tmp_path / "cache" / "images_Test"
    target.mkdir(parents=True)
    (target / "frame.bmp").write_bytes(b"frame")
    link = tmp_path / "data" / "images"

    for _ in range(2):
        prep.expose_path(
            target=target,
            link=link,
            target_is_directory=True,
        )

    assert (link / "frame.bmp").read_bytes() == b"frame"


@pytest.mark.parametrize("target_is_directory", [False, True])
def test_expose_path_rejects_cache_target_itself_without_deleting_it(
    tmp_path: Path,
    target_is_directory: bool,
) -> None:
    target = tmp_path / ("images_Test" if target_is_directory else "images_Test.zip")
    if target_is_directory:
        target.mkdir()
        payload = target / "frame.bmp"
    else:
        payload = target
    payload.write_bytes(b"cache-payload")

    with pytest.raises(ValueError, match="must differ from its cache target"):
        prep.expose_path(
            target=target,
            link=target,
            target_is_directory=target_is_directory,
        )

    assert payload.read_bytes() == b"cache-payload"
