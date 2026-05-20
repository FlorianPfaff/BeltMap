import pytest

from beltmap import _driver_runtime as rt


def test_env_float_rejects_nan_and_inf(monkeypatch):
    monkeypatch.setenv("BELTMAP_TEST_FLOAT", "nan")
    with pytest.raises(ValueError, match="must be finite"):
        rt.env_float("BELTMAP_TEST_FLOAT", 1.0)

    monkeypatch.setenv("BELTMAP_TEST_FLOAT", "inf")
    with pytest.raises(ValueError, match="must be finite"):
        rt.env_float("BELTMAP_TEST_FLOAT", 1.0)


def test_image_paths_excludes_output_subdirectory(tmp_path, monkeypatch):
    data = tmp_path / "images"
    output = data / "outputs"
    output.mkdir(parents=True)
    (data / "frame_000.png").write_bytes(b"")
    (output / "residual_frame_000000.png").write_bytes(b"")

    monkeypatch.setattr(rt, "DATA", data)
    monkeypatch.setattr(rt, "OUT", output)
    monkeypatch.delenv("FRAME_STRIDE", raising=False)
    monkeypatch.delenv("MAX_FRAMES", raising=False)

    paths, discovered_frame_count, frame_stride = rt.image_paths()

    assert paths == [data / "frame_000.png"]
    assert discovered_frame_count == 1
    assert frame_stride == 1


def test_image_paths_rejects_output_directory_equal_to_image_directory(
    tmp_path,
    monkeypatch,
):
    data = tmp_path / "images"
    data.mkdir(parents=True)
    (data / "frame_000.png").write_bytes(b"")

    monkeypatch.setattr(rt, "DATA", data)
    monkeypatch.setattr(rt, "OUT", data)

    with pytest.raises(ValueError, match="must not be the same directory"):
        rt.image_paths()
