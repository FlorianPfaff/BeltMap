import numpy as np
from PIL import Image

from beltmap.trust import image_paths, sequence_report


def test_trust_image_scanner_ignores_image_suffixed_directories(tmp_path):
    fake_image = tmp_path / "frame_000.png"
    fake_image.mkdir()
    real_image = tmp_path / "frame_001.png"
    Image.fromarray(np.zeros((4, 5), dtype=np.uint8)).save(real_image)

    assert image_paths(tmp_path) == [real_image]

    report = sequence_report(tmp_path)
    assert report["n_images"] == 1
    assert report["numbered_frames"] == 1
    assert report["missing_frame_numbers"] == []
    assert report["first_image"] == str(real_image)
