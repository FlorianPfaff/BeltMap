from __future__ import annotations

import numpy as np
from PIL import Image

import beltmap  # noqa: F401 - imports side-effect patches
import beltmap.trust as trust


def test_trust_image_discovery_patch_is_autoloaded() -> None:
    assert getattr(
        trust.image_paths,
        "_beltmap_trust_image_discovery_patched",
        False,
    )


def test_trust_reports_ignore_image_suffixed_directories(tmp_path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "frame_001.png").mkdir()

    real_image = image_dir / "frame_002.png"
    Image.fromarray(np.zeros((4, 5), dtype=np.uint8)).save(real_image)

    assert trust.image_paths(image_dir) == [real_image]

    sequence = trust.sequence_report(image_dir)
    assert sequence["n_images"] == 1
    assert sequence["first_image"] == str(real_image)
    assert sequence["last_image"] == str(real_image)

    quality = trust.quality_report(image_dir)
    assert quality["n_images"] == 1
    assert quality["sampled_frames"] == 1
    assert quality["frames"][0]["image"] == str(real_image)
