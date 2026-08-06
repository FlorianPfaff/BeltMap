from __future__ import annotations

import numpy as np

import beltmap
from beltmap import ParticleDetection, RecurrentArtifactConfig
from beltmap import recurrent_artifacts


def detection() -> ParticleDetection:
    return ParticleDetection(
        frame_index=0.0,
        label=1,
        y=1.5,
        x=1.5,
        area_px=1,
        bbox_top=1,
        bbox_left=1,
        bbox_bottom=2,
        bbox_right=2,
        peak_signal=3.0,
    )


def test_recurrent_artifact_default_patch_is_autoloaded() -> None:
    assert getattr(
        beltmap.build_recurrent_artifact_map,
        "_beltmap_recurrent_artifact_default_patched",
        False,
    )
    assert (
        recurrent_artifacts.build_recurrent_artifact_map
        is beltmap.build_recurrent_artifact_map
    )


def test_recurrent_artifact_map_can_use_omitted_config() -> None:
    result = beltmap.build_recurrent_artifact_map(
        [[detection()]],
        phase_px_by_frame=[0.0],
        revolution_by_frame=[0],
        map_shape=(4, 4),
    )

    assert result.revolution_count == 1
    assert result.counts[1, 1] == 1
    assert result.mask[1, 1]


def test_recurrent_artifact_map_preserves_explicit_threshold() -> None:
    result = beltmap.build_recurrent_artifact_map(
        [[detection()]],
        phase_px_by_frame=[0.0],
        revolution_by_frame=[0],
        map_shape=(4, 4),
        config=RecurrentArtifactConfig(
            min_revolutions=2,
            margin_px=0,
        ),
    )

    np.testing.assert_array_equal(result.counts[1:2, 1:2], [[1]])
    assert not result.mask[1, 1]
