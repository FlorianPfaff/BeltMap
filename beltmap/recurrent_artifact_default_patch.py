"""Give recurrent-artifact map construction a valid implicit configuration."""

from __future__ import annotations

import sys
from typing import Any, Sequence

from . import recurrent_artifacts as _recurrent_artifacts

_PATCHED_ATTR = "_beltmap_recurrent_artifact_default_patched"
_ORIGINAL_ATTR = "_beltmap_original_build_recurrent_artifact_map"


def _unwrap_patched_callable(func: Any) -> Any:
    return getattr(func, _ORIGINAL_ATTR, func)


_original_build_recurrent_artifact_map = _unwrap_patched_callable(
    _recurrent_artifacts.build_recurrent_artifact_map
)


def build_recurrent_artifact_map_with_valid_default(
    detections_by_frame: Sequence[
        Sequence[_recurrent_artifacts.ParticleDetection]
    ],
    phase_px_by_frame: Sequence[float],
    revolution_by_frame: Sequence[int],
    *,
    map_shape: tuple[int, int],
    config: _recurrent_artifacts.RecurrentArtifactConfig | None = None,
    frame_shape: tuple[int, int] | None = None,
) -> _recurrent_artifacts.RecurrentArtifactMap:
    if config is None:
        config = _recurrent_artifacts.RecurrentArtifactConfig(min_revolutions=1)
    return _original_build_recurrent_artifact_map(
        detections_by_frame,
        phase_px_by_frame,
        revolution_by_frame,
        map_shape=map_shape,
        config=config,
        frame_shape=frame_shape,
    )


setattr(build_recurrent_artifact_map_with_valid_default, _PATCHED_ATTR, True)
setattr(
    build_recurrent_artifact_map_with_valid_default,
    _ORIGINAL_ATTR,
    _original_build_recurrent_artifact_map,
)
_recurrent_artifacts.build_recurrent_artifact_map = (
    build_recurrent_artifact_map_with_valid_default
)

_package = sys.modules.get(__package__)
if _package is not None:
    setattr(
        _package,
        "build_recurrent_artifact_map",
        build_recurrent_artifact_map_with_valid_default,
    )
