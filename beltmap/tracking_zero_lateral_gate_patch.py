"""Allow exact-zero lateral-velocity gates in track scoring."""

from __future__ import annotations

from dataclasses import replace
import sys
from typing import Any, Sequence

from . import tracking as _tracking

_PATCHED_ATTR = "_beltmap_zero_lateral_velocity_gate_patched"
_ORIGINAL_ATTR = "_beltmap_original_score_particle_velocities"


def _unwrap_patched_callable(func: Any) -> Any:
    return getattr(func, _ORIGINAL_ATTR, func)


_original_score_particle_velocities = _unwrap_patched_callable(
    _tracking.score_particle_velocities
)


def score_particle_velocities_with_zero_lateral_gate(
    velocities: Sequence[_tracking.ParticleVelocity],
    *,
    config: _tracking.TrackFilterConfig | None = None,
    tracks: Sequence[_tracking.ParticleTrack] | None = None,
) -> list[_tracking.ParticleTrackScore]:
    cfg = config or _tracking.TrackFilterConfig()
    max_abs_x_velocity = _tracking._optional_finite_config_value(
        cfg.max_abs_x_velocity_px_per_frame,
        "max_abs_x_velocity_px_per_frame",
    )
    if max_abs_x_velocity != 0.0:
        return _original_score_particle_velocities(
            velocities,
            config=cfg,
            tracks=tracks,
        )

    unrestricted_cfg = replace(
        cfg,
        max_abs_x_velocity_px_per_frame=None,
    )
    unrestricted_scores = _original_score_particle_velocities(
        velocities,
        config=unrestricted_cfg,
        tracks=tracks,
    )

    patched_scores: list[_tracking.ParticleTrackScore] = []
    for score in unrestricted_scores:
        passes_lateral = score.abs_x_velocity_px_per_frame == 0.0
        patched_scores.append(
            replace(
                score,
                passes_lateral_velocity=passes_lateral,
                accepted=score.accepted and passes_lateral,
                plausibility_score=(
                    score.plausibility_score if passes_lateral else 0.0
                ),
            )
        )
    return patched_scores


setattr(score_particle_velocities_with_zero_lateral_gate, _PATCHED_ATTR, True)
setattr(
    score_particle_velocities_with_zero_lateral_gate,
    _ORIGINAL_ATTR,
    _original_score_particle_velocities,
)
_tracking.score_particle_velocities = score_particle_velocities_with_zero_lateral_gate

_package = sys.modules.get(__package__)
if _package is not None:
    setattr(
        _package,
        "score_particle_velocities",
        score_particle_velocities_with_zero_lateral_gate,
    )
