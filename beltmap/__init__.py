"""Conveyor-belt map reconstruction and phase-estimation tools."""

from .detection import (
    DETECTION_MODES,
    detect_particles_from_residual,
    detect_particles_from_residual_hysteresis,
    detection_signal_from_residual,
    normalize_detection_mode,
)
from .cross_map_agreement import (
    CrossMapAgreementConfig,
    CrossMapAgreementMapScore,
    CrossMapAgreementScore,
    filter_detections_by_agreement,
    score_cross_map_agreement,
)
from .period_state import (
    BeltPeriodState,
    fresh_period_state,
    metadata_fields,
    phase_fraction_and_radians,
    require_period_known,
    reused_period_state,
)
from .phase import (
    BeltMotionModel,
    PhaseEstimate,
    PhaseRegistrationConfig,
    PhaseTrajectorySmoothingConfig,
    estimate_phase,
    refine_phase_by_registration,
    render_belt_view,
    smooth_phase_estimates,
)
from .rendering import (
    BeltRegion,
    CleanBeltRender,
    render_expected_clean_belt,
)
from .residual import (
    ResidualConfig,
    ResidualImage,
    estimate_local_noise,
    generate_residual_image,
    render_clean_belt_residual,
)
from .revolution_split import (
    RevolutionSplit,
    build_revolution_split,
    parse_revolution_indices,
)
from .recurrent_artifacts import (
    RecurrentArtifactConfig,
    RecurrentArtifactDetectionScore,
    RecurrentArtifactMap,
    belt_revolution_indices,
    build_recurrent_artifact_map,
    filter_recurrent_artifact_detections,
    score_recurrent_artifact_detections,
    score_recurrent_artifact_detections_excluding_current_revolution,
)
from .tracking import (
    ParticleComponentConfig,
    ParticleDetection,
    ParticleTrack,
    ParticleTrackScore,
    ParticleTrackingConfig,
    ParticleVelocity,
    TrackFilterConfig,
    estimate_particle_velocities_vs_belt,
    extract_particle_detections,
    extract_particle_velocities_vs_belt,
    filter_particle_velocities,
    score_particle_velocities,
    track_particle_detections,
)

# Import for side effect: make direct imports of beltmap.yolo_recurrence use the
# duplicate-safe YOLO detection key, not only the CLI wrapper path.
from . import yolo_recurrence_key_patch as _yolo_recurrence_key_patch  # noqa: F401,E402

# Import for side effect: make the ghost objective understand the nested JSON
# schema written by beltmap-map-only-negative-control metrics files.
from . import ghost_objective_map_only_json_patch as _ghost_objective_map_only_json_patch  # noqa: F401,E402
