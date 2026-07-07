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

# Import for side effect: keep subpixel phase-registration losses and scores bounded.
from . import phase_registration_loss_patch as _phase_registration_loss_patch  # noqa: F401,E402

# Import for side effect: make direct imports of beltmap.yolo_recurrence use the
# duplicate-safe YOLO detection key, not only the CLI wrapper path.
from . import yolo_recurrence_key_patch as _yolo_recurrence_key_patch  # noqa: F401,E402

# Import for side effect: make YOLO recurrence filtered runs keep
# detections_per_frame.csv consistent with the exported detections.csv rows.
from . import yolo_recurrence_per_frame_patch as _yolo_recurrence_per_frame_patch  # noqa: F401,E402

# Import for side effect: keep auxiliary detections_per_frame.csv columns when
# YOLO recurrence filtered runs rewrite frame-level detection counts.
from . import yolo_recurrence_per_frame_fields_patch as _yolo_recurrence_per_frame_fields_patch  # noqa: F401,E402

# Import for side effect: make YOLO recurrence metadata fall back when optional
# numeric fields are serialized as blank strings in legacy run metadata.
from . import yolo_recurrence_metadata_patch as _yolo_recurrence_metadata_patch  # noqa: F401,E402

# Import for side effect: make YOLO recurrence contact sheets focus on the
# scored detection and revisit patches instead of downscaled full crops.
from . import yolo_recurrence_contact_patch as _yolo_recurrence_contact_patch  # noqa: F401,E402

# Import for side effect: make YOLO export fail on ambiguous image directories
# instead of silently overwriting duplicate stems or duplicate frame indices.
from . import yolo_export_image_patch as _yolo_export_image_patch  # noqa: F401,E402

# Import for side effect: make the ghost objective understand the nested JSON
# schema written by beltmap-map-only-negative-control metrics files.
from . import ghost_objective_map_only_json_patch as _ghost_objective_map_only_json_patch  # noqa: F401,E402

# Import for side effect: make GhostRepair defect masks clip crop-local
# detection margins to the visible crop height recorded by map-only metrics.
from . import ghost_repair_crop_clip_patch as _ghost_repair_crop_clip_patch  # noqa: F401,E402

# Import for side effect: keep the packaged driver from turning inferred map
# support height into a cyclic belt period.
from . import driver_period_state_patch as _driver_period_state_patch  # noqa: F401,E402

__all__ = [
    "BeltMotionModel",
    "BeltPeriodState",
    "BeltRegion",
    "CleanBeltRender",
    "CrossMapAgreementConfig",
    "CrossMapAgreementMapScore",
    "CrossMapAgreementScore",
    "DETECTION_MODES",
    "PhaseEstimate",
    "PhaseRegistrationConfig",
    "PhaseTrajectorySmoothingConfig",
    "ParticleComponentConfig",
    "ParticleDetection",
    "ParticleTrack",
    "ParticleTrackScore",
    "ParticleTrackingConfig",
    "ParticleVelocity",
    "ResidualConfig",
    "ResidualImage",
    "RevolutionSplit",
    "RecurrentArtifactConfig",
    "RecurrentArtifactDetectionScore",
    "RecurrentArtifactMap",
    "TrackFilterConfig",
    "belt_revolution_indices",
    "build_revolution_split",
    "build_recurrent_artifact_map",
    "detect_particles_from_residual",
    "detect_particles_from_residual_hysteresis",
    "detection_signal_from_residual",
    "normalize_detection_mode",
    "estimate_particle_velocities_vs_belt",
    "estimate_local_noise",
    "estimate_phase",
    "extract_particle_detections",
    "extract_particle_velocities_vs_belt",
    "filter_detections_by_agreement",
    "filter_recurrent_artifact_detections",
    "filter_particle_velocities",
    "fresh_period_state",
    "generate_residual_image",
    "metadata_fields",
    "parse_revolution_indices",
    "phase_fraction_and_radians",
    "refine_phase_by_registration",
    "render_clean_belt_residual",
    "render_belt_view",
    "require_period_known",
    "reused_period_state",
    "smooth_phase_estimates",
    "render_expected_clean_belt",
    "score_cross_map_agreement",
    "score_particle_velocities",
    "score_recurrent_artifact_detections",
    "score_recurrent_artifact_detections_excluding_current_revolution",
    "track_particle_detections",
]
