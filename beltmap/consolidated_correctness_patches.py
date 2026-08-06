"""Load the substantive correctness fixes retained during PR cleanup."""

# Core numerical and telemetry invariants.
from . import runtime_json_patch as _runtime_json_patch  # noqa: F401
from . import operational_improvements_bilinear_border_patch as _bilinear_border_patch  # noqa: F401
from . import adaptive_sampling_coverage_patch as _adaptive_sampling_coverage_patch  # noqa: F401
from . import period_estimation_finite_overlap_patch as _period_estimation_patch  # noqa: F401

# Matching, benchmarking, and scientific reporting.
from . import cross_map_agreement_zero_sign_patch as _cross_map_zero_sign_patch  # noqa: F401
from . import benchmark_cardinality_matching_patch as _benchmark_cardinality_patch  # noqa: F401
from . import benchmark_event_matching_patch as _benchmark_event_patch  # noqa: F401
from . import benchmark_map_overlap_patch as _benchmark_map_overlap_patch  # noqa: F401
from . import trust_detection_drift_frame_patch as _trust_detection_drift_patch  # noqa: F401
from . import texture_stress_velocity_assignment_patch as _texture_stress_patch  # noqa: F401
from . import visual_qc_period_state_patch as _visual_qc_period_patch  # noqa: F401
from . import evaluation_path_collision_patch as _evaluation_collision_patch  # noqa: F401

# Period, recurrence, and tracking semantics.
from . import revolution_recurrence_circular_mean_patch as _circular_mean_patch  # noqa: F401
from . import revolution_split_observed_order_patch as _revolution_split_patch  # noqa: F401
from . import recurrent_artifact_default_patch as _recurrent_default_patch  # noqa: F401
from . import tracking_zero_lateral_gate_patch as _zero_lateral_gate_patch  # noqa: F401
from . import tracking_filter_row_patch as _tracking_filter_row_patch  # noqa: F401
from . import phase_estimate_reuse_validation_patch as _phase_reuse_patch  # noqa: F401
from . import map_only_period_state_patch as _map_only_period_patch  # noqa: F401

# Flux wrappers must load in this order: unit semantics first, finite-output guard second.
from . import flux_velocity_units_patch as _flux_velocity_units_patch  # noqa: F401
from . import flux_summary_finite_patch as _flux_summary_finite_patch  # noqa: F401
