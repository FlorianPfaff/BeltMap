"""Shared CSV output schemas for BeltMap run artifacts.

This module is a small first step toward splitting the monolithic driver output
surface into focused modules.  Keep field order stable: downstream comparison,
filtering, and paper-evidence scripts treat these CSV headers as part of the
run-output contract.
"""

from __future__ import annotations

DETECTIONS_PER_FRAME_FIELDS = ["frame_index", "n_detections"]

DRIVER_DETECTION_FIELDS = [
    "frame_index",
    "image",
    "label",
    "y",
    "x",
    "area_px",
    "bbox_top",
    "bbox_left",
    "bbox_bottom",
    "bbox_right",
    "mean_signal",
    "peak_signal",
    "map_support_min",
    "map_support_mean",
    "map_risk_mean",
    "map_risk_max",
    "map_interpolated_fraction",
    "map_low_support_fraction",
    "recurrent_artifact_overlap_fraction",
    "recurrent_artifact_probability",
    "recurrent_artifact_required_peak_signal",
]

YOLO_DETECTION_FIELDS = [
    "frame_index",
    "label",
    "y",
    "x",
    "area_px",
    "bbox_top",
    "bbox_left",
    "bbox_bottom",
    "bbox_right",
    "score",
    "confidence",
    "class_id",
    "source",
]

TRACK_DETECTION_FIELDS = [
    "track_id",
    "track_detection_index",
    *DRIVER_DETECTION_FIELDS,
]

RECURRENT_ARTIFACT_DETECTION_FIELDS = [
    *DRIVER_DETECTION_FIELDS,
    "recurrent_artifact_rejected",
]

MAP_RISK_DETECTION_FIELDS = [
    *DRIVER_DETECTION_FIELDS,
    "map_risk_rejected",
]

CROSS_MAP_AGREEMENT_FIELDS = [
    *DRIVER_DETECTION_FIELDS,
    "cross_map_agreement_accepted",
    "cross_map_agreement_confirming_maps",
]

VELOCITY_FIELDS = [
    "track_id",
    "n_detections",
    "frame_start",
    "frame_end",
    "velocity_y_px_per_frame",
    "velocity_x_px_per_frame",
    "speed_px_per_frame",
    "belt_velocity_y_px_per_frame",
    "velocity_ratio_y",
    "belt_minus_particle_velocity_y_px_per_frame",
]

TRACK_SCORE_FIELDS = [
    "track_id",
    "n_detections",
    "frame_start",
    "frame_end",
    "velocity_y_px_per_frame",
    "velocity_x_px_per_frame",
    "velocity_ratio_y",
    "abs_x_velocity_px_per_frame",
    "passes_min_track_length",
    "passes_velocity_ratio",
    "passes_lateral_velocity",
    "n_recurrent_artifact_scored_detections",
    "mean_recurrent_artifact_overlap_fraction",
    "max_recurrent_artifact_overlap_fraction",
    "mean_recurrent_artifact_probability",
    "max_recurrent_artifact_probability",
    "recurrent_artifact_hit_fraction",
    "recurrent_artifact_track_score",
    "passes_recurrent_artifact",
    "accepted",
    "plausibility_score",
]

PHASE_FIELDS = [
    "frame_index",
    "image",
    "phase_px",
    "phase_fraction",
    "phase_rad",
    "predicted_phase_px",
    "correction_px",
    "phase_drift_px",
    "loss",
    "score",
    "second_best_loss",
    "loss_gap",
    "loss_gap_ratio",
    "loss_curvature",
    "uncertainty_px",
    "method",
]

PHOTOMETRIC_FIELDS = [
    "frame_index",
    "image",
    "gain",
    "offset",
    "n_pixels",
    "rmse_gray",
    "trimmed_fraction",
    "status",
]

REVOLUTION_SPLIT_GHOST_DETECTION_FIELDS = [
    *DRIVER_DETECTION_FIELDS,
    "revolution_index",
    "split",
    "train_artifact_rejected",
]

LOCAL_ILLUMINATION_FIELDS = [
    "frame_index",
    "image",
    "tile_px",
    "mask_threshold",
    "mask_mode",
    "mask_grow_threshold",
    "mask_dilation_px",
    "mask_margin_px",
    "mask_min_area_px",
    "fit_pixels",
    "masked_pixels",
    "field_median_gray",
    "field_p05_gray",
    "field_p95_gray",
    "field_max_abs_gray",
    "residual_median_before_gray",
    "residual_median_after_gray",
    "residual_rmse_before_gray",
    "residual_rmse_after_gray",
    "status",
]
