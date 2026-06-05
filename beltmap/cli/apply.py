from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from beltmap.detection import normalize_detection_mode

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}

OPTION_SPECS: tuple[tuple[str, str, str, tuple[tuple[str, ...], ...], str, str | None], ...] = (
    ("image_dir", "BELTMAP_IMAGE_DIR", "path", (("image_dir",), ("paths", "image_dir")), "Directory containing input images.", "DIR"),
    ("output_dir", "BELTMAP_OUTPUT_DIR", "path", (("output_dir",), ("paths", "output_dir")), "Directory where BeltMap outputs are written.", "DIR"),
    ("reuse_belt_map_path", "REUSE_BELT_MAP_PATH", "path", (("reuse_belt_map_path",), ("reuse", "belt_map_path")), "Optional existing belt_map.npy to reuse instead of rebuilding the belt map.", "NPY"),
    ("reuse_phase_estimates_path", "REUSE_PHASE_ESTIMATES_PATH", "path", (("reuse_phase_estimates_path",), ("reuse", "phase_estimates_path")), "Optional existing phase_estimates.csv to reuse with a reused belt map.", "CSV"),
    ("reuse_static_noise_path", "REUSE_STATIC_NOISE_PATH", "path", (("reuse_static_noise_path",), ("reuse", "static_noise_path")), "Optional existing static_noise.npy to reuse for residual normalization.", "NPY"),
    ("reuse_static_background_path", "REUSE_STATIC_BACKGROUND_PATH", "path", (("reuse_static_background_path",), ("reuse", "static_background_path")), "Optional existing static_background.npy additive residual map to subtract during detection.", "NPY"),
    ("reuse_recurrent_artifact_map_path", "REUSE_RECURRENT_ARTIFACT_MAP_PATH", "path", (("reuse_recurrent_artifact_map_path",), ("reuse", "recurrent_artifact_map_path")), "Optional existing recurrent_artifact_map.npy to reuse for recurrent artifact filtering.", "NPY"),
    ("reuse_map_support_path", "REUSE_MAP_SUPPORT_PATH", "path", (("reuse_map_support_path",), ("reuse", "map_support_path")), "Optional existing belt_map_support.npy to reuse with a reused belt map for map-risk diagnostics.", "NPY"),
    ("belt_region", "BELT_REGION", "region", (("belt_region",), ("belt", "region")), "Belt crop as top,left,height,width. Omit to use the full frame.", "TOP,LEFT,HEIGHT,WIDTH"),
    ("belt_velocity_px_per_frame", "BELT_VELOCITY_PX_PER_FRAME", "velocity", (("belt_velocity_px_per_frame",), ("belt", "velocity_px_per_frame")), "Signed belt image velocity, or 'auto'. Numeric values use --belt-velocity-frame-unit when frames are strided.", "PX_PER_FRAME|auto"),
    ("belt_velocity_frame_unit", "BELT_VELOCITY_FRAME_UNIT", "path", (("belt_velocity_frame_unit",), ("belt", "velocity_frame_unit")), "Frame unit for a supplied belt velocity: selected_frame or source_frame.", "UNIT"),
    ("belt_period_px", "BELT_PERIOD_PX", "int", (("belt_period_px",), ("belt", "period_px")), "Optional belt circumference/period in pixels.", "PX"),
    ("detection_threshold", "DETECTION_THRESHOLD", "float", (("detection_threshold",), ("detection", "threshold")), "Threshold on normalized residuals for bright particles.", "Z"),
    ("detection_mode", "DETECTION_MODE", "path", (("detection_mode",), ("detection", "mode"), ("detection", "method")), "Detection residual polarity: positive, negative, or absolute. Legacy detector method aliases threshold, hysteresis, and hysteresis_abs are also accepted in config files.", "MODE"),
    ("detection_low_threshold", "DETECTION_LOW_THRESHOLD", "float", (("detection_low_threshold",), ("detection", "low_threshold"), ("detection", "grow_threshold")), "Optional lower hysteresis threshold for final detection. Use 0 to disable.", "Z"),
    ("min_area_px", "MIN_AREA_PX", "int", (("min_area_px",), ("detection", "min_area_px")), "Minimum connected-component area for detections.", "PX"),
    ("detection_max_area_px", "DETECTION_MAX_AREA_PX", "int", (("detection_max_area_px",), ("detection", "max_area_px")), "Optional maximum connected-component area for detections. Use 0 to disable.", "PX"),
    ("detection_min_bbox_width_px", "DETECTION_MIN_BBOX_WIDTH_PX", "int", (("detection_min_bbox_width_px",), ("detection", "min_bbox_width_px")), "Optional minimum detection bounding-box width. Use 0 to disable.", "PX"),
    ("detection_min_bbox_height_px", "DETECTION_MIN_BBOX_HEIGHT_PX", "int", (("detection_min_bbox_height_px",), ("detection", "min_bbox_height_px")), "Optional minimum detection bounding-box height. Use 0 to disable.", "PX"),
    ("detection_max_bbox_aspect_ratio", "DETECTION_MAX_BBOX_ASPECT_RATIO", "float", (("detection_max_bbox_aspect_ratio",), ("detection", "max_bbox_aspect_ratio")), "Optional maximum detection bounding-box aspect ratio. Use 0 to disable.", "RATIO"),
    ("detection_min_bbox_extent", "DETECTION_MIN_BBOX_EXTENT", "float", (("detection_min_bbox_extent",), ("detection", "min_bbox_extent")), "Optional minimum area/bounding-box-area extent. Use 0 to disable.", "FRACTION"),
    ("detection_split_merged_components", "DETECTION_SPLIT_MERGED_COMPONENTS", "bool", (("detection_split_merged_components",), ("detection", "split_merged_components")), "Split merged connected components at narrow projection valleys.", None),
    ("detection_split_min_projection_gap_px", "DETECTION_SPLIT_MIN_PROJECTION_GAP_PX", "int", (("detection_split_min_projection_gap_px",), ("detection", "split_min_projection_gap_px")), "Minimum projection-valley width used when splitting merged detections.", "PX"),
    ("detection_split_min_component_area_px", "DETECTION_SPLIT_MIN_COMPONENT_AREA_PX", "int", (("detection_split_min_component_area_px",), ("detection", "split_min_component_area_px")), "Minimum area for each split detection component. Use 0 to inherit min_area_px.", "PX"),
    ("residual_noise_radius_px", "RESIDUAL_NOISE_RADIUS_PX", "int", (("residual_noise_radius_px",), ("residual", "noise_radius_px")), "Local residual-noise box radius.", "PX"),
    ("residual_clip_sigma", "RESIDUAL_CLIP_SIGMA", "float", (("residual_clip_sigma",), ("residual", "clip_sigma")), "Symmetric residual clipping level for local-noise estimation. Use 0 to disable.", "SIGMA"),
    ("residual_min_noise", "RESIDUAL_MIN_NOISE", "float", (("residual_min_noise",), ("residual", "min_noise")), "Minimum local residual-noise scale.", "GRAY"),
    ("residual_noise_exclusion_sigma", "RESIDUAL_NOISE_EXCLUSION_SIGMA", "float", (("residual_noise_exclusion_sigma",), ("residual", "noise_exclusion_sigma")), "Positive-residual threshold for excluding particle-like pixels from local-noise estimation. Use 0 to disable.", "SIGMA"),
    ("residual_noise_exclusion_radius_px", "RESIDUAL_NOISE_EXCLUSION_RADIUS_PX", "int", (("residual_noise_exclusion_radius_px",), ("residual", "noise_exclusion_radius_px")), "Dilation radius around particle-like pixels excluded from local-noise windows.", "PX"),
    ("photometric_enabled", "PHOTOMETRIC_ENABLED", "bool", (("photometric_enabled",), ("photometric", "enabled")), "Fit and apply a robust per-frame gain/offset correction before residual detection.", None),
    ("photometric_trim_fraction", "PHOTOMETRIC_TRIM_FRACTION", "float", (("photometric_trim_fraction",), ("photometric", "trim_fraction")), "Fraction of largest photometric-fit residuals trimmed on each iteration.", "FRACTION"),
    ("photometric_max_iterations", "PHOTOMETRIC_MAX_ITERATIONS", "int", (("photometric_max_iterations",), ("photometric", "max_iterations")), "Maximum robust photometric gain/offset fitting iterations.", "N"),
    ("photometric_min_pixels", "PHOTOMETRIC_MIN_PIXELS", "int", (("photometric_min_pixels",), ("photometric", "min_pixels")), "Minimum valid pixels required for a photometric gain/offset fit.", "N"),
    ("detection_local_illumination_correction", "DETECTION_LOCAL_ILLUMINATION_CORRECTION", "bool", (("detection_local_illumination_correction",), ("detection", "local_illumination_correction")), "Apply local residual-median illumination correction before final detection.", None),
    ("detection_local_illumination_tile_px", "DETECTION_LOCAL_ILLUMINATION_TILE_PX", "int", (("detection_local_illumination_tile_px",), ("detection", "local_illumination_tile_px")), "Tile size used for local illumination correction before final detection.", "PX"),
    ("detection_local_illumination_min_pixels", "DETECTION_LOCAL_ILLUMINATION_MIN_PIXELS", "int", (("detection_local_illumination_min_pixels",), ("detection", "local_illumination_min_pixels")), "Minimum non-particle residual pixels required to fit a local illumination field.", "N"),
    ("detection_local_illumination_mask_threshold", "DETECTION_LOCAL_ILLUMINATION_MASK_THRESHOLD", "float", (("detection_local_illumination_mask_threshold",), ("detection", "local_illumination_mask_threshold")), "Strong normalized-residual threshold used to mask particles while fitting local illumination. Omit to follow map.particle_mask_threshold.", "Z"),
    ("detection_local_illumination_mask_mode", "DETECTION_LOCAL_ILLUMINATION_MASK_MODE", "path", (("detection_local_illumination_mask_mode",), ("detection", "local_illumination_mask_mode")), "Particle-mask mode used while fitting local illumination. Omit to follow map.particle_mask_mode.", "MODE"),
    ("detection_local_illumination_mask_grow_threshold", "DETECTION_LOCAL_ILLUMINATION_MASK_GROW_THRESHOLD", "float", (("detection_local_illumination_mask_grow_threshold",), ("detection", "local_illumination_mask_grow_threshold")), "Lower hysteresis threshold used for local-illumination particle masking. Omit to follow map.particle_mask_grow_threshold.", "Z"),
    ("detection_local_illumination_mask_dilation_px", "DETECTION_LOCAL_ILLUMINATION_MASK_DILATION_PX", "int", (("detection_local_illumination_mask_dilation_px",), ("detection", "local_illumination_mask_dilation_px")), "Morphological dilation radius for local-illumination particle masks. Omit to follow map.particle_mask_dilation_px.", "PX"),
    ("detection_local_illumination_mask_margin_px", "DETECTION_LOCAL_ILLUMINATION_MASK_MARGIN_PX", "int", (("detection_local_illumination_mask_margin_px",), ("detection", "local_illumination_mask_margin_px")), "Safety margin around particle boxes while fitting local illumination. Omit to follow map.particle_mask_margin_px.", "PX"),
    ("detection_local_illumination_mask_min_area_px", "DETECTION_LOCAL_ILLUMINATION_MASK_MIN_AREA_PX", "int", (("detection_local_illumination_mask_min_area_px",), ("detection", "local_illumination_mask_min_area_px")), "Minimum particle-mask component area while fitting local illumination. Omit to follow map.particle_mask_min_area_px.", "PX"),
    ("min_track_length", "MIN_TRACK_LENGTH", "int", (("min_track_length",), ("tracking", "min_track_length")), "Minimum detections per track for velocity estimates.", "N"),
    ("max_match_distance_px", "MAX_MATCH_DISTANCE_PX", "float", (("max_match_distance_px",), ("tracking", "max_match_distance_px")), "Optional tracking match distance. Omit to derive it from belt speed.", "PX"),
    ("tracking_max_frame_gap", "TRACKING_MAX_FRAME_GAP", "float", (("tracking_max_frame_gap",), ("tracking", "max_frame_gap")), "Maximum selected-frame gap allowed when linking detections into one track.", "FRAMES"),
    ("tracking_velocity_fit_method", "TRACKING_VELOCITY_FIT_METHOD", "path", (("tracking_velocity_fit_method",), ("tracking", "velocity_fit_method")), "Velocity fit method: linear or theil_sen.", "METHOD"),
    ("track_filter_min_length", "TRACK_FILTER_MIN_LENGTH", "int", (("track_filter_min_length",), ("track_filter", "min_length")), "Minimum detections per accepted filtered track.", "N"),
    ("track_filter_min_velocity_ratio_y", "TRACK_FILTER_MIN_VELOCITY_RATIO_Y", "float", (("track_filter_min_velocity_ratio_y",), ("track_filter", "min_velocity_ratio_y")), "Minimum accepted particle/belt vertical velocity ratio.", "RATIO"),
    ("track_filter_max_velocity_ratio_y", "TRACK_FILTER_MAX_VELOCITY_RATIO_Y", "float", (("track_filter_max_velocity_ratio_y",), ("track_filter", "max_velocity_ratio_y")), "Maximum accepted particle/belt vertical velocity ratio.", "RATIO"),
    ("track_filter_max_abs_x_velocity_px_per_frame", "TRACK_FILTER_MAX_ABS_X_VELOCITY_PX_PER_FRAME", "float", (("track_filter_max_abs_x_velocity_px_per_frame",), ("track_filter", "max_abs_x_velocity_px_per_frame")), "Optional maximum accepted absolute lateral velocity. Use 0 to disable.", "PX_PER_FRAME"),
    ("track_filter_max_recurrent_artifact_track_score", "TRACK_FILTER_MAX_RECURRENT_ARTIFACT_SCORE", "float", (("track_filter_max_recurrent_artifact_track_score",), ("track_filter", "max_recurrent_artifact_track_score")), "Optional maximum accepted track-level recurrent artifact score. Use 0 to disable.", "FRACTION"),
    ("track_filter_recurrent_artifact_detection_threshold", "TRACK_FILTER_RECURRENT_ARTIFACT_DETECTION_THRESHOLD", "float", (("track_filter_recurrent_artifact_detection_threshold",), ("track_filter", "recurrent_artifact_detection_threshold")), "Per-detection recurrent artifact evidence threshold used for track-level scoring.", "FRACTION"),
    ("max_frames", "MAX_FRAMES", "int", (("max_frames",), ("frames", "max_frames")), "Maximum number of selected frames to process. Use 0 for all frames.", "N"),
    ("frame_stride", "FRAME_STRIDE", "int", (("frame_stride",), ("frames", "stride")), "Process every Nth frame after natural sorting.", "N"),
    ("map_sample_frames", "MAP_SAMPLE_FRAMES", "int", (("map_sample_frames",), ("map", "sample_frames")), "Number of frames sampled to build the belt map.", "N"),
    ("map_sample_strategy", "MAP_SAMPLE_STRATEGY", "path", (("map_sample_strategy",), ("map", "sample_strategy")), "Alias for map_sampling_strategy: uniform, phase_coverage, or adaptive_phase_coverage.", "STRATEGY"),
    ("map_adaptive_candidate_frames", "MAP_ADAPTIVE_CANDIDATE_FRAMES", "int", (("map_adaptive_candidate_frames",), ("map", "adaptive_candidate_frames")), "Candidate frame count for adaptive map sampling. Use 0 to consider all frames.", "N"),
    ("map_sampling_strategy", "MAP_SAMPLING_STRATEGY", "path", (("map_sampling_strategy",), ("map", "sampling_strategy")), "Frame sampling strategy for belt-map reconstruction: uniform or adaptive_phase_coverage.", "STRATEGY"),
    ("map_reconstruction_trim_fraction", "MAP_RECONSTRUCTION_TRIM_FRACTION", "float", (("map_reconstruction_trim_fraction",), ("map", "reconstruction_trim_fraction")), "Symmetric per-pixel trim/winsorization fraction for robust belt-map reconstruction. Use 0 for the arithmetic mean.", "FRACTION"),
    ("map_fractional_splat", "MAP_FRACTIONAL_SPLAT", "bool", (("map_fractional_splat",), ("map", "fractional_splat")), "Use fractional row weights when accumulating belt-map pixels.", None),
    ("map_frame_median_offset_correction", "MAP_FRAME_MEDIAN_OFFSET_CORRECTION", "bool", (("map_frame_median_offset_correction",), ("map", "frame_median_offset_correction")), "Normalize sampled frames by a robust median brightness offset before belt-map accumulation.", None),
    ("map_local_illumination_correction", "MAP_LOCAL_ILLUMINATION_CORRECTION", "bool", (("map_local_illumination_correction",), ("map", "local_illumination_correction")), "Estimate and subtract a low-frequency additive illumination field during map accumulation.", None),
    ("map_local_illumination_tile_px", "MAP_LOCAL_ILLUMINATION_TILE_PX", "int", (("map_local_illumination_tile_px",), ("map", "local_illumination_tile_px")), "Tile size used for local residual-median illumination correction.", "PX"),
    ("map_mask_iterations", "MAP_MASK_ITERATIONS", "int", (("map_mask_iterations",), ("map", "mask_iterations")), "Particle-mask refinement iterations while building the belt map.", "N"),
    ("map_particle_mask_threshold", "MAP_PARTICLE_MASK_THRESHOLD", "float", (("map_particle_mask_threshold",), ("map", "particle_mask_threshold"), ("map_particle_mask", "threshold")), "Strong threshold used for particle masking during map building.", "Z"),
    ("map_particle_mask_mode", "MAP_PARTICLE_MASK_MODE", "path", (("map_particle_mask_mode",), ("map", "particle_mask_mode"), ("map_particle_mask", "mode"), ("map_particle_mask", "method")), "Map-building particle mask mode: positive, negative, absolute, or hysteresis_abs.", "MODE"),
    ("map_particle_mask_grow_threshold", "MAP_PARTICLE_MASK_GROW_THRESHOLD", "float", (("map_particle_mask_grow_threshold",), ("map", "particle_mask_grow_threshold"), ("map_particle_mask", "grow_threshold"), ("map_particle_mask", "low_threshold")), "Lower absolute-residual threshold used to grow hysteresis map masks.", "Z"),
    ("map_particle_mask_dilation_px", "MAP_PARTICLE_MASK_DILATION_PX", "int", (("map_particle_mask_dilation_px",), ("map", "particle_mask_dilation_px"), ("map_particle_mask", "dilation_px")), "Morphological dilation radius for map-building particle masks.", "PX"),
    ("map_particle_mask_margin_px", "MAP_PARTICLE_MASK_MARGIN_PX", "int", (("map_particle_mask_margin_px",), ("map", "particle_mask_margin_px"), ("map_particle_mask", "margin_px")), "Safety margin around detected particle boxes during map building.", "PX"),
    ("map_particle_mask_min_area_px", "MAP_PARTICLE_MASK_MIN_AREA_PX", "int", (("map_particle_mask_min_area_px",), ("map", "particle_mask_min_area_px"), ("map_particle_mask", "min_area_px")), "Minimum component area for map-building masks; independent of final detection.min_area_px.", "PX"),
    ("map_aggregation", "MAP_AGGREGATION", "path", (("map_aggregation",), ("map", "aggregation")), "Belt-map aggregation method: mean, huber, trimmed_mean, or winsorized_mean.", "METHOD"),
    ("map_robust_iterations", "MAP_ROBUST_ITERATIONS", "int", (("map_robust_iterations",), ("map", "robust_iterations")), "Robust Huber refinement iterations for aggregation='huber'.", "N"),
    ("map_robust_huber_delta", "MAP_ROBUST_HUBER_DELTA", "float", (("map_robust_huber_delta",), ("map", "robust_huber_delta")), "Huber cutoff in robust residual scale units.", "SIGMA"),
    ("map_robust_min_scale", "MAP_ROBUST_MIN_SCALE", "float", (("map_robust_min_scale",), ("map", "robust_min_scale")), "Minimum residual scale for Huber weighting.", "GRAY"),
    ("map_risk_min_support", "MAP_RISK_MIN_SUPPORT", "float", (("map_risk_min_support",), ("map_risk", "min_support")), "Effective observations required before a belt-map pixel is no longer low-support.", "COUNT"),
    ("map_risk_reject_max_mean", "MAP_RISK_REJECT_MAX_MEAN", "float", (("map_risk_reject_max_mean",), ("map_risk", "reject_max_mean")), "Reject detections whose bbox mean map risk exceeds this value. 1 disables.", "FRACTION"),
    ("map_risk_reject_max_interpolated_fraction", "MAP_RISK_REJECT_MAX_INTERPOLATED_FRACTION", "float", (("map_risk_reject_max_interpolated_fraction",), ("map_risk", "reject_max_interpolated_fraction")), "Reject detections whose bbox interpolated fraction exceeds this value. 1 disables.", "FRACTION"),
    ("map_risk_reject_max_low_support_fraction", "MAP_RISK_REJECT_MAX_LOW_SUPPORT_FRACTION", "float", (("map_risk_reject_max_low_support_fraction",), ("map_risk", "reject_max_low_support_fraction")), "Reject detections whose bbox low-support fraction exceeds this value. 1 disables.", "FRACTION"),
    ("static_noise_sample_frames", "STATIC_NOISE_SAMPLE_FRAMES", "auto_int", (("static_noise_sample_frames",), ("static_noise", "sample_frames")), "Number of residual frames sampled to learn a static residual-noise map. Use 0 to disable or 'auto' to sample automatically.", "N|auto"),
    ("static_noise_min_scale", "STATIC_NOISE_MIN_SCALE", "float", (("static_noise_min_scale",), ("static_noise", "min_scale")), "Minimum per-pixel static residual-noise floor.", "GRAY"),
    ("static_noise_mask_threshold", "STATIC_NOISE_MASK_THRESHOLD", "float", (("static_noise_mask_threshold",), ("static_noise", "mask_threshold")), "Optional normalized residual threshold for masking particles while learning static noise. Use 0 to disable.", "Z"),
    ("static_noise_mask_margin_px", "STATIC_NOISE_MASK_MARGIN_PX", "int", (("static_noise_mask_margin_px",), ("static_noise", "mask_margin_px")), "Safety margin around particle boxes while learning static noise.", "PX"),
    ("static_noise_mask_min_area_px", "STATIC_NOISE_MASK_MIN_AREA_PX", "int", (("static_noise_mask_min_area_px",), ("static_noise", "mask_min_area_px")), "Minimum component area for static-noise masks; defaults to the map-particle mask gate.", "PX"),
    ("static_background_sample_frames", "STATIC_BACKGROUND_SAMPLE_FRAMES", "auto_int", (("static_background_sample_frames",), ("static_background", "sample_frames")), "Number of residual frames sampled to learn an additive image-fixed background. Use 0 to disable or 'auto' to sample automatically.", "N|auto"),
    ("static_background_mask_threshold", "STATIC_BACKGROUND_MASK_THRESHOLD", "float", (("static_background_mask_threshold",), ("static_background", "mask_threshold")), "Optional normalized residual threshold for masking particles while learning static background. Use 0 to disable.", "Z"),
    ("static_background_mask_margin_px", "STATIC_BACKGROUND_MASK_MARGIN_PX", "int", (("static_background_mask_margin_px",), ("static_background", "mask_margin_px")), "Safety margin around particle boxes while learning static background.", "PX"),
    ("static_background_mask_min_area_px", "STATIC_BACKGROUND_MASK_MIN_AREA_PX", "int", (("static_background_mask_min_area_px",), ("static_background", "mask_min_area_px")), "Minimum component area for static-background masks; defaults to the map-particle mask gate.", "PX"),
    ("recurrent_artifact_min_revolutions", "RECURRENT_ARTIFACT_MIN_REVOLUTIONS", "int", (("recurrent_artifact_min_revolutions",), ("recurrent_artifact", "min_revolutions")), "Minimum distinct belt revolutions required to mark recurrent artifact pixels. Use 0 to disable building unless a reused map is set.", "N"),
    ("recurrent_artifact_margin_px", "RECURRENT_ARTIFACT_MARGIN_PX", "int", (("recurrent_artifact_margin_px",), ("recurrent_artifact", "margin_px")), "Safety margin around detection boxes when accumulating recurrent artifacts.", "PX"),
    ("recurrent_artifact_max_overlap_fraction", "RECURRENT_ARTIFACT_MAX_OVERLAP_FRACTION", "float", (("recurrent_artifact_max_overlap_fraction",), ("recurrent_artifact", "max_overlap_fraction")), "Reject detections whose belt-coordinate bbox overlaps recurrent artifacts above this fraction.", "FRACTION"),
    ("recurrent_artifact_min_recurrence_probability", "RECURRENT_ARTIFACT_MIN_RECURRENCE_PROBABILITY", "float", (("recurrent_artifact_min_recurrence_probability",), ("recurrent_artifact", "min_recurrence_probability")), "Minimum exposure-normalized recurrence probability required before pixels contribute to the recurrent-artifact prior.", "PROBABILITY"),
    ("recurrent_artifact_mode", "RECURRENT_ARTIFACT_MODE", "path", (("recurrent_artifact_mode",), ("recurrent_artifact", "mode")), "Recurrent artifact filter mode: hard rejects by overlap; soft keeps strong peaks; probabilistic uses exposure-normalized recurrence probabilities.", "MODE"),
    ("recurrent_artifact_soft_penalty_weight", "RECURRENT_ARTIFACT_SOFT_PENALTY_WEIGHT", "float", (("recurrent_artifact_soft_penalty_weight",), ("recurrent_artifact", "soft_penalty_weight")), "Soft-mode peak-signal penalty per artifact-overlap fraction.", "WEIGHT"),
    ("recurrent_artifact_candidate_max_area_px", "RECURRENT_ARTIFACT_CANDIDATE_MAX_AREA_PX", "int", (("recurrent_artifact_candidate_max_area_px",), ("recurrent_artifact", "candidate_max_area_px")), "Optional max component area used when building the recurrent-artifact prior. Use 0 to include all detections.", "PX"),
    ("recurrent_artifact_candidate_max_peak_signal", "RECURRENT_ARTIFACT_CANDIDATE_MAX_PEAK_SIGNAL", "float", (("recurrent_artifact_candidate_max_peak_signal",), ("recurrent_artifact", "candidate_max_peak_signal")), "Optional max peak signal used when building the recurrent-artifact prior. Use 0 to include all detections.", "SIGNAL"),
    ("recurrent_artifact_reject_max_area_px", "RECURRENT_ARTIFACT_REJECT_MAX_AREA_PX", "int", (("recurrent_artifact_reject_max_area_px",), ("recurrent_artifact", "reject_max_area_px")), "Optional max component area eligible for recurrent-artifact rejection. Use 0 to allow rejecting all detections.", "PX"),
    ("recurrent_artifact_reject_max_peak_signal", "RECURRENT_ARTIFACT_REJECT_MAX_PEAK_SIGNAL", "float", (("recurrent_artifact_reject_max_peak_signal",), ("recurrent_artifact", "reject_max_peak_signal")), "Optional max peak signal eligible for recurrent-artifact rejection. Use 0 to allow rejecting all detections.", "SIGNAL"),
    ("revolution_split_enabled", "REVOLUTION_SPLIT_ENABLED", "bool", (("revolution_split_enabled",), ("revolution_split", "enabled")), "Build the belt map from train revolutions and hold out eval revolutions for ghost diagnostics.", None),
    ("revolution_split_eval_every", "REVOLUTION_SPLIT_EVAL_EVERY", "int", (("revolution_split_eval_every",), ("revolution_split", "eval_every")), "Hold out every Nth observed belt revolution when explicit eval revolutions are unset.", "N"),
    ("revolution_split_eval_offset", "REVOLUTION_SPLIT_EVAL_OFFSET", "int", (("revolution_split_eval_offset",), ("revolution_split", "eval_offset")), "Modulo offset used with revolution_split.eval_every.", "N"),
    ("revolution_split_eval_revolutions", "REVOLUTION_SPLIT_EVAL_REVOLUTIONS", "path", (("revolution_split_eval_revolutions",), ("revolution_split", "eval_revolutions")), "Comma-separated held-out revolution indices or ranges, for example 1,4,7-9. Overrides eval_every when set.", "LIST"),
    ("revolution_split_min_train_revolutions", "REVOLUTION_SPLIT_MIN_TRAIN_REVOLUTIONS", "int", (("revolution_split_min_train_revolutions",), ("revolution_split", "min_train_revolutions")), "Minimum number of train revolutions required by the split.", "N"),
    ("revolution_split_min_eval_revolutions", "REVOLUTION_SPLIT_MIN_EVAL_REVOLUTIONS", "int", (("revolution_split_min_eval_revolutions",), ("revolution_split", "min_eval_revolutions")), "Minimum number of held-out eval revolutions required by the split.", "N"),
    ("revolution_split_ghost_min_revolutions", "REVOLUTION_SPLIT_GHOST_MIN_REVOLUTIONS", "int", (("revolution_split_ghost_min_revolutions",), ("revolution_split", "ghost_min_revolutions")), "Minimum train revolutions required for the train-only artifact map used to score held-out ghosts.", "N"),
    ("cross_map_agreement_enabled", "CROSS_MAP_AGREEMENT_ENABLED", "bool", (("cross_map_agreement_enabled",), ("cross_map_agreement", "enabled")), "Build two disjoint-sample belt maps and score final detections by cross-map agreement.", None),
    ("cross_map_agreement_filter", "CROSS_MAP_AGREEMENT_FILTER", "bool", (("cross_map_agreement_filter",), ("cross_map_agreement", "filter")), "Reject detections that fail cross-map agreement. Disable for score-only diagnostics.", None),
    ("cross_map_agreement_min_confirming_maps", "CROSS_MAP_AGREEMENT_MIN_CONFIRMING_MAPS", "int", (("cross_map_agreement_min_confirming_maps",), ("cross_map_agreement", "min_confirming_maps")), "Minimum number of independent maps that must reproduce a detection.", "N"),
    ("cross_map_agreement_min_samples_per_map", "CROSS_MAP_AGREEMENT_MIN_SAMPLES_PER_MAP", "int", (("cross_map_agreement_min_samples_per_map",), ("cross_map_agreement", "min_samples_per_map")), "Minimum sampled frames required for each cross-fitted map.", "N"),
    ("cross_map_agreement_max_centroid_distance_px", "CROSS_MAP_AGREEMENT_MAX_CENTROID_DISTANCE_PX", "float", (("cross_map_agreement_max_centroid_distance_px",), ("cross_map_agreement", "max_centroid_distance_px")), "Maximum centroid distance for a confirming component.", "PX"),
    ("cross_map_agreement_min_bbox_iou", "CROSS_MAP_AGREEMENT_MIN_BBOX_IOU", "float", (("cross_map_agreement_min_bbox_iou",), ("cross_map_agreement", "min_bbox_iou")), "Minimum bbox IoU for a confirming component. Use 0 to rely on centroid distance.", "FRACTION"),
    ("cross_map_agreement_min_peak_ratio", "CROSS_MAP_AGREEMENT_MIN_PEAK_RATIO", "float", (("cross_map_agreement_min_peak_ratio",), ("cross_map_agreement", "min_peak_ratio")), "Minimum min/max peak-signal ratio between primary and confirming components.", "RATIO"),
    ("cross_map_agreement_require_sign_consistency", "CROSS_MAP_AGREEMENT_REQUIRE_SIGN_CONSISTENCY", "bool", (("cross_map_agreement_require_sign_consistency",), ("cross_map_agreement", "require_sign_consistency")), "Require the un-oriented residual sign to agree between primary and confirming components.", None),
    ("velocity_search_radius_px", "VELOCITY_SEARCH_RADIUS_PX", "int", (("velocity_search_radius_px",), ("auto_velocity", "search_radius_px")), "Max vertical shift searched during automatic belt-velocity estimation.", "PX"),
    ("velocity_estimation_pairs", "VELOCITY_ESTIMATION_PAIRS", "int", (("velocity_estimation_pairs",), ("auto_velocity", "estimation_pairs")), "Number of adjacent frame pairs used for automatic velocity estimation.", "N"),
    ("auto_velocity_min_abs_px_per_frame", "AUTO_VELOCITY_MIN_ABS_PX_PER_FRAME", "float", (("auto_velocity_min_abs_px_per_frame",), ("auto_velocity", "min_abs_px_per_frame")), "Minimum accepted absolute auto-estimated belt velocity.", "PX_PER_FRAME"),
    ("auto_velocity_max_edge_fraction", "AUTO_VELOCITY_MAX_EDGE_FRACTION", "float", (("auto_velocity_max_edge_fraction",), ("auto_velocity", "max_edge_fraction")), "Maximum accepted fraction of auto-velocity shifts that hit the search edge.", "FRACTION"),
    ("allow_full_frame_auto_velocity", "ALLOW_FULL_FRAME_AUTO_VELOCITY", "bool", (("allow_full_frame_auto_velocity",), ("auto_velocity", "allow_full_frame")), "Allow automatic belt-velocity estimation on a full-frame belt region.", None),
    ("registration_search_radius_px", "REGISTRATION_SEARCH_RADIUS_PX", "float", (("registration_search_radius_px",), ("registration", "search_radius_px")), "Phase-registration search radius in pixels.", "PX"),
    ("registration_search_step_px", "REGISTRATION_SEARCH_STEP_PX", "float", (("registration_search_step_px",), ("registration", "search_step_px")), "Phase-registration search step in pixels.", "PX"),
    ("registration_subpixel_refinement", "REGISTRATION_SUBPIXEL_REFINEMENT", "bool", (("registration_subpixel_refinement",), ("registration", "subpixel_refinement")), "Refine the best phase-registration offset with a local quadratic fit.", None),
    ("registration_robust_normalization", "REGISTRATION_ROBUST_NORMALIZATION", "bool", (("registration_robust_normalization",), ("registration", "robust_normalization")), "Normalize registration images by a robust MAD scale instead of standard deviation.", None),
    ("registration_quality_enabled", "REGISTRATION_QUALITY_ENABLED", "bool", (("registration_quality_enabled",), ("registration_quality", "enabled")), "Enable frame-level registration-quality gates after phase estimation.", None),
    ("registration_quality_action", "REGISTRATION_QUALITY_ACTION", "path", (("registration_quality_action",), ("registration_quality", "action")), "Registration-quality response: report, inflate, or skip.", "ACTION"),
    ("registration_quality_min_score", "REGISTRATION_QUALITY_MIN_SCORE", "float", (("registration_quality_min_score",), ("registration_quality", "min_score")), "Minimum registration score before a frame is treated as low quality. Use 0 to disable.", "SCORE"),
    ("registration_quality_min_loss_gap_ratio", "REGISTRATION_QUALITY_MIN_LOSS_GAP_RATIO", "float", (("registration_quality_min_loss_gap_ratio",), ("registration_quality", "min_loss_gap_ratio")), "Minimum relative loss gap to the next-best registration candidate. Use 0 to disable.", "RATIO"),
    ("registration_quality_max_uncertainty_px", "REGISTRATION_QUALITY_MAX_UNCERTAINTY_PX", "float", (("registration_quality_max_uncertainty_px",), ("registration_quality", "max_uncertainty_px")), "Maximum curvature-derived registration uncertainty. Use 0 to disable.", "PX"),
    ("registration_quality_max_abs_correction_px", "REGISTRATION_QUALITY_MAX_ABS_CORRECTION_PX", "float", (("registration_quality_max_abs_correction_px",), ("registration_quality", "max_abs_correction_px")), "Maximum absolute phase correction accepted without quality action. Use 0 to disable.", "PX"),
    ("registration_quality_noise_inflation_factor", "REGISTRATION_QUALITY_NOISE_INFLATION_FACTOR", "float", (("registration_quality_noise_inflation_factor",), ("registration_quality", "noise_inflation_factor")), "Residual-noise multiplier for low-quality frames when action='inflate'.", "FACTOR"),
    ("registration_quality_uncertainty_inflation_scale", "REGISTRATION_QUALITY_UNCERTAINTY_INFLATION_SCALE", "float", (("registration_quality_uncertainty_inflation_scale",), ("registration_quality", "uncertainty_inflation_scale")), "Additional residual-noise multiplier per uncertainty pixel when action='inflate'.", "SCALE"),
    ("phase_estimation_mode", "PHASE_ESTIMATION_MODE", "path", (("phase_estimation_mode",), ("phase", "estimation_mode")), "Phase source for residual rendering: motion_model, registration, or smoothed_registration.", "MODE"),
    ("phase_refinement_iterations", "PHASE_REFINEMENT_ITERATIONS", "int", (("phase_refinement_iterations",), ("phase_refinement", "iterations")), "Phase-feedback map-refinement iterations. Use 0 to disable.", "N"),
    ("phase_refinement_min_score", "PHASE_REFINEMENT_MIN_SCORE", "float", (("phase_refinement_min_score",), ("phase_refinement", "min_score")), "Minimum registration score accepted for phase-feedback refinement.", "SCORE"),
    ("phase_refinement_max_abs_correction_px", "PHASE_REFINEMENT_MAX_ABS_CORRECTION_PX", "float", (("phase_refinement_max_abs_correction_px",), ("phase_refinement", "max_abs_correction_px")), "Maximum absolute registration correction accepted for phase-feedback refinement. Use 0 to disable this gate.", "PX"),
    ("phase_refinement_smoothing_window_frames", "PHASE_REFINEMENT_SMOOTHING_WINDOW_FRAMES", "int", (("phase_refinement_smoothing_window_frames",), ("phase_refinement", "smoothing_window_frames")), "Rolling-median smoothing window for accepted phase corrections.", "N"),
    ("phase_drift_enabled", "PHASE_DRIFT_ENABLED", "bool", (("phase_drift_enabled",), ("phase_drift", "enabled")), "Enable online residual phase-drift compensation during detection.", None),
    ("phase_drift_smoothing_alpha", "PHASE_DRIFT_SMOOTHING_ALPHA", "float", (("phase_drift_smoothing_alpha",), ("phase_drift", "smoothing_alpha")), "Exponential smoothing factor for online phase-drift updates.", "ALPHA"),
    ("phase_drift_min_score", "PHASE_DRIFT_MIN_SCORE", "float", (("phase_drift_min_score",), ("phase_drift", "min_score")), "Minimum registration score accepted by the online phase-drift filter.", "SCORE"),
    ("phase_drift_max_abs_residual_correction_px", "PHASE_DRIFT_MAX_ABS_RESIDUAL_CORRECTION_PX", "float", (("phase_drift_max_abs_residual_correction_px",), ("phase_drift", "max_abs_residual_correction_px")), "Maximum residual registration correction accepted by the online phase-drift filter. Use 0 to disable.", "PX"),
    ("phase_drift_max_abs_px", "PHASE_DRIFT_MAX_ABS_PX", "float", (("phase_drift_max_abs_px",), ("phase_drift", "max_abs_px")), "Maximum accumulated online phase drift. Use 0 to disable.", "PX"),
    ("phase_smoothing_window_frames", "PHASE_SMOOTHING_WINDOW_FRAMES", "int", (("phase_smoothing_window_frames",), ("phase_smoothing", "window_frames")), "Offline phase-correction smoothing window. Use 0 to disable.", "N"),
    ("phase_smoothing_min_score", "PHASE_SMOOTHING_MIN_SCORE", "float", (("phase_smoothing_min_score",), ("phase_smoothing", "min_score")), "Minimum registration score used by offline phase smoothing. Use 0 to disable.", "SCORE"),
    ("phase_smoothing_max_abs_correction_px", "PHASE_SMOOTHING_MAX_ABS_CORRECTION_PX", "float", (("phase_smoothing_max_abs_correction_px",), ("phase_smoothing", "max_abs_correction_px")), "Maximum correction used by offline phase smoothing. Use 0 to disable.", "PX"),
    ("phase_smoothing_min_support", "PHASE_SMOOTHING_MIN_SUPPORT", "int", (("phase_smoothing_min_support",), ("phase_smoothing", "min_support")), "Minimum neighboring estimates for offline phase smoothing.", "N"),
    ("progress_interval_frames", "PROGRESS_INTERVAL_FRAMES", "int", (("progress_interval_frames",), ("progress", "interval_frames")), "Print progress every N frames during long stages.", "N"),
    ("partial_output_interval_frames", "PARTIAL_OUTPUT_INTERVAL_FRAMES", "int", (("partial_output_interval_frames",), ("progress", "partial_output_interval_frames")), "Write partial CSV outputs every N processed frames. Use 0 for final only.", "N"),
    ("debug_residual_preview_frames", "DEBUG_RESIDUAL_PREVIEW_FRAMES", "int", (("debug_residual_preview_frames",), ("debug", "residual_preview_frames")), "Save residual PNG previews for the first N frames.", "N"),
    ("debug_residual_preview_interval_frames", "DEBUG_RESIDUAL_PREVIEW_INTERVAL_FRAMES", "int", (("debug_residual_preview_interval_frames",), ("debug", "residual_preview_interval_frames")), "Also save residual previews every N frames. Use 0 to disable.", "N"),
)

OPTION_BY_NAME = {spec[0]: spec for spec in OPTION_SPECS}
OPTION_BY_ENV = {spec[1]: spec for spec in OPTION_SPECS}
CONFIG_KEY_TO_NAME = {key: spec[0] for spec in OPTION_SPECS for key in spec[3]}
CONFIG_KEY_ALIASES = {
    # Backwards-compatible aliases used by earlier result-improvement notes.
    ("detection", "grow_threshold"): "detection_low_threshold",
}
DETECTION_METHOD_MODE_ALIASES = {
    "threshold": "positive",
    "hysteresis": "positive",
    "hysteresis_abs": "absolute",
    "positive": "positive",
    "negative": "negative",
    "absolute": "absolute",
}
MAP_SAMPLING_STRATEGY_ALIAS_OPTIONS = (
    "map_sample_strategy",
    "map_sampling_strategy",
)
OPTION_SOURCE_PRIORITY = {
    "config": 0,
    "env": 1,
    "cli": 2,
}

CONFIG_TEMPLATE = """# BeltMap image-sequence driver configuration.
# CLI flags override environment variables, and environment variables override values from this file.

[paths]
image_dir = "data/images"
output_dir = "outputs"

[reuse]
# belt_map_path = "outputs/belt_map.npy"
# phase_estimates_path = "outputs/phase_estimates.csv"
# static_noise_path = "outputs/static_noise.npy"
# static_background_path = "outputs/static_background.npy"
# recurrent_artifact_map_path = "outputs/recurrent_artifact_map.npy"
# map_support_path = "outputs/belt_map_support.npy"

[frames]
max_frames = 0
stride = 1

[belt]
region = [0, 220, 1330, 1800]
velocity_px_per_frame = "auto"
# Required when velocity_px_per_frame is numeric and frames.stride > 1.
# Use "source_frame" for adjacent original input frames or "selected_frame" for processed frames.
# velocity_frame_unit = "source_frame"
# period_px = 14723

[detection]
threshold = 5.0
mode = "positive"
low_threshold = 0.0
min_area_px = 4
max_area_px = 0
min_bbox_width_px = 0
min_bbox_height_px = 0
max_bbox_aspect_ratio = 0.0
min_bbox_extent = 0.0
split_merged_components = false
split_min_projection_gap_px = 2
split_min_component_area_px = 0
local_illumination_correction = false
local_illumination_tile_px = 64
local_illumination_min_pixels = 128
# Local illumination fitting is an ablation. Leave these unset to follow the
# map particle-mask settings; override them only for controlled experiments.
# local_illumination_mask_threshold = 5.0
# local_illumination_mask_mode = "positive"
# local_illumination_mask_grow_threshold = 2.0
# local_illumination_mask_dilation_px = 0
# local_illumination_mask_margin_px = 8
# local_illumination_mask_min_area_px = 4

[residual]
noise_radius_px = 15
clip_sigma = 5.0
min_noise = 1e-6
noise_exclusion_sigma = 4.0
noise_exclusion_radius_px = 2

[photometric]
enabled = false
trim_fraction = 0.05
max_iterations = 3
min_pixels = 128

[tracking]
min_track_length = 2
# max_match_distance_px = 90.0
max_frame_gap = 1.0
velocity_fit_method = "linear"

[track_filter]
min_length = 5
min_velocity_ratio_y = 0.0
max_velocity_ratio_y = 1.1
max_abs_x_velocity_px_per_frame = 0.0

[map]
sample_frames = 120
sampling_strategy = "uniform"
adaptive_candidate_frames = 0
reconstruction_trim_fraction = 0.0
fractional_splat = true
frame_median_offset_correction = false
local_illumination_correction = false
local_illumination_tile_px = 64
mask_iterations = 1
# aggregation accepts "mean", "huber", "trimmed_mean", or "winsorized_mean".
aggregation = "mean"
robust_iterations = 1
robust_huber_delta = 3.0
robust_min_scale = 1.0

[map_particle_mask]
# Used only while learning belt_map.npy and auxiliary maps. Keep this gate
# independent from [detection].min_area_px so small fragments can be excluded
# from the learned background even when final detections use a larger area gate.
mode = "positive"
threshold = 5.0
grow_threshold = 2.0
dilation_px = 0
margin_px = 8
min_area_px = 8

[map_risk]
min_support = 1.0
reject_max_mean = 1.0
reject_max_interpolated_fraction = 1.0
reject_max_low_support_fraction = 1.0

[static_noise]
sample_frames = 0
min_scale = 0.0
mask_threshold = 0.0
mask_margin_px = 8
mask_min_area_px = 8

[static_background]
sample_frames = 0
mask_threshold = 0.0
mask_margin_px = 8
mask_min_area_px = 8

[recurrent_artifact]
min_revolutions = 0
margin_px = 2
max_overlap_fraction = 0.3
min_recurrence_probability = 0.0
mode = "hard"
soft_penalty_weight = 1.0
candidate_max_area_px = 0
candidate_max_peak_signal = 0.0
reject_max_area_px = 0
reject_max_peak_signal = 0.0

[cross_map_agreement]
enabled = false
filter = true
min_confirming_maps = 2
min_samples_per_map = 10
max_centroid_distance_px = 4.0
min_bbox_iou = 0.0
min_peak_ratio = 0.25
require_sign_consistency = true

[revolution_split]
enabled = false
eval_every = 3
eval_offset = 0
eval_revolutions = ""
min_train_revolutions = 1
min_eval_revolutions = 1
ghost_min_revolutions = 2

[auto_velocity]
search_radius_px = 90
estimation_pairs = 100
min_abs_px_per_frame = 0.25
max_edge_fraction = 0.2
allow_full_frame = false

[registration]
search_radius_px = 8.0
search_step_px = 0.5
subpixel_refinement = true
robust_normalization = true

[phase]
estimation_mode = "registration"

[registration_quality]
enabled = false
action = "report"
min_score = 0.0
min_loss_gap_ratio = 0.0
max_uncertainty_px = 0.0
max_abs_correction_px = 0.0
noise_inflation_factor = 2.0
uncertainty_inflation_scale = 0.0

[phase_refinement]
iterations = 0
min_score = 0.0
max_abs_correction_px = 0.0
smoothing_window_frames = 25

[phase_drift]
enabled = true
smoothing_alpha = 0.15
min_score = 0.05
max_abs_residual_correction_px = 0.0
max_abs_px = 0.0

[phase_smoothing]
window_frames = 0
min_score = 0.0
max_abs_correction_px = 0.0
min_support = 3

[progress]
interval_frames = 25
partial_output_interval_frames = 250

[debug]
residual_preview_frames = 3
residual_preview_interval_frames = 0
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beltmap-apply",
        description="Apply BeltMap to an image sequence and write residual, detection, tracking, and velocity outputs.",
    )
    parser.add_argument("-c", "--config", type=Path, help="TOML or JSON config file. Values can be flat or grouped into sections.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved driver environment and exit without running the image driver.")
    parser.add_argument("--print-config", action="store_true", help="Print the resolved driver environment before running the image driver.")
    parser.add_argument("--write-config-template", type=Path, metavar="PATH", help="Write a TOML config template and exit.")
    for name, env_var, kind, _keys, help_text, metavar in OPTION_SPECS:
        flag = f"--{name.replace('_', '-')}"
        if kind == "bool":
            parser.add_argument(flag, dest=name, action=argparse.BooleanOptionalAction, default=None, help=f"{help_text} [env: {env_var}]")
        else:
            parser.add_argument(flag, dest=name, metavar=metavar, help=f"{help_text} [env: {env_var}]")
    return parser


def load_config_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    elif suffix in {".toml", ".tml"}:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported config file extension {suffix!r}; use .toml or .json")
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a top-level object/table")
    return data


def flatten_config(data: Mapping[str, Any], prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    flattened: dict[tuple[str, ...], Any] = {}
    for key, value in data.items():
        path = prefix + (key,)
        if isinstance(value, Mapping):
            flattened.update(flatten_config(value, path))
        else:
            flattened[path] = value
    return flattened


def normalize_value(name: str, value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
    kind = OPTION_BY_NAME[name][2]
    if kind == "bool":
        return "1" if parse_bool(value, name) else "0"
    if kind == "int":
        return str(parse_int(value, name))
    if kind == "float":
        return f"{parse_float(value, name):.15g}"
    if kind == "auto_int":
        if isinstance(value, str) and value.strip().lower() == "auto":
            return "auto"
        return str(parse_int(value, name))
    if kind == "velocity":
        if isinstance(value, str) and value.lower() == "auto":
            return "auto"
        return f"{parse_float(value, name):.15g}"
    if kind == "region":
        return format_region(value, name)
    if name == "detection_mode":
        return normalize_detection_mode(str(value))
    return str(value)


def parse_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in TRUE_VALUES:
            return True
        if lowered in FALSE_VALUES:
            return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def parse_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    parsed = int(value)
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return parsed


def parse_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return parsed


def format_region(value: Any, name: str) -> str:
    parts = [part.strip() for part in value.split(",")] if isinstance(value, str) else list(value)
    if len(parts) != 4:
        raise ValueError(f"{name} must contain exactly four values: top,left,height,width")
    return ",".join(str(parse_int(part, name)) for part in parts)


def values_from_config(path: Path | None) -> tuple[dict[str, str], dict[str, str]]:
    if path is None:
        return {}, {}
    values: dict[str, str] = {}
    sources: dict[str, str] = {}
    for key_path, raw_value in flatten_config(load_config_file(path)).items():
        name = CONFIG_KEY_TO_NAME.get(key_path)
        value = raw_value
        if key_path == ("detection", "method"):
            name = "detection_mode"
        if name is None:
            name = CONFIG_KEY_ALIASES.get(key_path)
        if name is None:
            raise ValueError(f"Unknown config option {'.'.join(key_path)!r}")
        normalized = normalize_value(name, value)
        if normalized is not None:
            if name in values:
                raise ValueError(
                    f"Config option {name!r} was specified more than once"
                )
            values[name] = normalized
            sources[name] = f"config:{path}"
    return values, sources


def values_from_environment(environ: Mapping[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    values: dict[str, str] = {}
    sources: dict[str, str] = {}
    for name, env_var, *_rest in OPTION_SPECS:
        raw_value = environ.get(env_var)
        if raw_value and raw_value.strip():
            values[name] = normalize_value(name, raw_value) or ""
            sources[name] = f"env:{env_var}"
    return values, sources


def values_from_args(namespace: argparse.Namespace) -> tuple[dict[str, str], dict[str, str]]:
    values: dict[str, str] = {}
    sources: dict[str, str] = {}
    for name, *_rest in OPTION_SPECS:
        raw_value = getattr(namespace, name)
        if raw_value is not None:
            normalized = normalize_value(name, raw_value)
            if normalized is not None:
                values[name] = normalized
                sources[name] = "cli"
    return values, sources


def option_source_priority(source: str) -> int:
    """Return the precedence layer for a resolved option source."""

    return OPTION_SOURCE_PRIORITY.get(source.split(":", 1)[0], -1)


def coalesce_map_sampling_strategy_aliases(
    values: dict[str, str],
    sources: dict[str, str],
) -> None:
    """Resolve legacy/canonical map-sampling strategy aliases to one option.

    The canonical ``map_sampling_strategy`` spelling wins within one precedence
    layer, while CLI values still override environment values and environment
    values still override config values when either alias spelling is used.
    """

    present = [
        name for name in MAP_SAMPLING_STRATEGY_ALIAS_OPTIONS if name in values
    ]
    if not present:
        return
    winner = max(
        present,
        key=lambda name: (
            option_source_priority(sources[name]),
            name == "map_sampling_strategy",
        ),
    )
    value = values[winner]
    source = sources[winner]
    for name in MAP_SAMPLING_STRATEGY_ALIAS_OPTIONS:
        values.pop(name, None)
        sources.pop(name, None)
    values["map_sampling_strategy"] = value
    sources["map_sampling_strategy"] = source


def resolve_driver_env(namespace: argparse.Namespace, environ: Mapping[str, str] | None = None) -> tuple[dict[str, str], dict[str, Any]]:
    current_environ = os.environ if environ is None else environ
    merged: dict[str, str] = {}
    sources: dict[str, str] = {}
    for layer_values, layer_sources in (values_from_config(namespace.config), values_from_environment(current_environ), values_from_args(namespace)):
        merged.update(layer_values)
        sources.update(layer_sources)
    coalesce_map_sampling_strategy_aliases(merged, sources)
    env_updates = {OPTION_BY_NAME[name][1]: value for name, value in merged.items() if value != ""}
    report = {
        "precedence": ["config", "environment", "cli"],
        "options": {
            name: {"env_var": OPTION_BY_NAME[name][1], "value": value, "source": sources[name]}
            for name, value in sorted(merged.items())
        },
        "driver_environment": dict(sorted(env_updates.items())),
    }
    return env_updates, report


def write_config_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(CONFIG_TEMPLATE, encoding="utf-8")


def write_resolved_config(report: Mapping[str, Any], env_updates: Mapping[str, str]) -> Path:
    output_dir = Path(env_updates.get("BELTMAP_OUTPUT_DIR") or os.getenv("BELTMAP_OUTPUT_DIR", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "config_resolved.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


def run_driver(env_updates: Mapping[str, str], report: Mapping[str, Any]) -> None:
    os.environ.update(env_updates)
    write_resolved_config(report, env_updates)
    from beltmap import driver
    driver.main()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.write_config_template is not None:
        write_config_template(args.write_config_template)
        return 0
    try:
        env_updates, report = resolve_driver_env(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if args.print_config or args.dry_run:
        print(json.dumps(report, indent=2), flush=True)
    if args.dry_run:
        return 0
    run_driver(env_updates, report)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
