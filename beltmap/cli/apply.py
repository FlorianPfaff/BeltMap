from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

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
    ("belt_region", "BELT_REGION", "region", (("belt_region",), ("belt", "region")), "Belt crop as top,left,height,width. Omit to use the full frame.", "TOP,LEFT,HEIGHT,WIDTH"),
    ("belt_velocity_px_per_frame", "BELT_VELOCITY_PX_PER_FRAME", "velocity", (("belt_velocity_px_per_frame",), ("belt", "velocity_px_per_frame")), "Signed belt image velocity, or 'auto'. Numeric values use --belt-velocity-frame-unit when frames are strided.", "PX_PER_FRAME|auto"),
    ("belt_velocity_frame_unit", "BELT_VELOCITY_FRAME_UNIT", "path", (("belt_velocity_frame_unit",), ("belt", "velocity_frame_unit")), "Frame unit for a supplied belt velocity: selected_frame or source_frame.", "UNIT"),
    ("belt_period_px", "BELT_PERIOD_PX", "int", (("belt_period_px",), ("belt", "period_px")), "Optional belt circumference/period in pixels.", "PX"),
    ("detection_threshold", "DETECTION_THRESHOLD", "float", (("detection_threshold",), ("detection", "threshold")), "Threshold on normalized residuals for bright particles.", "Z"),
    ("detection_mode", "DETECTION_MODE", "path", (("detection_mode",), ("detection", "mode")), "Detection residual polarity: positive, negative, or absolute.", "MODE"),
    ("detection_low_threshold", "DETECTION_LOW_THRESHOLD", "float", (("detection_low_threshold",), ("detection", "low_threshold")), "Optional lower hysteresis threshold for final detection. Use 0 to disable.", "Z"),
    ("min_area_px", "MIN_AREA_PX", "int", (("min_area_px",), ("detection", "min_area_px")), "Minimum connected-component area for detections.", "PX"),
    ("detection_max_area_px", "DETECTION_MAX_AREA_PX", "int", (("detection_max_area_px",), ("detection", "max_area_px")), "Optional maximum connected-component area for detections. Use 0 to disable.", "PX"),
    ("detection_min_bbox_width_px", "DETECTION_MIN_BBOX_WIDTH_PX", "int", (("detection_min_bbox_width_px",), ("detection", "min_bbox_width_px")), "Optional minimum detection bounding-box width. Use 0 to disable.", "PX"),
    ("detection_min_bbox_height_px", "DETECTION_MIN_BBOX_HEIGHT_PX", "int", (("detection_min_bbox_height_px",), ("detection", "min_bbox_height_px")), "Optional minimum detection bounding-box height. Use 0 to disable.", "PX"),
    ("detection_max_bbox_aspect_ratio", "DETECTION_MAX_BBOX_ASPECT_RATIO", "float", (("detection_max_bbox_aspect_ratio",), ("detection", "max_bbox_aspect_ratio")), "Optional maximum detection bounding-box aspect ratio. Use 0 to disable.", "RATIO"),
    ("detection_min_bbox_extent", "DETECTION_MIN_BBOX_EXTENT", "float", (("detection_min_bbox_extent",), ("detection", "min_bbox_extent")), "Optional minimum area/bounding-box-area extent. Use 0 to disable.", "FRACTION"),
    ("residual_noise_radius_px", "RESIDUAL_NOISE_RADIUS_PX", "int", (("residual_noise_radius_px",), ("residual", "noise_radius_px")), "Local residual-noise box radius.", "PX"),
    ("residual_clip_sigma", "RESIDUAL_CLIP_SIGMA", "float", (("residual_clip_sigma",), ("residual", "clip_sigma")), "Symmetric residual clipping level for local-noise estimation. Use 0 to disable.", "SIGMA"),
    ("residual_min_noise", "RESIDUAL_MIN_NOISE", "float", (("residual_min_noise",), ("residual", "min_noise")), "Minimum local residual-noise scale.", "GRAY"),
    ("residual_noise_exclusion_sigma", "RESIDUAL_NOISE_EXCLUSION_SIGMA", "float", (("residual_noise_exclusion_sigma",), ("residual", "noise_exclusion_sigma")), "Positive-residual threshold for excluding particle-like pixels from local-noise estimation. Use 0 to disable.", "SIGMA"),
    ("residual_noise_exclusion_radius_px", "RESIDUAL_NOISE_EXCLUSION_RADIUS_PX", "int", (("residual_noise_exclusion_radius_px",), ("residual", "noise_exclusion_radius_px")), "Dilation radius around particle-like pixels excluded from local-noise windows.", "PX"),
    ("min_track_length", "MIN_TRACK_LENGTH", "int", (("min_track_length",), ("tracking", "min_track_length")), "Minimum detections per track for velocity estimates.", "N"),
    ("max_match_distance_px", "MAX_MATCH_DISTANCE_PX", "float", (("max_match_distance_px",), ("tracking", "max_match_distance_px")), "Optional tracking match distance. Omit to derive it from belt speed.", "PX"),
    ("tracking_assignment_method", "TRACKING_ASSIGNMENT_METHOD", "path", (("tracking_assignment_method",), ("tracking", "assignment_method")), "Tracking association method: global or greedy.", "METHOD"),
    ("tracking_area_cost_weight_px", "TRACKING_AREA_COST_WEIGHT_PX", "float", (("tracking_area_cost_weight_px",), ("tracking", "area_cost_weight_px")), "Additional assignment cost for log area-ratio changes.", "PX"),
    ("tracking_signal_cost_weight_px", "TRACKING_SIGNAL_COST_WEIGHT_PX", "float", (("tracking_signal_cost_weight_px",), ("tracking", "signal_cost_weight_px")), "Additional assignment cost for log signal-ratio changes.", "PX"),
    ("tracking_lateral_cost_weight", "TRACKING_LATERAL_COST_WEIGHT", "float", (("tracking_lateral_cost_weight",), ("tracking", "lateral_cost_weight")), "Additional assignment cost per pixel of lateral residual motion.", "WEIGHT"),
    ("tracking_max_area_ratio", "TRACKING_MAX_AREA_RATIO", "float", (("tracking_max_area_ratio",), ("tracking", "max_area_ratio")), "Optional maximum frame-to-frame detection area ratio. Use 0 to disable.", "RATIO"),
    ("track_filter_min_length", "TRACK_FILTER_MIN_LENGTH", "int", (("track_filter_min_length",), ("track_filter", "min_length")), "Minimum detections per accepted filtered track.", "N"),
    ("track_filter_min_velocity_ratio_y", "TRACK_FILTER_MIN_VELOCITY_RATIO_Y", "float", (("track_filter_min_velocity_ratio_y",), ("track_filter", "min_velocity_ratio_y")), "Minimum accepted particle/belt vertical velocity ratio.", "RATIO"),
    ("track_filter_max_velocity_ratio_y", "TRACK_FILTER_MAX_VELOCITY_RATIO_Y", "float", (("track_filter_max_velocity_ratio_y",), ("track_filter", "max_velocity_ratio_y")), "Maximum accepted particle/belt vertical velocity ratio.", "RATIO"),
    ("track_filter_max_abs_x_velocity_px_per_frame", "TRACK_FILTER_MAX_ABS_X_VELOCITY_PX_PER_FRAME", "float", (("track_filter_max_abs_x_velocity_px_per_frame",), ("track_filter", "max_abs_x_velocity_px_per_frame")), "Optional maximum accepted absolute lateral velocity. Use 0 to disable.", "PX_PER_FRAME"),
    ("max_frames", "MAX_FRAMES", "int", (("max_frames",), ("frames", "max_frames")), "Maximum number of selected frames to process. Use 0 for all frames.", "N"),
    ("frame_stride", "FRAME_STRIDE", "int", (("frame_stride",), ("frames", "stride")), "Process every Nth frame after natural sorting.", "N"),
    ("map_sample_frames", "MAP_SAMPLE_FRAMES", "int", (("map_sample_frames",), ("map", "sample_frames")), "Number of frames sampled to build the belt map.", "N"),
    ("map_reconstruction_trim_fraction", "MAP_RECONSTRUCTION_TRIM_FRACTION", "float", (("map_reconstruction_trim_fraction",), ("map", "reconstruction_trim_fraction")), "Symmetric per-pixel trim fraction for robust belt-map reconstruction. Use 0 for the arithmetic mean.", "FRACTION"),
    ("map_fractional_splat", "MAP_FRACTIONAL_SPLAT", "bool", (("map_fractional_splat",), ("map", "fractional_splat")), "Use fractional row weights when accumulating belt-map pixels.", None),
    ("map_mask_iterations", "MAP_MASK_ITERATIONS", "int", (("map_mask_iterations",), ("map", "mask_iterations")), "Particle-mask refinement iterations while building the belt map.", "N"),
    ("map_particle_mask_threshold", "MAP_PARTICLE_MASK_THRESHOLD", "float", (("map_particle_mask_threshold",), ("map", "particle_mask_threshold")), "Strong threshold used for particle masking during map building.", "Z"),
    ("map_particle_mask_mode", "MAP_PARTICLE_MASK_MODE", "path", (("map_particle_mask_mode",), ("map", "particle_mask_mode")), "Map-building particle mask mode: positive, absolute, or hysteresis_abs.", "MODE"),
    ("map_particle_mask_grow_threshold", "MAP_PARTICLE_MASK_GROW_THRESHOLD", "float", (("map_particle_mask_grow_threshold",), ("map", "particle_mask_grow_threshold")), "Lower absolute-residual threshold used to grow hysteresis map masks.", "Z"),
    ("map_particle_mask_dilation_px", "MAP_PARTICLE_MASK_DILATION_PX", "int", (("map_particle_mask_dilation_px",), ("map", "particle_mask_dilation_px")), "Morphological dilation radius for map-building particle masks.", "PX"),
    ("map_particle_mask_margin_px", "MAP_PARTICLE_MASK_MARGIN_PX", "int", (("map_particle_mask_margin_px",), ("map", "particle_mask_margin_px")), "Safety margin around detected particle boxes during map building.", "PX"),
    ("map_particle_mask_min_area_px", "MAP_PARTICLE_MASK_MIN_AREA_PX", "int", (("map_particle_mask_min_area_px",), ("map", "particle_mask_min_area_px")), "Minimum component area for map-building particle masks.", "PX"),
    ("map_aggregation", "MAP_AGGREGATION", "path", (("map_aggregation",), ("map", "aggregation")), "Belt-map aggregation method: mean or huber.", "METHOD"),
    ("map_robust_iterations", "MAP_ROBUST_ITERATIONS", "int", (("map_robust_iterations",), ("map", "robust_iterations")), "Robust Huber refinement iterations for aggregation='huber'.", "N"),
    ("map_robust_huber_delta", "MAP_ROBUST_HUBER_DELTA", "float", (("map_robust_huber_delta",), ("map", "robust_huber_delta")), "Huber cutoff in robust residual scale units.", "SIGMA"),
    ("map_robust_min_scale", "MAP_ROBUST_MIN_SCALE", "float", (("map_robust_min_scale",), ("map", "robust_min_scale")), "Minimum residual scale for Huber weighting.", "GRAY"),
    ("static_noise_sample_frames", "STATIC_NOISE_SAMPLE_FRAMES", "int", (("static_noise_sample_frames",), ("static_noise", "sample_frames")), "Number of residual frames sampled to learn a static residual-noise map. Use 0 to disable.", "N"),
    ("static_noise_min_scale", "STATIC_NOISE_MIN_SCALE", "float", (("static_noise_min_scale",), ("static_noise", "min_scale")), "Minimum per-pixel static residual-noise floor.", "GRAY"),
    ("static_noise_mask_threshold", "STATIC_NOISE_MASK_THRESHOLD", "float", (("static_noise_mask_threshold",), ("static_noise", "mask_threshold")), "Optional normalized residual threshold for masking particles while learning static noise. Use 0 to disable.", "Z"),
    ("static_noise_mask_margin_px", "STATIC_NOISE_MASK_MARGIN_PX", "int", (("static_noise_mask_margin_px",), ("static_noise", "mask_margin_px")), "Safety margin around particle boxes while learning static noise.", "PX"),
    ("static_noise_mask_min_area_px", "STATIC_NOISE_MASK_MIN_AREA_PX", "int", (("static_noise_mask_min_area_px",), ("static_noise", "mask_min_area_px")), "Minimum component area for particle masks while learning static noise.", "PX"),
    ("static_background_sample_frames", "STATIC_BACKGROUND_SAMPLE_FRAMES", "int", (("static_background_sample_frames",), ("static_background", "sample_frames")), "Number of residual frames sampled to learn an additive image-fixed background. Use 0 to disable.", "N"),
    ("static_background_mask_threshold", "STATIC_BACKGROUND_MASK_THRESHOLD", "float", (("static_background_mask_threshold",), ("static_background", "mask_threshold")), "Optional normalized residual threshold for masking particles while learning static background. Use 0 to disable.", "Z"),
    ("static_background_mask_margin_px", "STATIC_BACKGROUND_MASK_MARGIN_PX", "int", (("static_background_mask_margin_px",), ("static_background", "mask_margin_px")), "Safety margin around particle boxes while learning static background.", "PX"),
    ("static_background_mask_min_area_px", "STATIC_BACKGROUND_MASK_MIN_AREA_PX", "int", (("static_background_mask_min_area_px",), ("static_background", "mask_min_area_px")), "Minimum component area for particle masks while learning static background.", "PX"),
    ("recurrent_artifact_min_revolutions", "RECURRENT_ARTIFACT_MIN_REVOLUTIONS", "int", (("recurrent_artifact_min_revolutions",), ("recurrent_artifact", "min_revolutions")), "Minimum distinct belt revolutions required to mark recurrent artifact pixels. Use 0 to disable building unless a reused map is set.", "N"),
    ("recurrent_artifact_margin_px", "RECURRENT_ARTIFACT_MARGIN_PX", "int", (("recurrent_artifact_margin_px",), ("recurrent_artifact", "margin_px")), "Safety margin around detection boxes when accumulating recurrent artifacts.", "PX"),
    ("recurrent_artifact_max_overlap_fraction", "RECURRENT_ARTIFACT_MAX_OVERLAP_FRACTION", "float", (("recurrent_artifact_max_overlap_fraction",), ("recurrent_artifact", "max_overlap_fraction")), "Reject detections whose belt-coordinate bbox overlaps recurrent artifacts above this fraction.", "FRACTION"),
    ("recurrent_artifact_min_recurrence_probability", "RECURRENT_ARTIFACT_MIN_RECURRENCE_PROBABILITY", "float", (("recurrent_artifact_min_recurrence_probability",), ("recurrent_artifact", "min_recurrence_probability")), "Minimum exposure-normalized recurrence probability required before pixels contribute to the recurrent-artifact prior.", "PROBABILITY"),
    ("recurrent_artifact_mode", "RECURRENT_ARTIFACT_MODE", "path", (("recurrent_artifact_mode",), ("recurrent_artifact", "mode")), "Recurrent artifact filter mode: hard rejects by overlap; soft keeps strong peaks; probabilistic uses exposure-normalized recurrence probabilities.", "MODE"),
    ("recurrent_artifact_soft_penalty_weight", "RECURRENT_ARTIFACT_SOFT_PENALTY_WEIGHT", "float", (("recurrent_artifact_soft_penalty_weight",), ("recurrent_artifact", "soft_penalty_weight")), "Soft-mode peak-signal penalty per artifact-overlap fraction.", "WEIGHT"),
    ("velocity_search_radius_px", "VELOCITY_SEARCH_RADIUS_PX", "int", (("velocity_search_radius_px",), ("auto_velocity", "search_radius_px")), "Max vertical shift searched during automatic belt-velocity estimation.", "PX"),
    ("velocity_estimation_pairs", "VELOCITY_ESTIMATION_PAIRS", "int", (("velocity_estimation_pairs",), ("auto_velocity", "estimation_pairs")), "Number of adjacent frame pairs used for automatic velocity estimation.", "N"),
    ("auto_velocity_min_abs_px_per_frame", "AUTO_VELOCITY_MIN_ABS_PX_PER_FRAME", "float", (("auto_velocity_min_abs_px_per_frame",), ("auto_velocity", "min_abs_px_per_frame")), "Minimum accepted absolute auto-estimated belt velocity.", "PX_PER_FRAME"),
    ("auto_velocity_max_edge_fraction", "AUTO_VELOCITY_MAX_EDGE_FRACTION", "float", (("auto_velocity_max_edge_fraction",), ("auto_velocity", "max_edge_fraction")), "Maximum accepted fraction of auto-velocity shifts that hit the search edge.", "FRACTION"),
    ("allow_full_frame_auto_velocity", "ALLOW_FULL_FRAME_AUTO_VELOCITY", "bool", (("allow_full_frame_auto_velocity",), ("auto_velocity", "allow_full_frame")), "Allow automatic belt-velocity estimation on a full-frame belt region.", None),
    ("registration_search_radius_px", "REGISTRATION_SEARCH_RADIUS_PX", "float", (("registration_search_radius_px",), ("registration", "search_radius_px")), "Phase-registration search radius in pixels.", "PX"),
    ("registration_search_step_px", "REGISTRATION_SEARCH_STEP_PX", "float", (("registration_search_step_px",), ("registration", "search_step_px")), "Phase-registration search step in pixels.", "PX"),
    ("phase_refinement_iterations", "PHASE_REFINEMENT_ITERATIONS", "int", (("phase_refinement_iterations",), ("phase_refinement", "iterations")), "Phase-feedback map-refinement iterations. Use 0 to disable.", "N"),
    ("phase_refinement_min_score", "PHASE_REFINEMENT_MIN_SCORE", "float", (("phase_refinement_min_score",), ("phase_refinement", "min_score")), "Minimum registration score accepted for phase-feedback refinement.", "SCORE"),
    ("phase_refinement_max_abs_correction_px", "PHASE_REFINEMENT_MAX_ABS_CORRECTION_PX", "float", (("phase_refinement_max_abs_correction_px",), ("phase_refinement", "max_abs_correction_px")), "Maximum absolute registration correction accepted for phase-feedback refinement. Use 0 to disable this gate.", "PX"),
    ("phase_refinement_smoothing_window_frames", "PHASE_REFINEMENT_SMOOTHING_WINDOW_FRAMES", "int", (("phase_refinement_smoothing_window_frames",), ("phase_refinement", "smoothing_window_frames")), "Rolling-median smoothing window for accepted phase corrections.", "N"),
    ("progress_interval_frames", "PROGRESS_INTERVAL_FRAMES", "int", (("progress_interval_frames",), ("progress", "interval_frames")), "Print progress every N frames during long stages.", "N"),
    ("partial_output_interval_frames", "PARTIAL_OUTPUT_INTERVAL_FRAMES", "int", (("partial_output_interval_frames",), ("progress", "partial_output_interval_frames")), "Write partial CSV outputs every N processed frames. Use 0 for final only.", "N"),
    ("debug_residual_preview_frames", "DEBUG_RESIDUAL_PREVIEW_FRAMES", "int", (("debug_residual_preview_frames",), ("debug", "residual_preview_frames")), "Save residual PNG previews for the first N frames.", "N"),
    ("debug_residual_preview_interval_frames", "DEBUG_RESIDUAL_PREVIEW_INTERVAL_FRAMES", "int", (("debug_residual_preview_interval_frames",), ("debug", "residual_preview_interval_frames")), "Also save residual previews every N frames. Use 0 to disable.", "N"),
)

OPTION_BY_NAME = {spec[0]: spec for spec in OPTION_SPECS}
OPTION_BY_ENV = {spec[1]: spec for spec in OPTION_SPECS}
CONFIG_KEY_TO_NAME = {key: spec[0] for spec in OPTION_SPECS for key in spec[3]}

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

[residual]
noise_radius_px = 15
clip_sigma = 5.0
min_noise = 1e-6
noise_exclusion_sigma = 4.0
noise_exclusion_radius_px = 2

[tracking]
min_track_length = 2
# max_match_distance_px = 90.0
assignment_method = "global"
area_cost_weight_px = 0.0
signal_cost_weight_px = 0.0
lateral_cost_weight = 0.0
max_area_ratio = 0.0

[track_filter]
min_length = 5
min_velocity_ratio_y = 0.0
max_velocity_ratio_y = 1.1
max_abs_x_velocity_px_per_frame = 0.0

[map]
sample_frames = 120
reconstruction_trim_fraction = 0.1
fractional_splat = true
mask_iterations = 1
particle_mask_mode = "positive"
particle_mask_threshold = 5.0
particle_mask_grow_threshold = 2.0
particle_mask_dilation_px = 0
particle_mask_margin_px = 8
particle_mask_min_area_px = 4
aggregation = "mean"
robust_iterations = 1
robust_huber_delta = 3.0
robust_min_scale = 1.0

[static_noise]
sample_frames = 0
min_scale = 0.0
mask_threshold = 0.0
mask_margin_px = 8
mask_min_area_px = 4

[static_background]
sample_frames = 0
mask_threshold = 0.0
mask_margin_px = 8
mask_min_area_px = 4

[recurrent_artifact]
min_revolutions = 0
margin_px = 2
max_overlap_fraction = 0.3
min_recurrence_probability = 0.0
mode = "hard"
soft_penalty_weight = 1.0

[auto_velocity]
search_radius_px = 90
estimation_pairs = 100
min_abs_px_per_frame = 0.25
max_edge_fraction = 0.2
allow_full_frame = false

[registration]
search_radius_px = 8.0
search_step_px = 0.5

[phase_refinement]
iterations = 0
min_score = 0.0
max_abs_correction_px = 0.0
smoothing_window_frames = 25

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
        if name is None:
            raise ValueError(f"Unknown config option {'.'.join(key_path)!r}")
        normalized = normalize_value(name, raw_value)
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


def resolve_driver_env(namespace: argparse.Namespace, environ: Mapping[str, str] | None = None) -> tuple[dict[str, str], dict[str, Any]]:
    current_environ = os.environ if environ is None else environ
    merged: dict[str, str] = {}
    sources: dict[str, str] = {}
    for layer_values, layer_sources in (values_from_config(namespace.config), values_from_environment(current_environ), values_from_args(namespace)):
        merged.update(layer_values)
        sources.update(layer_sources)
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
