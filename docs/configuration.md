# Configuration reference

This page documents the runtime options accepted by `beltmap-apply` and passed
through to the packaged BeltMap image-sequence driver. It is intended as the
stable reference for users who do not want to inspect `beltmap-apply --help` or
read the driver source.

`beltmap-apply` accepts values from three layers. Later layers override earlier
layers:

```text
config file < environment variables < explicit CLI flags
```

Use this command to inspect the effective driver environment without processing
images:

```bash
beltmap-apply --config beltmap.toml --dry-run
```

The resolved configuration is also written to `config_resolved.json` in the
configured output directory before a normal run starts. Running the legacy
`scripts/apply_beltmap_to_images.py` script directly bypasses the config-file
parser and uses environment variables only.

## Config-file formats

`beltmap-apply --config` accepts TOML or JSON. TOML config files can be grouped
into sections:

```toml
[paths]
image_dir = "data/images"
output_dir = "outputs"

[belt]
region = [0, 220, 1330, 1800]
velocity_px_per_frame = 59.3
period_px = 14723
```

The same options can also be written as flat keys, for example:

```toml
image_dir = "data/images"
output_dir = "outputs"
belt_region = [0, 220, 1330, 1800]
belt_velocity_px_per_frame = 59.3
belt_period_px = 14723
```

Generate a full template with:

```bash
beltmap-apply --write-config-template beltmap.toml
```

The table below uses the sectioned TOML key names. The corresponding flat key is
usually the CLI flag name with leading dashes removed and hyphens replaced by
underscores.

## Value conventions

- `belt.region` is `top,left,height,width` in environment variables and CLI
  flags, and either `[top, left, height, width]` or a comma-separated string in
  config files.
- `belt.velocity_px_per_frame` is either a signed number in pixels per frame or
  the string `"auto"`. Automatic velocity is estimated between selected frames
  after `frames.stride` has been applied. If a numeric velocity is supplied and
  `frames.stride > 1`, set `belt.velocity_frame_unit` to `"selected_frame"` when
  the value is already per processed/selected frame, or `"source_frame"` when it
  was measured between adjacent original input frames. Source-frame velocities
  are multiplied by `frames.stride` before map building, phase prediction,
  tracking priors, and velocity-ratio outputs. Positive velocity means the belt
  texture moves downward in image coordinates.
- Boolean environment values accept `1`, `true`, `yes`, `on`, `0`, `false`,
  `no`, and `off`.
- Empty optional environment variables are ignored by `beltmap-apply`.
- The defaults below are the effective driver defaults when no config,
  environment variable, or CLI flag supplies the option. The generated template
  may intentionally contain example values, such as a non-full-frame belt
  region.

## Option table

| Config key | Environment variable | CLI flag | Default when unset | Unit | Meaning |
|---|---|---|---:|---|---|
| `paths.image_dir` | `BELTMAP_IMAGE_DIR` | `--image-dir` | `data/images` | path | Directory searched recursively for input images. Supported extensions are `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, and `.bmp`. |
| `paths.output_dir` | `BELTMAP_OUTPUT_DIR` | `--output-dir` | `outputs` | path | Directory where maps, CSV files, metadata, progress logs, previews, and validation artifacts are written. |
| `reuse.belt_map_path` | `REUSE_BELT_MAP_PATH` | `--reuse-belt-map-path` | unset | path | Existing `belt_map.npy` to reuse instead of rebuilding the belt map. The driver writes a copy plus fresh detection, tracking, velocity, metadata, and preview outputs into `paths.output_dir`. |
| `reuse.phase_estimates_path` | `REUSE_PHASE_ESTIMATES_PATH` | `--reuse-phase-estimates-path` | unset | path | Existing `phase_estimates.csv` to reuse with `reuse.belt_map_path`. If unset, phases are recomputed by local registration against the reused map. |
| `reuse.static_noise_path` | `REUSE_STATIC_NOISE_PATH` | `--reuse-static-noise-path` | unset | path | Existing `static_noise.npy` to reuse as a per-pixel residual-noise floor during detection. |
| `reuse.static_background_path` | `REUSE_STATIC_BACKGROUND_PATH` | `--reuse-static-background-path` | unset | path | Existing `static_background.npy` additive image-fixed residual map to subtract during detection. |
| `reuse.recurrent_artifact_map_path` | `REUSE_RECURRENT_ARTIFACT_MAP_PATH` | `--reuse-recurrent-artifact-map-path` | unset | path | Existing `recurrent_artifact_map.npy` to reuse for recurrent artifact filtering. This enables the filter without rebuilding the recurrent artifact map. |
| `frames.max_frames` | `MAX_FRAMES` | `--max-frames` | `0` | frames | Maximum number of selected frames to process after sorting and striding. `0` means process all selected frames. |
| `frames.stride` | `FRAME_STRIDE` | `--frame-stride` | `1` | frames | Process every Nth frame after natural filename sorting. Must be at least 1. |
| `belt.region` | `BELT_REGION` | `--belt-region` | full frame | px | Belt crop as `top,left,height,width`. Coordinates are full-frame image coordinates. Omit only when the full frame is belt texture. |
| `belt.velocity_px_per_frame` | `BELT_VELOCITY_PX_PER_FRAME` | `--belt-velocity-px-per-frame` | `auto` | px/frame | Signed vertical belt texture velocity, or `auto`. Numeric values use `belt.velocity_frame_unit` when `frames.stride > 1`; `auto` is estimated on selected-frame pairs. |
| `belt.velocity_frame_unit` | `BELT_VELOCITY_FRAME_UNIT` | `--belt-velocity-frame-unit` | contextual | unit | Required when `belt.velocity_px_per_frame` is numeric and `frames.stride > 1`. Use `selected_frame` when the supplied velocity is already per processed/selected frame. Use `source_frame` when it is per adjacent original input frame; the driver multiplies it by `frames.stride`. |
| `belt.period_px` | `BELT_PERIOD_PX` | `--belt-period-px` | unset | px | Belt circumference/period in belt-map pixels. If unset or non-positive, the driver builds a finite map covering the selected sequence phase range. |
| `detection.threshold` | `DETECTION_THRESHOLD` | `--detection-threshold` | `5.0` | z | Threshold on normalized residuals for final bright-particle detection. |
| `detection.mode` | `DETECTION_MODE` | `--detection-mode` | `positive` | mode | Detection residual polarity: `positive`, `negative`, or `absolute`. Legacy config-file values `threshold`, `hysteresis`, and `hysteresis_abs` are accepted as aliases for `positive`, `positive`, and `absolute`; use `detection.low_threshold` to enable hysteresis growth. |
| `detection.low_threshold` | `DETECTION_LOW_THRESHOLD` | `--detection-low-threshold` | `0` | z | Optional lower hysteresis threshold for final detection. `0` disables hysteresis. |
| `detection.min_area_px` | `MIN_AREA_PX` | `--min-area-px` | `4` | px | Minimum connected-component area for final particle detections. Must be at least 1. |
| `detection.max_area_px` | `DETECTION_MAX_AREA_PX` | `--detection-max-area-px` | `0` | px | Optional maximum connected-component area. `0` disables this gate. |
| `detection.min_bbox_width_px` | `DETECTION_MIN_BBOX_WIDTH_PX` | `--detection-min-bbox-width-px` | `0` | px | Optional minimum component bounding-box width. `0` disables this gate. |
| `detection.min_bbox_height_px` | `DETECTION_MIN_BBOX_HEIGHT_PX` | `--detection-min-bbox-height-px` | `0` | px | Optional minimum component bounding-box height. `0` disables this gate. |
| `detection.max_bbox_aspect_ratio` | `DETECTION_MAX_BBOX_ASPECT_RATIO` | `--detection-max-bbox-aspect-ratio` | `0` | ratio | Optional maximum bounding-box aspect ratio `max(height/width, width/height)`. `0` disables this gate. |
| `detection.min_bbox_extent` | `DETECTION_MIN_BBOX_EXTENT` | `--detection-min-bbox-extent` | `0` | fraction | Optional minimum component extent `area / (bbox_width * bbox_height)`. `0` disables this gate. |
| `detection.split_merged_components` | `DETECTION_SPLIT_MERGED_COMPONENTS` | `--detection-split-merged-components` / `--no-detection-split-merged-components` | `false` | bool | Split connected components at narrow row/column projection valleys before writing detections. Useful for dense particles joined by a weak bridge. |
| `detection.split_min_projection_gap_px` | `DETECTION_SPLIT_MIN_PROJECTION_GAP_PX` | `--detection-split-min-projection-gap-px` | `2` | px | Minimum valley width used by the merged-component splitter. |
| `detection.split_min_component_area_px` | `DETECTION_SPLIT_MIN_COMPONENT_AREA_PX` | `--detection-split-min-component-area-px` | `0` | px | Minimum area for each side of a split. `0` reuses `detection.min_area_px`. |
| `residual.noise_radius_px` | `RESIDUAL_NOISE_RADIUS_PX` | `--residual-noise-radius-px` | `15` | px | Local box radius used to estimate the residual-noise scale. |
| `residual.clip_sigma` | `RESIDUAL_CLIP_SIGMA` | `--residual-clip-sigma` | `5.0` | sigma | Symmetric residual clipping level before local variance estimation. `0` disables clipping. |
| `residual.min_noise` | `RESIDUAL_MIN_NOISE` | `--residual-min-noise` | `1e-6` | gray | Minimum local residual-noise scale. Must be positive. |
| `residual.noise_exclusion_sigma` | `RESIDUAL_NOISE_EXCLUSION_SIGMA` | `--residual-noise-exclusion-sigma` | `4.0` | sigma | Positive-residual threshold for excluding particle-like pixels from local-noise estimation. `0` disables this exclusion. |
| `residual.noise_exclusion_radius_px` | `RESIDUAL_NOISE_EXCLUSION_RADIUS_PX` | `--residual-noise-exclusion-radius-px` | `2` | px | Dilation radius around particle-like pixels excluded from local-noise windows. |
| `photometric.enabled` | `PHOTOMETRIC_ENABLED` | `--photometric-enabled` / `--no-photometric-enabled` | `false` | bool | Fit a robust per-frame gain/offset correction before residual detection. |
| `photometric.trim_fraction` | `PHOTOMETRIC_TRIM_FRACTION` | `--photometric-trim-fraction` | `0.05` | fraction | Fraction of largest photometric-fit residuals trimmed on each iteration. Must be in `[0, 0.5)`. |
| `photometric.max_iterations` | `PHOTOMETRIC_MAX_ITERATIONS` | `--photometric-max-iterations` | `3` | iterations | Maximum robust photometric gain/offset fitting iterations. |
| `photometric.min_pixels` | `PHOTOMETRIC_MIN_PIXELS` | `--photometric-min-pixels` | `128` | pixels | Minimum valid pixels required for a photometric gain/offset fit. |
| `tracking.min_track_length` | `MIN_TRACK_LENGTH` | `--min-track-length` | `2` | detections | Minimum number of detections required before a particle track contributes a velocity row. Must be at least 1 at driver parsing and at least 2 for velocity estimation. |
| `tracking.max_match_distance_px` | `MAX_MATCH_DISTANCE_PX` | `--max-match-distance-px` | `max(5, 1.5 * abs(belt_velocity))` | px | Maximum frame-to-frame nearest-neighbor association distance for tracking. Leave unset to derive it from the belt speed. |
| `tracking.max_frame_gap` | `TRACKING_MAX_FRAME_GAP` | `--tracking-max-frame-gap` | `1.0` | selected frames | Maximum selected-frame gap allowed when linking detections into one track. Raise this to bridge occasional missed detections. |
| `tracking.velocity_fit_method` | `TRACKING_VELOCITY_FIT_METHOD` | `--tracking-velocity-fit-method` | `linear` | method | Velocity slope estimator: `linear` for least-squares or `theil_sen` for a robust median pairwise slope. |
| `track_filter.min_length` | `TRACK_FILTER_MIN_LENGTH` | `--track-filter-min-length` | `max(5, tracking.min_track_length)` | detections | Minimum detections per accepted filtered velocity row. Raw `velocities.csv` is not modified. |
| `track_filter.min_velocity_ratio_y` | `TRACK_FILTER_MIN_VELOCITY_RATIO_Y` | `--track-filter-min-velocity-ratio-y` | `0.0` | ratio | Minimum accepted `velocity_ratio_y` for `filtered_velocities.csv`. |
| `track_filter.max_velocity_ratio_y` | `TRACK_FILTER_MAX_VELOCITY_RATIO_Y` | `--track-filter-max-velocity-ratio-y` | `1.1` | ratio | Maximum accepted `velocity_ratio_y` for `filtered_velocities.csv`. |
| `track_filter.max_abs_x_velocity_px_per_frame` | `TRACK_FILTER_MAX_ABS_X_VELOCITY_PX_PER_FRAME` | `--track-filter-max-abs-x-velocity-px-per-frame` | `0` | px/frame | Optional lateral-velocity gate. `0` disables this gate. |
| `map.sample_frames` | `MAP_SAMPLE_FRAMES` | `--map-sample-frames` | `120` | frames | Number of frames sampled across the selected sequence to reconstruct the belt map. Must be at least 1. |
| `map.sample_strategy` | `MAP_SAMPLE_STRATEGY` | `--map-sample-strategy` | `uniform` | strategy | Alias for `map.sampling_strategy`; accepts `uniform`, `phase_coverage`, or `adaptive_phase_coverage`. |
| `map.adaptive_candidate_frames` | `MAP_ADAPTIVE_CANDIDATE_FRAMES` | `--map-adaptive-candidate-frames` | `0` | frames | Candidate pool size for adaptive sampling. `0` considers all selected frames. |
| `map.sampling_strategy` | `MAP_SAMPLING_STRATEGY` | `--map-sampling-strategy` | `uniform` | mode | Frame sampling strategy for map reconstruction. `uniform` preserves the original linspace sampling; `adaptive_phase_coverage` spreads samples across nominal belt-coordinate coverage. |
| `map.reconstruction_trim_fraction` | `MAP_RECONSTRUCTION_TRIM_FRACTION` | `--map-reconstruction-trim-fraction` | `0` | fraction | Symmetric per-pixel trim fraction for robust belt-map reconstruction. Must be in `[0, 0.5)`. |
| `map.fractional_splat` | `MAP_FRACTIONAL_SPLAT` | `--map-fractional-splat` / `--no-map-fractional-splat` | `true` | bool | Use linear fractional row weights when accumulating belt-map pixels. When false, each image row contributes to its nearest belt-map row. |
| `map.mask_iterations` | `MAP_MASK_ITERATIONS` | `--map-mask-iterations` | `1` | passes | Number of particle-masked belt-map refinement passes after the initial provisional map. `0` disables particle masking during map reconstruction. |
| `map.particle_mask_threshold` | `MAP_PARTICLE_MASK_THRESHOLD` | `--map-particle-mask-threshold` | `detection.threshold` | z | Strong residual threshold used to seed particle masks while building the clean belt map. |
| `map.particle_mask_mode` | `MAP_PARTICLE_MASK_MODE` | `--map-particle-mask-mode` | `positive` | mode | Map-building particle-mask mode. Valid values are `positive`, `absolute`, and `hysteresis_abs`. |
| `map.particle_mask_grow_threshold` | `MAP_PARTICLE_MASK_GROW_THRESHOLD` | `--map-particle-mask-grow-threshold` | `2.0` | z | Lower absolute-residual threshold used to grow `hysteresis_abs` map masks from strong seeds. Ignored by `positive` and `absolute`. |
| `map.particle_mask_dilation_px` | `MAP_PARTICLE_MASK_DILATION_PX` | `--map-particle-mask-dilation-px` | `0` | px | Morphological dilation radius for `hysteresis_abs` map masks before applying the rectangular safety margin. `0` disables dilation. |
| `map.particle_mask_margin_px` | `MAP_PARTICLE_MASK_MARGIN_PX` | `--map-particle-mask-margin-px` | `8` | px | Safety margin added around detected or grown particle regions during map reconstruction. |
| `map.particle_mask_min_area_px` | `MAP_PARTICLE_MASK_MIN_AREA_PX` | `--map-particle-mask-min-area-px` | `detection.min_area_px` | px | Minimum component area used for particle masking during map reconstruction. Must be at least 1. |
| `static_noise.sample_frames` | `STATIC_NOISE_SAMPLE_FRAMES` | `--static-noise-sample-frames` | `0` | frames | Number of belt-subtracted residual frames sampled to learn `static_noise.npy`. `0` disables static residual-noise learning. |
| `static_noise.min_scale` | `STATIC_NOISE_MIN_SCALE` | `--static-noise-min-scale` | `0` | gray | Minimum value written into the learned static residual-noise map. |
| `static_noise.mask_threshold` | `STATIC_NOISE_MASK_THRESHOLD` | `--static-noise-mask-threshold` | `0` | z | Optional normalized-residual threshold for masking particle boxes while learning static noise. `0` disables this particle mask. |
| `static_noise.mask_margin_px` | `STATIC_NOISE_MASK_MARGIN_PX` | `--static-noise-mask-margin-px` | `8` | px | Safety margin around particle boxes while learning static noise. Used only when `static_noise.mask_threshold > 0`. |
| `static_noise.mask_min_area_px` | `STATIC_NOISE_MASK_MIN_AREA_PX` | `--static-noise-mask-min-area-px` | `detection.min_area_px` | px | Minimum component area for particle masks while learning static noise. |
| `static_background.sample_frames` | `STATIC_BACKGROUND_SAMPLE_FRAMES` | `--static-background-sample-frames` | `0` | frames | Number of belt-subtracted residual frames sampled to learn `static_background.npy`. `0` disables additive static-background learning. |
| `static_background.mask_threshold` | `STATIC_BACKGROUND_MASK_THRESHOLD` | `--static-background-mask-threshold` | `0` | z | Optional normalized-residual threshold for masking particle boxes while learning the static background. `0` disables this particle mask. |
| `static_background.mask_margin_px` | `STATIC_BACKGROUND_MASK_MARGIN_PX` | `--static-background-mask-margin-px` | `8` | px | Safety margin around particle boxes while learning the static background. Used only when `static_background.mask_threshold > 0`. |
| `static_background.mask_min_area_px` | `STATIC_BACKGROUND_MASK_MIN_AREA_PX` | `--static-background-mask-min-area-px` | `detection.min_area_px` | px | Minimum component area for particle masks while learning the static background. |
| `recurrent_artifact.min_revolutions` | `RECURRENT_ARTIFACT_MIN_REVOLUTIONS` | `--recurrent-artifact-min-revolutions` | `0` | revolutions | Minimum distinct belt revolutions in which a belt-coordinate neighborhood must fire before it is marked as recurrent artifact. `0` disables building this filter unless `reuse.recurrent_artifact_map_path` is set. |
| `recurrent_artifact.margin_px` | `RECURRENT_ARTIFACT_MARGIN_PX` | `--recurrent-artifact-margin-px` | `2` | px | Safety margin around detected component boxes when accumulating recurrent artifacts in belt coordinates. |
| `recurrent_artifact.max_overlap_fraction` | `RECURRENT_ARTIFACT_MAX_OVERLAP_FRACTION` | `--recurrent-artifact-max-overlap-fraction` | `0.3` | fraction | Reject a detection when this fraction of its belt-coordinate bounding box overlaps the recurrent artifact map. |
| `recurrent_artifact.mode` | `RECURRENT_ARTIFACT_MODE` | `--recurrent-artifact-mode` | `hard` | mode | `hard` rejects by overlap alone. `soft` rejects recurrent detections only when their peak residual is weak relative to the detection threshold. |
| `recurrent_artifact.soft_penalty_weight` | `RECURRENT_ARTIFACT_SOFT_PENALTY_WEIGHT` | `--recurrent-artifact-soft-penalty-weight` | `1.0` | weight | Additional soft-mode peak-signal requirement per artifact-overlap fraction. Ignored in `hard` mode. |
| `auto_velocity.search_radius_px` | `VELOCITY_SEARCH_RADIUS_PX` | `--velocity-search-radius-px` | `50` | px | Maximum vertical shift searched for each adjacent-frame pair during automatic belt-velocity estimation. Increase if the belt moves farther than this between frames. |
| `auto_velocity.estimation_pairs` | `VELOCITY_ESTIMATION_PAIRS` | `--velocity-estimation-pairs` | `100` | pairs | Number of adjacent-frame pairs used for automatic belt-velocity estimation, capped by the available sequence length. |
| `auto_velocity.min_abs_px_per_frame` | `AUTO_VELOCITY_MIN_ABS_PX_PER_FRAME` | `--auto-velocity-min-abs-px-per-frame` | `0.25` | px/frame | Minimum accepted absolute value of the auto-estimated belt velocity. Helps reject static-background-dominated crops. |
| `auto_velocity.max_edge_fraction` | `AUTO_VELOCITY_MAX_EDGE_FRACTION` | `--auto-velocity-max-edge-fraction` | `0.2` | fraction | Maximum accepted fraction of adjacent-frame shifts that land near the search-radius edge. Must be in `[0, 1]`. |
| `auto_velocity.allow_full_frame` | `ALLOW_FULL_FRAME_AUTO_VELOCITY` | `--allow-full-frame-auto-velocity` / `--no-allow-full-frame-auto-velocity` | `false` | bool | Allow `belt.velocity_px_per_frame = "auto"` when `belt.region` is the full frame. Keep this false unless the full frame really contains only belt texture. |
| `registration.search_radius_px` | `REGISTRATION_SEARCH_RADIUS_PX` | `--registration-search-radius-px` | `8.0` | px | Local phase-registration search radius around the constant-speed prediction. |
| `registration.search_step_px` | `REGISTRATION_SEARCH_STEP_PX` | `--registration-search-step-px` | `0.5` | px | Local phase-registration candidate spacing. Must be positive. |
| `registration.subpixel_refinement` | `REGISTRATION_SUBPIXEL_REFINEMENT` | `--registration-subpixel-refinement` / `--no-registration-subpixel-refinement` | `true` | bool | Refine the best phase-registration offset with a local quadratic fit. |
| `registration.robust_normalization` | `REGISTRATION_ROBUST_NORMALIZATION` | `--registration-robust-normalization` / `--no-registration-robust-normalization` | `true` | bool | Normalize high-pass registration images by a robust MAD scale. |
| `phase_drift.enabled` | `PHASE_DRIFT_ENABLED` | `--phase-drift-enabled` / `--no-phase-drift-enabled` | `true` | bool | Enable online residual phase-drift compensation during detection. |
| `phase_drift.smoothing_alpha` | `PHASE_DRIFT_SMOOTHING_ALPHA` | `--phase-drift-smoothing-alpha` | `0.15` | alpha | Exponential smoothing factor for accepted drift updates. |
| `phase_drift.min_score` | `PHASE_DRIFT_MIN_SCORE` | `--phase-drift-min-score` | `0.05` | score | Minimum registration score accepted by the online phase-drift filter. |
| `phase_drift.max_abs_residual_correction_px` | `PHASE_DRIFT_MAX_ABS_RESIDUAL_CORRECTION_PX` | `--phase-drift-max-abs-residual-correction-px` | `0` | px | Optional gate on individual residual corrections accepted by the drift filter. `0` disables the gate. |
| `phase_drift.max_abs_px` | `PHASE_DRIFT_MAX_ABS_PX` | `--phase-drift-max-abs-px` | `0` | px | Optional cap on accumulated online drift. `0` disables the cap. |
| `phase_smoothing.window_frames` | `PHASE_SMOOTHING_WINDOW_FRAMES` | `--phase-smoothing-window-frames` | `0` | frames | Optional two-pass phase smoothing. When positive, BeltMap first registers all frames, smooths the correction trajectory, then renders detections from smoothed phases. |
| `phase_smoothing.min_score` | `PHASE_SMOOTHING_MIN_SCORE` | `--phase-smoothing-min-score` | `0` | score | Minimum registration score admitted into the smoothing fit. `0` disables this gate. |
| `phase_smoothing.max_abs_correction_px` | `PHASE_SMOOTHING_MAX_ABS_CORRECTION_PX` | `--phase-smoothing-max-abs-correction-px` | `0` | px | Maximum absolute correction admitted into the smoothing fit. `0` disables this gate. |
| `phase_smoothing.min_support` | `PHASE_SMOOTHING_MIN_SUPPORT` | `--phase-smoothing-min-support` | `3` | frames | Minimum neighboring estimates used for local phase smoothing. |
| `progress.interval_frames` | `PROGRESS_INTERVAL_FRAMES` | `--progress-interval-frames` | `25` | frames | Progress-log interval for long velocity, map-building, and detection stages. Must be at least 1. |
| `progress.partial_output_interval_frames` | `PARTIAL_OUTPUT_INTERVAL_FRAMES` | `--partial-output-interval-frames` | `250` | frames | Interval for writing partial detection and phase CSV outputs during long runs. `0` means final outputs only. |
| `debug.residual_preview_frames` | `DEBUG_RESIDUAL_PREVIEW_FRAMES` | `--debug-residual-preview-frames` | `3` | frames | Save normalized residual PNG previews for the first N processed frames. |
| `debug.residual_preview_interval_frames` | `DEBUG_RESIDUAL_PREVIEW_INTERVAL_FRAMES` | `--debug-residual-preview-interval-frames` | `0` | frames | Also save residual previews every N processed frames. `0` disables interval previews. |

## CLI-only helper flags

These flags configure `beltmap-apply` itself and are not passed to the driver as
environment variables.

| Flag | Meaning |
|---|---|
| `--config PATH` | Read a TOML or JSON config file. Values can be flat or sectioned. |
| `--dry-run` | Print the resolved driver environment and exit without running the image driver. |
| `--print-config` | Print the resolved driver environment before running the image driver. |
| `--write-config-template PATH` | Write a TOML configuration template and exit. |

## Map particle-mask modes

`map.particle_mask_mode` controls how particle-contaminated observations are
excluded while reconstructing `belt_map.npy`.

| Mode | Behavior | Best use |
|---|---|---|
| `positive` | Threshold positive normalized residuals, extract connected components, and expand their bounding boxes. | Bright particles on a darker belt. This is the default and preserves the original behavior. |
| `absolute` | Threshold `abs(normalized_residual)`, extract connected components, and expand their bounding boxes. | Particles that create both dark and bright residuals, but do not need hysteresis growing. |
| `hysteresis_abs` | Find strong absolute-residual seed pixels, grow them through connected lower-threshold absolute-residual regions, remove small regions, optionally dilate, then apply the rectangular margin. | Larger or structured particle artifacts where a strong core should pull in weaker surrounding residuals. |

For `hysteresis_abs`, tune these together:

```toml
[map]
particle_mask_mode = "hysteresis_abs"
particle_mask_threshold = 5.0
particle_mask_grow_threshold = 2.0
particle_mask_dilation_px = 2
particle_mask_margin_px = 8
```

## Map-frame sampling

By default, map reconstruction samples frames uniformly across the selected
sequence. For periodic belt data, a small sample budget can still overrepresent
some belt phases. Set `map.sampling_strategy = "adaptive_phase_coverage"` to
select frames that cover more nominal belt-coordinate bins before filling the
remaining budget:

```toml
[map]
sample_frames = 500
sampling_strategy = "adaptive_phase_coverage"
adaptive_candidate_frames = 3000
```

Use this with stable velocity/period settings. If phase registration diagnostics
show large corrections or boundary hits, fix velocity, crop, or phase-refinement
settings before trusting adaptive coverage.

## Offline phase smoothing

Per-frame registration can be noisy when one frame contains large particles,
scratches, or illumination bursts. `phase_smoothing` runs a two-pass detection
path: first register all frames, then smooth accepted correction estimates with
the existing robust local-linear smoother, then render residuals from those
smoothed phases. This increases runtime because frames are read once for phase
planning and once for residual rendering, so keep it disabled for quick sweeps.

```toml
[phase_smoothing]
window_frames = 25
min_score = 0.05
max_abs_correction_px = 4
min_support = 3
```

## Dense-particle component splitting

When high density causes neighboring particles to be joined by a weak residual
bridge, enable projection-valley splitting before track association:

```toml
[detection]
split_merged_components = true
split_min_projection_gap_px = 1
split_min_component_area_px = 4
```

For velocity estimates with occasional centroid outliers, use the robust slope
fit:

```toml
[tracking]
velocity_fit_method = "theil_sen"
```

## Detection-only reuse mode

Use reuse mode when the belt map is already good and you only want to retune
particle detection, tracking, and velocity extraction:

```toml
[paths]
image_dir = "data/images"
output_dir = "outputs-threshold-3p5"

[reuse]
belt_map_path = "outputs-threshold-5/belt_map.npy"
phase_estimates_path = "outputs-threshold-5/phase_estimates.csv"

[detection]
threshold = 3.5
```

When `reuse.belt_map_path` is set, BeltMap skips map reconstruction and loads
the existing `belt_map.npy`. If a sibling `metadata.json` exists next to that
map, the driver reuses `reference_phase_px` from it; otherwise the reference
phase defaults to `0`.

When `reuse.phase_estimates_path` is also set, the driver uses those per-frame
phases directly. Otherwise it recomputes per-frame phases by registering each
selected frame against the reused map, then proceeds with residual rendering,
detection, tracking, and velocity estimation.

## Residual normalization

By default, BeltMap now excludes strong positive residuals from the local
noise-scale window before computing the normalized residual. This prevents bright
particles from inflating their own local denominator and becoming harder to
detect:

```toml
[residual]
noise_radius_px = 15
clip_sigma = 5.0
noise_exclusion_sigma = 4.0
noise_exclusion_radius_px = 2
```

Set `residual.noise_exclusion_sigma = 0` to recover the previous behavior. The
exclusion mask is used only to estimate `local_noise`; the particle pixels remain
valid output pixels and are still normalized by the surrounding local noise.
Static noise maps, when enabled, are still applied afterwards as a per-pixel
floor through `max(local_noise, static_noise)`.

## Photometric gain/offset correction

When `photometric.enabled = true`, the driver fits a robust per-frame line

```text
observed ~= gain * expected_background + offset
```

on valid belt pixels before computing the residual. This is useful when the
clean belt texture is geometrically aligned but brightness changes from exposure,
illumination drift, or LED flicker leave broad positive/negative residuals.
The fit trims the largest residuals so ordinary loose particles are less likely
to dominate the correction.

```toml
[photometric]
enabled = true
trim_fraction = 0.05
max_iterations = 3
min_pixels = 10000
```

The driver writes `photometric_fits.csv` with one row per processed frame. Large
gain/offset excursions or high `rmse_gray` values are useful diagnostics for
bad frames, illumination changes, or crop/registration errors.

## Static residual-noise map

Use `static_noise.sample_frames` to learn image-fixed residual variability after
subtracting the phase-dependent belt map. The driver estimates a robust per-pixel
scale from sampled raw residuals:

```text
static_noise(y, x) = 1.4826 * MAD_t(residual_t(y, x))
```

During detection, the normalized residual becomes:

```text
normalized = residual / max(local_noise, static_noise)
```

This does not subtract a static background. It only raises the normalization
denominator at pixels that repeatedly show image-fixed residual variation, so
fixed illumination/sensor artifacts are less likely to become detections.

```toml
[static_noise]
sample_frames = 500
mask_threshold = 4.0
mask_margin_px = 8
mask_min_area_px = 4
```

The learned `static_noise.npy` can be reused in threshold sweeps:

```toml
[reuse]
belt_map_path = "outputs-good-map/belt_map.npy"
phase_estimates_path = "outputs-good-map/phase_estimates.csv"
static_noise_path = "outputs-good-map/static_noise.npy"
```

## Recurrent artifact suppression

Use recurrent artifact suppression to reject detections that appear at the same
belt-coordinate location in multiple belt revolutions. This targets belt-fixed
scratches or map ghosts that survive ordinary residual thresholding.

```toml
[recurrent_artifact]
min_revolutions = 3
margin_px = 2
max_overlap_fraction = 0.3
mode = "hard"
soft_penalty_weight = 1.0
```

The driver first renders residuals and extracts ordinary detections. It then
maps each detection bounding box into belt coordinates with the frame phase,
counts in how many distinct revolutions each belt-coordinate pixel was touched,
and builds `recurrent_artifact_map.npy` from pixels reaching
`min_revolutions`. Final detection outputs, tracks, and velocities are written
after rejecting components whose belt-coordinate bounding box overlaps that map
by more than `max_overlap_fraction`.

In `mode = "hard"`, that overlap test rejects the detection directly. In
`mode = "soft"`, recurring components are rejected only if their `peak_signal`
does not clear `detection.threshold * (1 + soft_penalty_weight * overlap)`.
This is useful when hard recurrent filtering removes too many strong particles.

To reuse a previously built artifact map during threshold or shape-gate sweeps,
set `reuse.recurrent_artifact_map_path`. In that mode the driver loads the map,
copies it to the output directory, and applies the configured `mode`,
`max_overlap_fraction`, and `soft_penalty_weight` without rebuilding
`recurrent_artifact_counts.npy`.

For short diagnostics with only about two revolutions, use `min_revolutions = 2`.
For full runs with many revolutions, start with `min_revolutions = 3` or higher.

## Shape and scratch gating

Bright belt scratches often survive residual thresholding as thin or sparse
connected components. The detection shape gates reject those components before
tracking while leaving the residual threshold unchanged. For the brick sequence,
a useful diagnostic starting point is:

```toml
[detection]
threshold = 3.5
min_area_px = 4
min_bbox_width_px = 3
min_bbox_height_px = 3
max_bbox_aspect_ratio = 4.0
min_bbox_extent = 0.15
```

These gates are deliberately disabled by default because the right values depend
on particle size, imaging scale, and whether particles are partly cut by the belt
crop boundary.

## Track-level filtering

The detector intentionally writes raw threshold detections and raw velocity rows.
To select tracks that are more consistent with the experiment physics, BeltMap
also writes:

```text
track_scores.csv
filtered_velocities.csv
```

The default filter keeps velocity rows with at least five detections and
`0 <= velocity_ratio_y <= 1.1`. This matches the brick-particle assumption that
particles move in the belt direction but usually do not exceed the belt velocity,
while allowing a small tolerance above 1 for measurement noise. A lateral
velocity gate can be enabled with `track_filter.max_abs_x_velocity_px_per_frame`.

Existing runs can be filtered without rerunning image processing:

```bash
beltmap-filter-tracks \
  --output-dir outputs-threshold-3p5 \
  --min-track-length 5 \
  --min-velocity-ratio-y 0 \
  --max-velocity-ratio-y 1.1
```

## Common configurations

### Explicit belt velocity

Use this when the belt velocity and period are known from calibration or a prior
run:

```toml
[paths]
image_dir = "data/images"
output_dir = "outputs"

[belt]
region = [0, 220, 1330, 1800]
velocity_px_per_frame = 59.3
period_px = 14723
```

### Automatic belt velocity

Use this when the belt crop contains enough moving belt texture and the crop does
not include static background:

```toml
[belt]
region = [0, 220, 1330, 1800]
velocity_px_per_frame = "auto"

[auto_velocity]
search_radius_px = 90
estimation_pairs = 100
min_abs_px_per_frame = 0.25
max_edge_fraction = 0.2
allow_full_frame = false
```

Automatic velocity estimation with a full-frame belt region is rejected by
default because static background can dominate the correlation. Set
`auto_velocity.allow_full_frame = true` only when the full frame truly contains
belt texture and no static background.

### Long run with periodic partial outputs

Use this for large datasets where intermediate CSVs and sparse previews are
useful:

```toml
[frames]
stride = 1
max_frames = 0

[progress]
interval_frames = 25
partial_output_interval_frames = 250

[debug]
residual_preview_frames = 3
residual_preview_interval_frames = 500
```

## Related output files

- `config_resolved.json` records the effective config-file, environment, and CLI
  values used by `beltmap-apply`.
- `metadata.json` records run-level values after the driver has parsed defaults,
  estimated automatic quantities, and processed the sequence.
- `progress.jsonl` records stage-wise progress events during long runs.

See [output schemas](outputs.md) for the file-by-file output reference.
