# BeltMap output schemas

This page documents the files written by the BeltMap image-sequence driver and
by the `beltmap-apply` command. It is intended as the stable reference for
post-processing, plotting, validation scripts, and external integrations.

## Coordinate and unit conventions

Unless stated otherwise, image coordinates in CSV outputs are **crop-local**.
The origin `(y=0, x=0)` is the upper-left corner of the belt crop processed by
BeltMap, not necessarily the upper-left corner of the original full frame.

If `BELT_REGION` or `belt.region` is set to `top,left,height,width`, convert a
crop-local coordinate to a full-frame coordinate with:

```text
full_frame_y = top + y
full_frame_x = left + x
```

Bounding boxes use half-open NumPy/Python slicing convention:

```text
bbox_top <= y < bbox_bottom
bbox_left <= x < bbox_right
```

The vertical belt-coordinate convention is:

```text
belt_coordinate_y = image_y + phase_px
```

Positive `belt_velocity_px_per_frame` means the belt texture moves downward in
image coordinates. With the above convention, the belt phase decreases over
time. Angles are reported in radians, phases and displacements in pixels, and
velocities in pixels per frame.

## Output directory overview

A typical `beltmap-apply` run writes these files below the configured output
directory:

```text
outputs/
  belt_map.npy
  belt_map.png
  config_resolved.json
  detections.csv
  detections_per_frame.csv
  filtered_tracks.csv
  filtered_velocities.csv
  metadata.json
  phase_estimates.csv
  progress.jsonl
  progress_latest.json
  recurrent_artifact_counts.npy  # only when recurrent artifact map is built
  recurrent_artifact_counts.png  # only when recurrent artifact map is built
  recurrent_artifact_detections.csv # only when recurrent artifact filtering is enabled
  recurrent_artifact_map.npy     # only when recurrent artifact filtering is enabled
  recurrent_artifact_map.png     # only when recurrent artifact filtering is enabled
  raw_frame_000000.png
  raw_frame_000001.png
  raw_frame_000002.png
  residual_frame_000000.png
  residual_frame_000001.png
  residual_frame_000002.png
  residual_fixed_frame_000000.png
  residual_fixed_frame_000001.png
  residual_fixed_frame_000002.png
  static_background.npy   # only when static-background learning or reuse is enabled
  static_background.png   # only when static-background learning or reuse is enabled
  static_noise.npy        # only when static-noise learning or reuse is enabled
  static_noise.png        # only when static-noise learning or reuse is enabled
  tracks.csv
  track_scores.csv
  velocities.csv
```

Running `beltmap-validate --output-dir outputs` adds a Markdown validation
report and diagnostic plots:

```text
outputs/
  validation_report.md
  validation_summary.json
  phase_corrections.png
  phase_correction_timeseries.png
  registration_score.png
  detections_per_frame.png
  velocity_ratio_histogram.png
  track_length_histogram.png
  residual_histogram.png
  belt_map_coverage.png
  overlay_contact_sheet.png
  detections_overlay_sample_000000.png
  tracks_overlay_sample_000000.png
```

Running `beltmap-compare` on multiple output directories adds a comparison report
outside the individual run directories:

```text
comparison_report/
  comparison_report.md
  summary.csv
  detections_per_frame_comparison.png
  velocity_ratio_histogram_comparison.png
  detection_contact_sheet.png
  filtered_detection_contact_sheet.png
```

`config_resolved.json` is produced by the `beltmap-apply` CLI before the legacy
image driver is invoked. Running `scripts/apply_beltmap_to_images.py` directly
writes the driver outputs, but not necessarily the resolved CLI configuration
file.

## `belt_map.npy`

Purpose: stores the reconstructed particle-free conveyor-belt texture.

Format: NumPy `.npy` array.

Shape: `(belt_map_height_px, crop_width_px)`.

Dtype: currently written from a `float32` image array.

When detection-only reuse mode is enabled with `REUSE_BELT_MAP_PATH` or
`reuse.belt_map_path`, this file is a copy of the loaded map in the new output
directory. Detection, tracking, velocity, phase, progress, and metadata outputs
then correspond to the new run configuration.

Coordinates:

- axis 0 is belt-coordinate row `belt_coordinate_y`;
- axis 1 is crop-local image column `x`;
- rows wrap cyclically when a belt period is known or supplied.

Interpretation: row `phase_px + image_y` in this array is the clean belt value
expected at crop-local image row `image_y` for the corresponding frame.

## `static_background.npy`

Purpose: optional additive image-fixed background learned from belt-subtracted
residuals and subtracted during residual generation.

Format: NumPy `.npy` array.

Shape: `(crop_height_px, crop_width_px)`.

Units: grayscale residual intensity, the same units as `residual.raw`.

When `STATIC_BACKGROUND_SAMPLE_FRAMES` or
`static_background.sample_frames` is positive, the driver renders the clean
belt for sampled frames and computes belt-subtracted residuals:

```text
residual_t(y, x) = image_t(y, x) - belt_background_t(y, x)
```

It then estimates:

```text
static_background(y, x) = median_t(residual_t(y, x))
```

with optional particle masking. During detection the final raw residual is:

```text
raw = image - belt_background - static_background
```

When detection-only reuse mode is enabled with `REUSE_STATIC_BACKGROUND_PATH` or
`reuse.static_background_path`, this file is a copy of the loaded additive
static-background map in the new output directory.

## `belt_map.png`

Purpose: quick-look visualization of `belt_map.npy`.

Format: 8-bit grayscale PNG.

Scaling: display-only robust percentile scaling. Do not use this file for
quantitative analysis because the pixel values are contrast-normalized for
inspection.

Coordinates: same row and column layout as `belt_map.npy`.

## `static_noise.npy`

Purpose: optional image-fixed residual-noise floor used during particle
detection.

Format: NumPy `.npy` array.

Shape: `(crop_height_px, crop_width_px)`.

Units: grayscale residual intensity, the same units as `residual.raw`.

When `STATIC_NOISE_SAMPLE_FRAMES` or `static_noise.sample_frames` is positive,
the driver renders the clean belt for sampled frames, computes raw
belt-subtracted residuals, and estimates:

```text
static_noise(y, x) = 1.4826 * MAD_t(residual_t(y, x))
```

During detection, the normalized residual uses this map as a floor:

```text
normalized = residual / max(local_noise, static_noise)
```

When detection-only reuse mode is enabled with `REUSE_STATIC_NOISE_PATH` or
`reuse.static_noise_path`, this file is a copy of the loaded map in the new
output directory.

## `static_noise.png`

Purpose: quick-look visualization of `static_noise.npy`.

Format: 8-bit grayscale PNG.

Scaling: display-only robust percentile scaling. Do not use this file for
quantitative analysis.

## `recurrent_artifact_map.npy`

Purpose: optional belt-coordinate mask of detection artifacts that recur across
multiple belt revolutions.

Format: NumPy `.npy` boolean array.

Shape: `(belt_map_height_px, crop_width_px)`.

Coordinates: same belt-coordinate row and crop-local column layout as
`belt_map.npy`.

This file is written when recurrent artifact filtering is enabled, either by
building a map with `RECURRENT_ARTIFACT_MIN_REVOLUTIONS` /
`recurrent_artifact.min_revolutions` or by loading one with
`REUSE_RECURRENT_ARTIFACT_MAP_PATH` / `reuse.recurrent_artifact_map_path`.
Final `detections.csv`, `tracks.csv`, and velocity outputs are written after
rejecting detections whose belt-coordinate bounding boxes overlap this map too
strongly. With `recurrent_artifact.mode = "soft"`, overlapping detections can
survive if their peak residual is strong enough.

## `recurrent_artifact_detections.csv`

Purpose: per-detection recurrent-artifact diagnostic table written before final
rejection is applied.

Format: CSV with the same detection columns as `detections.csv`, plus
`recurrent_artifact_rejected`.

Rows include both kept and rejected first-pass detections. Use
`recurrent_artifact_overlap_fraction` to inspect how strongly a component
overlaps the belt-coordinate artifact map. In soft mode,
`recurrent_artifact_required_peak_signal` is the peak residual the component
had to exceed to survive the artifact filter. In hard mode this field is empty.
The final `detections.csv` contains only the kept rows.

## `recurrent_artifact_counts.npy`

Purpose: stores the distinct-revolution recurrence count used to build
`recurrent_artifact_map.npy`.

This file is written only when the recurrent artifact map is built in the
current run. It is not regenerated when a recurrent artifact map is loaded via
`reuse.recurrent_artifact_map_path`.

Format: NumPy `.npy` integer array.

Shape: `(belt_map_height_px, crop_width_px)`.

Interpretation: each pixel value is the number of distinct belt revolutions in
which at least one first-pass detection touched that belt-coordinate pixel.

## `recurrent_artifact_map.png` and `recurrent_artifact_counts.png`

Purpose: quick-look visualizations of the recurrent artifact mask and count
map.

Format: 8-bit grayscale PNG.

Scaling: display-only robust percentile scaling. Do not use these files for
quantitative analysis.

## `phase_estimates.csv`

Purpose: one phase estimate per processed frame.

Format: CSV with a header row.

Coordinates: phases are in belt-map pixels. Image paths are relative to the
input image directory.

Columns:

| Column | Unit | Meaning |
|---|---:|---|
| `frame_index` | frame | Zero-based processed-frame index after sorting, striding, and truncation. |
| `image` | path | Input image path relative to the configured image directory. |
| `phase_px` | px | Final corrected belt phase for the frame. |
| `phase_fraction` | 1 | `phase_px / belt_map_height_px` for the driver-created motion model. |
| `phase_rad` | rad | `phase_fraction * 2*pi`. |
| `predicted_phase_px` | px | Phase predicted by the signed constant-speed motion model before local correction. |
| `correction_px` | px | Registration offset added to the predicted phase. |
| `loss` | 1 | Trimmed mean-square registration loss; empty if no registration loss was computed. |
| `score` | 1 | Dimensionless registration score; empty if no registration score was computed. |
| `method` | text | Phase-estimation method, such as `motion_model`, `registration`, or `registration_smoothed`. |

Notes:

- `correction_px = phase_px - predicted_phase_px` modulo wrapping effects.
- `loss` is useful for diagnosing poor registration or weak belt texture.
- `score` is relative to the candidate-loss distribution and is not a calibrated
  probability.

## `detections.csv`

Purpose: one row per connected particle component detected in a processed frame.

Format: CSV with a header row.

Coordinates: centroids and bounding boxes are crop-local image coordinates.

Columns:

| Column | Unit | Meaning |
|---|---:|---|
| `frame_index` | frame | Zero-based processed-frame index after sorting, striding, and truncation. |
| `image` | path | Input image path relative to the configured image directory. |
| `label` | 1 | Connected-component label within the frame after thresholding. |
| `y` | px | Crop-local particle centroid row. |
| `x` | px | Crop-local particle centroid column. |
| `area_px` | px | Number of pixels in the connected component. |
| `bbox_top` | px | Top edge of the half-open crop-local bounding box. |
| `bbox_left` | px | Left edge of the half-open crop-local bounding box. |
| `bbox_bottom` | px | Bottom edge of the half-open crop-local bounding box. |
| `bbox_right` | px | Right edge of the half-open crop-local bounding box. |
| `mean_signal` | z | Mean normalized residual over the component. |
| `peak_signal` | z | Maximum normalized residual over the component. |
| `recurrent_artifact_overlap_fraction` | 1 | Fraction of the detection bbox covered by the recurrent artifact map; empty if recurrent artifact filtering was disabled. |
| `recurrent_artifact_required_peak_signal` | z | Soft-mode peak residual needed to survive artifact filtering; empty in hard mode or when recurrent artifact filtering was disabled. |

Notes:

- `detection.mode` controls whether the detector uses positive, negative, or
  absolute normalized residuals as its particle signal.
- `mean_signal` and `peak_signal` use the oriented detection signal of the
  current run. They are useful for ranking detections and for soft recurrent
  artifact filtering, but they are not calibrated physical intensities.
- Convert bounding boxes to full-frame coordinates by adding `top` to vertical
  fields and `left` to horizontal fields from the configured belt region.

## `detections_per_frame.csv`

Purpose: compact per-frame detection counts.

Format: CSV with a header row.

Columns:

| Column | Unit | Meaning |
|---|---:|---|
| `frame_index` | frame | Zero-based processed-frame index. |
| `n_detections` | count | Number of particle detections found in the frame. |

Use this file for quick time-series plots, sanity checks, and monitoring changes
in threshold or mask settings.

## `velocities.csv`

Purpose: one row per particle track with an estimated velocity and comparison to
belt motion.

Format: CSV with a header row.

Coordinates: velocities are estimated from crop-local detection centroids.

Columns:

| Column | Unit | Meaning |
|---|---:|---|
| `track_id` | 1 | Zero-based track identifier assigned by the PyRecEst-backed tracker. |
| `n_detections` | count | Number of detections associated with this track. |
| `frame_start` | frame | First frame index in the track. |
| `frame_end` | frame | Last frame index in the track. |
| `velocity_y_px_per_frame` | px/frame | Linear-slope estimate of vertical particle velocity in crop coordinates. |
| `velocity_x_px_per_frame` | px/frame | Linear-slope estimate of horizontal particle velocity in crop coordinates. |
| `speed_px_per_frame` | px/frame | Euclidean image-plane speed from the horizontal and vertical components. |
| `belt_velocity_y_px_per_frame` | px/frame | Signed vertical belt texture velocity used by the run. |
| `velocity_ratio_y` | 1 | `velocity_y_px_per_frame / belt_velocity_y_px_per_frame`. |
| `belt_minus_particle_velocity_y_px_per_frame` | px/frame | Difference between belt velocity and vertical particle velocity. |

Notes:

- A ratio between 0 and 1 means the particle moves in the belt direction but
  more slowly than the belt texture.
- Negative ratios indicate motion opposite to the signed belt direction.
- Tracks shorter than `MIN_TRACK_LENGTH` or `tracking.min_track_length` are not
  written as velocity rows.

## `tracks.csv`

Purpose: one row per detection assigned to a raw tracker trajectory.

Format: CSV with a header row.

Columns: `track_id`, `track_detection_index`, and the same crop-local detection
columns as `detections.csv`. Use this file when you need trajectory membership
for individual detections rather than one velocity row per track.

## `metadata.json`

Purpose: driver-level summary of the processed run.

Format: JSON object.

Important fields:

| Field | Unit | Meaning |
|---|---:|---|
| `n_images` | count | Number of selected images processed by the driver. |
| `discovered_frame_count` | count | Number of image files found before striding and truncation. |
| `frame_stride` | frames | Stride applied after natural sorting. |
| `first_image_shape` | px | `[height, width]` shape of the first original grayscale frame. |
| `belt_region` | px | Object with `top`, `left`, `height`, and `width` for the processed crop. |
| `belt_velocity_px_per_frame` | px/frame | Signed belt image velocity used by the run. |
| `belt_period_px_input` | px | User-supplied belt period, or `null` when no period was supplied. |
| `belt_map_height_px` | px | Height of the reconstructed belt map. |
| `reference_phase_px` | px | Phase assigned to the reference frame by the map builder. |
| `detection_threshold` | z | Threshold applied to the normalized residual image. |
| `detection_mode` | mode | Final detector polarity: `positive`, `negative`, or `absolute`. |
| `detection_low_threshold` | z | Optional lower hysteresis threshold for final detection, or `null` when disabled. |
| `min_area_px` | px | Minimum connected-component area for detection output. |
| `detection_max_area_px` | px | Optional maximum connected-component area, or `null` when disabled. |
| `detection_min_bbox_width_px` | px | Optional minimum component bounding-box width, or `null` when disabled. |
| `detection_min_bbox_height_px` | px | Optional minimum component bounding-box height, or `null` when disabled. |
| `detection_max_bbox_aspect_ratio` | ratio | Optional maximum component bounding-box aspect ratio, or `null` when disabled. |
| `detection_min_bbox_extent` | fraction | Optional minimum component extent, or `null` when disabled. |
| `map_mask_iterations` | count | Number of particle-mask refinement iterations used for map building. |
| `map_frame_median_offset_correction` | bool | Whether sampled frames were median-offset normalized during map reconstruction. |
| `map_particle_mask_threshold` | z | Threshold used for particle masking during map reconstruction. |
| `map_particle_mask_margin_px` | px | Bounding-box margin used when excluding particle pixels from the map. |
| `map_particle_mask_min_area_px` | px | Minimum component area used for map-building particle masks. |
| `static_background_map_used` | bool | Whether a learned or reused additive static residual-background map was subtracted during detection. |
| `reuse_static_background_path` | path | Source static-background map path, or an empty string when no map was reused. |
| `static_noise_map_used` | bool | Whether a learned or reused static residual-noise map was applied during detection. |
| `reuse_static_noise_path` | path | Source static-noise map path, or an empty string when no map was reused. |
| `reuse_recurrent_artifact_map_path` | path | Source recurrent artifact map path, or an empty string when no map was reused. |
| `static_noise_sample_frames` | frames | Number of frames requested for static residual-noise learning. |
| `static_noise_min_scale` | gray | Minimum static residual-noise scale used when writing the map. |
| `static_noise_mask_threshold` | z | Particle-mask threshold used while learning static noise, or `null` when disabled. |
| `static_noise_mask_margin_px` | px | Particle-box margin used while learning static noise. |
| `static_noise_mask_min_area_px` | px | Particle-mask minimum area used while learning static noise. |
| `static_background_sample_frames` | frames | Number of frames requested for static residual-background learning. |
| `static_background_mask_threshold` | z | Particle-mask threshold used while learning static background, or `null` when disabled. |
| `static_background_mask_margin_px` | px | Particle-box margin used while learning static background. |
| `static_background_mask_min_area_px` | px | Particle-mask minimum area used while learning static background. |
| `recurrent_artifact_filter_used` | bool | Whether recurrent artifact suppression was enabled. |
| `recurrent_artifact_min_revolutions` | revolutions | Distinct-revolution threshold used to build the artifact map. |
| `recurrent_artifact_margin_px` | px | Detection-box margin used while accumulating recurrent artifact counts. |
| `recurrent_artifact_max_overlap_fraction` | fraction | Per-detection overlap threshold used for rejection. |
| `recurrent_artifact_mode` | mode | Recurrent artifact rejection mode, either `hard` or `soft`. |
| `recurrent_artifact_soft_penalty_weight` | weight | Soft-mode peak-signal penalty weight. |
| `recurrent_artifact_source` | mode | `built`, `loaded`, or `none`. |
| `recurrent_artifact_revolutions` | count | Number of distinct revolution bins represented in the processed frame sequence. |
| `recurrent_artifact_pixels` | px | Number of belt-coordinate pixels marked as recurrent artifacts. |
| `n_recurrent_artifact_rejected` | count | Number of first-pass detections rejected by recurrent artifact suppression. |
| `n_phase_estimates` | count | Number of rows written to `phase_estimates.csv`. |
| `phase_estimation_mode` | mode | Configured phase source: `motion_model`, `registration`, or `smoothed_registration`. |
| `texture_phase_velocity_px_per_frame` | px/frame | Robust belt velocity estimated from the phase trajectory, when at least two phase rows are available. |
| `texture_phase_smoothed_velocity_px_per_frame` | px/frame | Median velocity from the phase/velocity smoother used as a diagnostic consistency check. |
| `n_detections` | count | Number of rows written to `detections.csv`. |
| `n_tracks` | count | Number of particle tracks created by the tracker. |
| `n_velocity_estimates` | count | Number of rows written to `velocities.csv`. |
| `n_filtered_velocity_estimates` | count | Number of rows written to `filtered_velocities.csv`. |
| `track_filter_min_length` | detections | Minimum detections per accepted filtered velocity row. |
| `track_filter_min_velocity_ratio_y` | ratio | Minimum accepted `velocity_ratio_y`. |
| `track_filter_max_velocity_ratio_y` | ratio | Maximum accepted `velocity_ratio_y`. |
| `track_filter_max_abs_x_velocity_px_per_frame` | px/frame | Optional lateral velocity gate, or `null` when disabled. |
| `auto_velocity_pair_shifts` | px/frame | List of adjacent-frame shifts used for automatic belt-velocity estimation. |
| `elapsed_s` | s | Total elapsed runtime reported by the driver. |

The metadata file is the recommended place to recover run-level configuration
and dimensions when post-processing CSV outputs.

## `config_resolved.json`

Purpose: records the effective values passed from `beltmap-apply` to the image
driver after resolving config-file values, environment variables, and CLI flags.

Format: JSON object written by the CLI before processing starts.

Top-level fields:

| Field | Meaning |
|---|---|
| `precedence` | Ordered list of configuration layers, currently `config`, `environment`, `cli`. |
| `options` | Object keyed by normalized option name. Each entry records the environment variable, value, and source. |
| `driver_environment` | Object of environment variables and values applied before invoking the driver. |

Example entry in `options`:

```json
{
  "belt_velocity_px_per_frame": {
    "env_var": "BELT_VELOCITY_PX_PER_FRAME",
    "value": "59.3",
    "source": "cli"
  }
}
```

Use this file to check what the driver actually received, especially when config
files, environment variables, and explicit CLI flags are mixed.

## `progress.jsonl`

Purpose: append-only progress log for long runs.

Format: newline-delimited JSON, one JSON object per event.

Stable common fields:

| Field | Unit | Meaning |
|---|---:|---|
| `timestamp` | ISO 8601 | UTC timestamp when the event was emitted. |
| `elapsed_s` | s | Elapsed seconds since driver startup. |
| `stage` | text | Coarse processing stage, such as `startup`, `images`, `velocity`, `belt_map`, `detect`, `track`, or `done`. |
| `message` | text | Human-readable progress message. |
| `rss_mb` | MB | Peak resident set size when available on the platform. |

Additional fields are stage-specific and may include paths, frame counts,
detection counts, map coverage statistics, frame rate, remaining-frame estimates,
or current input image names. Consumers should tolerate unknown extra fields.

## `progress_latest.json`

Purpose: latest progress event as a single JSON object.

Format: JSON object with the same schema as one line of `progress.jsonl`.

Use this file for dashboards or workflow monitors that only need the most recent
status without reading the whole JSONL log.

## `track_scores.csv`

Purpose: one row per raw velocity estimate, with track-level plausibility gates
used to create `filtered_velocities.csv`.

Format: CSV with a header row.

Columns:

| Column | Unit | Meaning |
|---|---:|---|
| `track_id` | 1 | Track identifier from the tracker and `velocities.csv`. |
| `n_detections` | count | Number of detections in the track. |
| `frame_start` | frame | First frame index in the track. |
| `frame_end` | frame | Last frame index in the track. |
| `velocity_y_px_per_frame` | px/frame | Vertical velocity copied from `velocities.csv`. |
| `velocity_x_px_per_frame` | px/frame | Horizontal velocity copied from `velocities.csv`. |
| `velocity_ratio_y` | 1 | Particle vertical velocity divided by belt vertical velocity. |
| `abs_x_velocity_px_per_frame` | px/frame | Absolute horizontal velocity magnitude. |
| `passes_min_track_length` | bool | Whether the track is long enough for the filter. |
| `passes_velocity_ratio` | bool | Whether `velocity_ratio_y` lies in the configured interval. |
| `passes_lateral_velocity` | bool | Whether the optional lateral-velocity gate passes. |
| `accepted` | bool | Whether all enabled gates pass. |
| `plausibility_score` | 1 | Smooth helper score in `[0, 1]`; use `accepted` for the hard filter. |

## `filtered_velocities.csv`

Purpose: subset of `velocities.csv` accepted by the track-level filter.

Format: same columns and units as `velocities.csv`.

Default filter: at least five detections and
`0 <= velocity_ratio_y <= 1.1`. The raw `velocities.csv` file is not modified.
Existing runs can be post-processed with `beltmap-filter-tracks`.

## `filtered_tracks.csv`

Purpose: subset of `tracks.csv` whose `track_id` is accepted by
`track_scores.csv`.

Format: same columns and units as `tracks.csv`.

This is useful for visual overlays or downstream analysis that should ignore
short fragments and physically implausible velocity rows.

## `raw_frame_*.png`

Purpose: visual debug previews of the processed raw crop for the same sampled
frames as `residual_frame_*.png`.

Format: 8-bit grayscale PNG.

Shape: same height and width as the processed crop, not necessarily the original
full frame.

Scaling: fixed 0..255 grayscale display range.

## `residual_frame_*.png` and `residual_fixed_frame_*.png`

Purpose: visual debug previews of selected normalized residual frames.

Format: 8-bit grayscale PNG.

Shape: same height and width as the processed crop, not necessarily the original
full frame.

Scaling: `residual_frame_*.png` uses display-only robust percentile scaling of
each normalized residual frame. `residual_fixed_frame_*.png` uses a fixed
normalized residual display range of -8..8 so frames can be compared visually.
Do not use these PNGs for quantitative thresholding or measurement; use the
driver CSV outputs or rerun residual generation in Python when numerical
residual values are needed.

When written:

- the first `DEBUG_RESIDUAL_PREVIEW_FRAMES` frames are saved;
- if `DEBUG_RESIDUAL_PREVIEW_INTERVAL_FRAMES > 0`, additional previews are saved
  at that interval.

## Validation report and diagnostic plots

Purpose: quick quality-control artifacts generated by
`beltmap-validate --output-dir outputs` after a driver run.

Files:

| File | Meaning |
|---|---|
| `validation_report.md` | Markdown summary of run metadata, missing expected files, phase-registration statistics, detection counts, velocity-ratio statistics, track-length statistics, and belt-map progress. |
| `validation_summary.json` | Machine-readable validation metrics for dashboards, workflow checks, and run comparisons. |
| `phase_corrections.png` | Histogram of finite `correction_px` values from `phase_estimates.csv`. |
| `phase_correction_timeseries.png` | Time series of `correction_px` versus `frame_index`; useful for spotting drift, periodic failures, or search-boundary clipping. |
| `registration_score.png` | Time series of finite registration `score` values from `phase_estimates.csv`. |
| `detections_per_frame.png` | Time series of `n_detections` values from `detections_per_frame.csv`. |
| `velocity_ratio_histogram.png` | Histogram of finite `velocity_ratio_y` values from `velocities.csv`. |
| `track_length_histogram.png` | Histogram of `n_detections` values from `velocities.csv`, useful for spotting configurations that produce only tiny tracks. |
| `residual_histogram.png` | Histogram over saved residual preview PNG intensities. |
| `belt_map_coverage.png` | Nominal phase-trajectory coverage proxy for the belt map. |
| `overlay_contact_sheet.png` | Side-by-side sheet pairing detection and track overlay samples by frame. |
| `detections_overlay_sample_*.png` | Residual preview samples with detection boxes and centroids. |
| `tracks_overlay_sample_*.png` | Residual preview samples with reconstructed track polylines. |

These PNGs are diagnostic plots, not measurement data. Use the corresponding
CSV files for quantitative post-processing.

## Comparison report

Purpose: compare two or more BeltMap output directories, especially
detection-only threshold sweeps that reuse the same reconstructed belt map.

Example:

```bash
beltmap-compare \
  --run T4.0=outputs/T4p0 \
  --run T3.5=outputs/T3p5 \
  --frames 0,248,496 \
  --truth-path labels/brick_validation_boxes.csv \
  --report-dir outputs/threshold_comparison
```

`--truth-path` is optional. When supplied, it may point to a CSV file with
`frame_index`, `bbox_top`, `bbox_left`, `bbox_bottom`, and `bbox_right` columns,
or to a JSON file containing a `particles`, `annotations`, `labels`, or
`detections` list with equivalent `top`, `left`, `bottom`, and `right` fields.
Coordinates are crop-local and use the same half-open box convention as
`detections.csv`. A sparse validation set can include labeled empty frames with
CSV rows that contain only `frame_index`, or with a JSON `scored_frames` list.
Predictions outside the scored frame set are ignored by the labeled metrics.

Files:

| File | Meaning |
|---|---|
| `comparison_report.md` | Markdown summary table, optional labeled detection target, visual contact sheet, and interpretation checklist. |
| `summary.csv` | Machine-readable proxy metrics and, when `--truth-path` is supplied, labeled detection precision/recall/F1 for each compared run. |
| `detections_per_frame_comparison.png` | Detection-count time series for all runs. |
| `velocity_ratio_histogram_comparison.png` | Overlaid velocity-ratio histograms. |
| `detection_contact_sheet.png` | Side-by-side residual preview images with detection boxes. |
| `filtered_detection_contact_sheet.png` | Side-by-side residual previews with detections from accepted filtered tracks. |

The comparison command tolerates partial runs. If `metadata.json` is absent, it
falls back to currently available CSV rows and preview images.

## Benchmark sweep outputs

Purpose: record operating-point curves for synthetic runs generated by
`beltmap-sweep --benchmark-truth-path ...`, especially detection-threshold
sweeps.

Files:

| File | Meaning |
|---|---|
| `sweep_manifest.json` | Generated run configs, output directories, and parameter overrides. |
| `sweep_metrics.csv` | One row per run with detection precision/recall/F1, false positives per frame, event F1, filtered event F1, PyRecEst track-length statistics, single-frame track counts, track fragmentation, birth false-positive rate, missed-event rate, velocity mean absolute error/bias/variance, phase RMSE, and map RMSE. |
| `sweep_metrics.json` | JSON form of the same benchmark summary rows. |
| `sweep_report.md` | Compact Markdown table for quick inspection before plotting curves. |

The sweep summary is intended for precision-recall curves, F1-vs-threshold,
false-positives-vs-recall, track-length-vs-threshold,
track-fragmentation-vs-threshold, and velocity-bias-vs-threshold plots.
