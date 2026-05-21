# Result-improvement guide

This note collects practical changes that usually improve BeltMap output quality
on real conveyor-belt sequences such as the Brick 10 g/s example. It is intended
as an experiment plan, not as a claim that one configuration is optimal for all
belts, cameras, and particle sizes.

## Recommended first Brick run

Start from `examples/brick_10gpers/beltmap.toml`. The committed default now uses
a stronger map build, phase-coverage map sampling, Huber map aggregation,
hysteresis detection, static-background and static-noise learning,
offline phase smoothing, dense-component splitting, robust velocity fitting, soft
recurrent-artifact suppression, conservative shape gates, optional fractional
map accumulation, and global tracking assignment.

Run:

```bash
beltmap-apply --config examples/brick_10gpers/beltmap.toml --print-config
beltmap-validate --output-dir outputs
```

The Brick GitHub Actions workflow should use the same values as this checked-in
config by default: 500 sampled map frames, adaptive phase-coverage sampling,
phase-feedback refinement, additive static-background learning, soft recurrent
artifact filtering, shape gates, and global tracking. Use workflow inputs for
ablation sweeps rather than treating the weaker historical defaults as the main
result path.

Inspect these outputs first:

- `validation_report.md`
- `overlay_contact_sheet.png`
- `detections_overlay_sample_*.png`
- `tracks_overlay_sample_*.png`
- `phase_estimates.csv`
- `photometric_fits.csv` when `[photometric].enabled = true`
- `track_scores.csv`
- `filtered_velocities.csv`

## Minimal ablation matrix

Use one high-quality belt map first, then reuse it for detection-only sweeps.
This avoids attributing a detection-threshold effect to a changed map.

If sparse real labels are available, rank the matrix by labeled precision/recall/F1 first; use proxy metrics only as a fallback.

### 1. Baseline

```toml
[detection]
mode = "positive"
threshold = 5.0
low_threshold = 0.0

[map]
sample_frames = 120
mask_iterations = 1
fractional_splat = false
particle_mask_mode = "positive"

[static_background]
sample_frames = 0

[static_noise]
sample_frames = 0

[recurrent_artifact]
min_revolutions = 0

[tracking]
assignment_method = "greedy"
```

### 2. Strong map

```toml
[map]
sample_frames = 500
sampling_strategy = "adaptive_phase_coverage"
sample_strategy = "phase_coverage"
adaptive_candidate_frames = 3000
reconstruction_trim_fraction = 0.05
mask_iterations = 2
fractional_splat = true
particle_mask_mode = "hysteresis_abs"
particle_mask_threshold = 4.0
particle_mask_grow_threshold = 1.5
particle_mask_dilation_px = 8
particle_mask_margin_px = 16
particle_mask_min_area_px = 8

[phase_refinement]
iterations = 1
max_abs_correction_px = 4
smoothing_window_frames = 25
```

For contaminated maps, compare `aggregation = "mean"` to `aggregation = "huber"` with `robust_iterations = 1`.

### 3. Strong map plus image-fixed residual model

Enable per-frame photometric correction if validation overlays show broad
positive/negative residual structure after the belt texture is geometrically
aligned:

```toml
[photometric]
enabled = true
trim_fraction = 0.05
```

```toml
[static_background]
sample_frames = 500
mask_threshold = 4.0
mask_margin_px = 16
mask_min_area_px = 4
```

Keep this map reusable during detection sweeps via `reuse.static_noise_path`;
otherwise threshold changes can be confounded by a different noise floor.

Add static noise only if validation overlays still show camera-fixed residual
variation after static-background subtraction:

```toml
[static_noise]
sample_frames = 500
min_scale = 0.5
mask_threshold = 4.0
mask_margin_px = 16
mask_min_area_px = 4
```

### 4. Strong map plus artifact and shape gates

```toml
[detection]
mode = "positive"
threshold = 4.5
low_threshold = 2.0
min_area_px = 4
min_bbox_width_px = 3
min_bbox_height_px = 3
max_bbox_aspect_ratio = 4.0
min_bbox_extent = 0.15

[recurrent_artifact]
min_revolutions = 3
margin_px = 4
max_overlap_fraction = 0.3
mode = "soft"
soft_penalty_weight = 1.0
```

### 5. Offline phase smoothing, merged-component splitting, and robust velocities

```toml
[phase_smoothing]
window_frames = 25
min_score = 0.05
max_abs_correction_px = 4

[detection]
split_merged_components = true
split_min_projection_gap_px = 1

[tracking]
velocity_fit_method = "theil_sen"
```

## What still needs labels

The synthetic benchmark checks phase, map reconstruction, detections, and
velocities when latent truth is known. Real-data quality should be selected by a
small manual validation set because true Brick detections are not encoded in the
image sequence. A practical minimum is 20 to 50 annotated frames containing
particle boxes, labeled empty frames, belt scratches, recurrent artifact regions,
edge particles, dense-particle frames, and illumination-change examples. Use that
set as the primary criterion for the final threshold, shape gates, recurrent
artifact settings, static-background/static-noise settings, photometric
correction, and track filters.

Keep the labeled set separate from this repository unless redistribution rights
are clear.

The Brick comparison workflow now supports an optional `truth_path` input. When it points to a checked-out label JSON, each variant writes `real_label_metrics.json` and the summary ranks labeled metrics alongside the existing proxy metrics.
