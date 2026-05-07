# Validation and benchmark report

This page defines a compact validation report for BeltMap runs. The goal is to
make an image-sequence run auditable without requiring a separate benchmark
framework: every diagnostic below can be derived from the standard driver
outputs in `outputs/`.

Use this document for two kinds of checks:

1. **Synthetic sanity checks** with known motion and known particle placement.
2. **Real-data diagnostics** where ground truth is unavailable but failures
   should still be visible from phase, residual, detection, and velocity traces.

## Standard inputs

A complete validation report should archive or link the following files from one
BeltMap run:

| File | Purpose |
| --- | --- |
| `config_resolved.json` | Exact resolved config passed to the driver. |
| `metadata.json` | Run summary, selected frame count, velocity, and output counts. |
| `progress.jsonl` | Stage-wise progress, timing, memory, and map-building summaries. |
| `phase_estimates.csv` | Per-frame phase prediction, correction, loss, score, and method. |
| `detections.csv` | Per-particle connected-component detections. |
| `detections_per_frame.csv` | Detection count time series. |
| `velocities.csv` | Per-track velocities and belt-relative velocity ratios. |
| `belt_map.png` | Visual preview of the reconstructed clean belt texture. |
| `residual_frame_*.png` | Optional residual previews for qualitative inspection. |

## Minimal report summary

Each run should start with a short summary table. Populate it from
`metadata.json` and `config_resolved.json`.

| Quantity | Source | Interpretation |
| --- | --- | --- |
| selected frames | `metadata.json:n_images` | Number of frames processed by the driver. |
| frame stride | `metadata.json:frame_stride` | Temporal subsampling used for this run. |
| belt region | `metadata.json:belt_region` | Crop used as the belt observation window. |
| belt velocity | `metadata.json:belt_velocity_px_per_frame` | Signed image velocity in pixels per frame. |
| belt period / map height | `metadata.json:belt_map_height_px` | Periodic or inferred belt-map height. |
| phase estimates | `metadata.json:n_phase_estimates` | Should match selected frames. |
| detections | `metadata.json:n_detections` | Total detected particle components. |
| tracks | `metadata.json:n_tracks` | Number of associated particle tracks. |
| velocity estimates | `metadata.json:n_velocity_estimates` | Tracks long enough for velocity output. |

## Diagnostic 1: phase correction histogram

Use `phase_estimates.csv:correction_px` to show how much local registration
changed the constant-speed prediction.

Expected behavior:

- Corrections should be centered near zero when velocity and reference phase are
  well calibrated.
- Corrections should stay comfortably inside the configured registration search
  radius.
- A pile-up near the search boundary indicates a wrong velocity, wrong belt
  period, wrong crop, or insufficient search radius.

Suggested plot:

```text
x-axis: correction_px
 y-axis: number of frames
```

Report at least these numbers:

```text
median(correction_px)
median(abs(correction_px))
max(abs(correction_px))
fraction(abs(correction_px) > 0.8 * registration_search_radius_px)
```

## Diagnostic 2: registration loss and score over time

Use `phase_estimates.csv:loss` and `phase_estimates.csv:score`.

Expected behavior:

- Loss should not show long monotonic drift.
- Score should not collapse for long contiguous frame ranges.
- Isolated bad frames are acceptable if residual previews confirm occlusions,
  lighting changes, or unusually dense particles.

Suggested plots:

```text
x-axis: frame_index       y-axis: loss
x-axis: frame_index       y-axis: score
```

Recommended checks:

```text
list frames with score below a chosen low-score threshold
compare low-score frames against residual_frame_*.png previews
check whether low-score regions coincide with detection bursts
```

## Diagnostic 3: belt-map coverage and map-building stability

The driver logs map-building information in `progress.jsonl`. Filter entries
with stage `belt_map` and inspect fields such as observed pixels, total pixels,
masked pixels, contributed pixels, sampled frames, crop size, and map height.

Expected behavior:

- Observed belt-map pixels should be a substantial fraction of total pixels.
- Masked pixels should be nonzero when particles are present and mask iterations
  are enabled.
- Masked pixels should not dominate the accumulation; if they do, the threshold
  or margin may be too aggressive.
- `belt_map.png` should show continuous belt texture rather than particle ghosts
  or horizontal interpolation bands.

Report at least:

```text
observed_pixels / total_pixels for the final map-building pass
masked_pixels for each particle-masked iteration
sampled_frames used to build the map
visual inspection result for belt_map.png
```

## Diagnostic 4: detection count over time

Use `detections_per_frame.csv`.

Expected behavior:

- Detection counts should be stable for stationary process conditions.
- Sudden spikes often indicate residual normalization failure, lighting changes,
  belt-map phase errors, or an overly low detection threshold.
- Long zero-detection stretches may be valid for empty-belt data, but should be
  suspicious for seeded synthetic tests or known particle flows.

Suggested plot:

```text
x-axis: frame_index
 y-axis: n_detections
```

Report:

```text
mean detections per frame
median detections per frame
max detections per frame
number of zero-detection frames
frames with unusually high counts
```

## Diagnostic 5: velocity-ratio histogram

Use `velocities.csv:velocity_ratio_y` and
`velocities.csv:belt_minus_particle_velocity_y_px_per_frame`.

Expected behavior for bright particles moving in the belt direction but slower
than the belt:

```text
0 < velocity_ratio_y < 1
belt_minus_particle_velocity_y_px_per_frame > 0
```

Values near 1 indicate particles moving with the belt texture. Values near 0
indicate nearly stationary particles in image coordinates. Negative values or
values above 1 may be valid in special cases, but should be explained.

Suggested plot:

```text
x-axis: velocity_ratio_y
 y-axis: number of tracks
```

Report:

```text
number of velocity estimates
median velocity_ratio_y
interquartile range of velocity_ratio_y
number of ratios outside the physically expected range for the experiment
```

## Diagnostic 6: residual preview inspection

Use `residual_frame_*.png` previews when enabled. At minimum, inspect the first
few frames and any frames flagged by low registration score or unusually high
detection count.

Expected behavior:

- Belt texture should be largely suppressed.
- Bright particles should remain visible as compact positive residuals.
- Non-belt background should be masked out or not influence detections.
- Strong belt-texture ghosts usually indicate a phase, period, crop, or velocity
  problem.

Include representative preview images in reports when possible:

```text
one typical good frame
one frame with the largest registration correction
one frame with the lowest registration score
one frame with the highest detection count
```

## Synthetic-sequence expectations

Synthetic sequences are useful because the correct direction, speed, and rough
particle count are known before running the driver. For the synthetic smoke-test
sequence used by the CI workflow, the expected conditions are:

| Quantity | Expected value or behavior |
| --- | --- |
| frame count | 12 selected frames |
| belt image velocity | `+2 px/frame` |
| belt period | `64 px` |
| particle motion | one bright particle moving downward by about `1 px/frame` |
| detections | at least one detection in the processed sequence |
| dominant velocity ratio | near `0.5` for a correctly tracked particle |

Use tolerances rather than exact equality for tracking-derived quantities. The
belt map is reconstructed from finite, particle-contaminated observations, and
registration/refinement choices can slightly perturb connected components and
track association.

Recommended synthetic acceptance checks:

```text
metadata.n_images == 12
metadata.belt_velocity_px_per_frame == 2.0
metadata.n_detections > 0
metadata.n_phase_estimates == metadata.n_images
max(abs(correction_px)) <= registration_search_radius_px
at least one velocity_ratio_y is in [0.25, 0.75]
```

## Real-data acceptance checklist

A real-data validation report is acceptable when all of the following are true:

- `config_resolved.json` and `metadata.json` are archived with the run.
- `phase_estimates.csv` has one row per selected frame.
- Phase corrections do not systematically hit the registration search boundary.
- Registration scores do not collapse for long unexplained intervals.
- Belt-map coverage in `progress.jsonl` is adequate for the configured map
  geometry.
- `belt_map.png` does not show obvious particle contamination or interpolation
  artifacts.
- Detection counts are plausible for the observed experiment.
- Velocity-ratio outliers are either rare or manually explained.
- Residual previews confirm that detections correspond to particles rather than
  belt texture or non-belt background.

## Suggested report template

Use the following structure for a compact report in a paper artifact, release
artifact, or internal experiment log:

```text
Run name:
Commit SHA:
Dataset / sequence:
Resolved config: path or artifact link
Outputs: path or artifact link

Summary:
  selected frames:
  belt region:
  belt velocity:
  belt period / map height:
  total detections:
  tracks:
  velocity estimates:

Phase registration:
  median correction:
  max absolute correction:
  low-score frames:
  boundary-hit fraction:

Map reconstruction:
  sampled frames:
  observed / total belt-map pixels:
  masked pixels by iteration:
  belt_map.png inspection:

Detection and tracking:
  mean detections per frame:
  zero-detection frames:
  median velocity ratio:
  velocity-ratio outliers:

Residual preview inspection:
  typical frame:
  lowest-score frame:
  highest-detection frame:

Conclusion:
  pass / pass with caveats / fail
  caveats:
```
