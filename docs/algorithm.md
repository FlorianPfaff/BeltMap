# Algorithm note

BeltMap treats an image sequence of a moving conveyor belt as a fixed belt texture observed through a fixed rectangular image crop. The central quantity is the vertical belt phase: the belt-map row that aligns with image row 0 in a given frame.

## Coordinate convention

Inside the belt crop, image rows and belt-map rows are related by

```text
belt_coordinate_y = image_y + phase_px
```

Image coordinates are assumed to use the standard image convention: `y` increases downward. The parameter `image_velocity_px_per_frame` is the signed vertical texture velocity in image coordinates. A positive velocity means the belt texture moves downward from one frame to the next. With the above coordinate convention, the belt phase therefore decreases over time:

```text
phase_px(t) = reference_phase_px
              - image_velocity_px_per_frame * (t - reference_frame)
```

If a belt period is known, phases and rendered rows are wrapped modulo that period. Otherwise the sequence driver builds a finite, non-periodic map whose height covers the sampled phase range.

## Clean belt rendering

The belt map is a two-dimensional image indexed by `(belt_coordinate_y, x)`. Rendering the expected particle-free belt crop for a frame consists of:

1. predicting or estimating `phase_px` for the frame;
2. evaluating rows `image_y + phase_px` for all crop rows;
3. interpolating fractional belt-map rows linearly along the vertical belt coordinate;
4. placing the rendered crop into the requested full-frame output, while marking pixels outside the belt crop invalid.

The validity mask is important because residuals outside the belt crop should not influence subtraction, noise estimation, or particle detection.

## Phase estimation and registration

The default phase estimate comes from the constant-speed motion model above. If an observed frame and a belt map are available, BeltMap can refine the predicted phase by one-dimensional local registration:

1. crop the observed image to the belt region;
2. test candidate phase offsets around the predicted phase;
3. render the belt map for each candidate phase;
4. high-pass both observed and rendered crops by subtracting a NumPy box blur;
5. normalize both prepared crops by their standard deviation;
6. compute a trimmed mean-squared residual over valid pixels;
7. select the candidate offset with the smallest loss.

The search is deliberately one-dimensional. BeltMap assumes that camera geometry, crop location, and horizontal alignment are fixed, while the belt advances vertically through the crop.

## Belt-map reconstruction

The image-sequence driver reconstructs a clean belt map by accumulating sampled frames in belt coordinates:

1. choose a belt velocity, either supplied explicitly or estimated from vertical frame-to-frame correlation shifts;
2. choose the map geometry from the supplied belt period or from the phase range of the selected sequence;
3. sample frames from the image sequence;
4. map every crop pixel to its predicted belt-coordinate row;
5. linearly splat fractional belt-coordinate rows into the two neighboring belt-map rows;
6. average all weighted observations that land in the same belt-map pixel;
7. interpolate belt-map pixels that were not directly observed.

When `MAP_MASK_ITERATIONS > 0`, this accumulation is refined by particle masking. BeltMap first builds a provisional map, renders each sampled frame, detects bright residual components, expands their bounding boxes, and excludes those pixels from the next map accumulation. This treats particle-covered observations as missing data rather than as belt texture.

## Residual normalization and particle detection

Given an observed frame and the expected clean belt background, BeltMap computes the raw residual

```text
raw_residual = image - expected_background
```

and the normalized residual

```text
normalized_residual = raw_residual / local_noise
```

The local noise scale is estimated robustly from valid residual pixels. The current implementation uses a global median center, a median-absolute-deviation scale estimate, optional clipping, and a masked local box variance. Invalid pixels are filled with `NaN` and excluded from later processing.

Optionally, the image-sequence driver can also learn a static residual-noise map
from sampled belt-subtracted residuals:

```text
static_noise(y, x) = 1.4826 * MAD_t(raw_residual_t(y, x))
```

When enabled, final detection uses the larger of the frame-local scale and this
image-fixed scale:

```text
normalized_residual = raw_residual / max(local_noise, static_noise)
```

This is a normalization floor, not a background subtraction term. It is intended
to down-weight fixed residual structures such as illumination or detector
artifacts without changing the expected clean belt render.

The default detector is intentionally simple and interpretable for bright particles on a darker belt:

```text
particle_mask = normalized_residual > threshold
```

Invalid pixels, non-finite pixels, and pixels excluded by a user mask are always returned as non-particle pixels.

## Recurrent artifact suppression

When enabled, recurrent artifact suppression runs after the first-pass detector
and before tracking. It maps each detection bounding box from image coordinates
to belt coordinates with

```text
belt_y = image_y + phase_px
```

and counts how many distinct belt revolutions touched each belt-coordinate
pixel. Pixels reached in at least `recurrent_artifact.min_revolutions` distinct
revolutions form a recurrent artifact map. Detections whose belt-coordinate
bounding boxes overlap that map above
`recurrent_artifact.max_overlap_fraction` are removed before writing final
detection, tracking, and velocity outputs.

The default `hard` mode removes those detections by overlap alone. The optional
`soft` mode keeps strong peaks and rejects only weak recurring detections, using
`detection.threshold * (1 + recurrent_artifact.soft_penalty_weight * overlap)`
as the required peak residual.

When recurrent artifact filtering is active, detections carry the diagnostic
columns `recurrent_artifact_overlap_fraction` and
`recurrent_artifact_required_peak_signal`. The driver also writes
`recurrent_artifact_detections.csv` with both kept and rejected first-pass
detections so threshold changes can be inspected without treating the artifact
map as a black-box deletion step.

This targets belt-fixed scratches and map ghosts. It should not remove ordinary
loose particles unless they repeatedly appear at the same belt-coordinate
location across separate revolutions.

## Connected components, tracks, and velocities

Particle detections are extracted as connected components of the particle mask. Components can use 4- or 8-neighborhood connectivity, are filtered by area, and receive either unweighted or residual-weighted centroids.

Frame-to-frame association uses PyRecEst global nearest-neighbor multitarget tracking with a maximum match distance. New tracks start from the configured velocity prior, and established tracks are predicted from their recent measured velocity to reduce fragmentation when particles drift away from the nominal belt-speed prior. PyRecEst also receives conservative pairwise area/signal consistency costs from recent track medians so ambiguous associations prefer detections that still look like the same particle without overreacting to one noisy component. Per-track particle velocity is then estimated by a linear slope of centroid position versus frame index. Belt-relative output includes

```text
velocity_ratio_y = particle_velocity_y / belt_velocity_y
```

For particles moving in the belt direction but more slowly than the belt, this ratio lies between 0 and 1.

## Assumptions and limitations

BeltMap is designed for a fixed camera and a fixed conveyor-belt crop. In its current form, it assumes:

- the belt motion is well approximated by a signed constant vertical image velocity;
- the belt region has already been cropped or specified accurately;
- horizontal alignment and perspective distortion are negligible after cropping;
- the belt texture has enough vertical structure for phase registration;
- the configured detector polarity matches the residual signature of particles;
- non-belt background is excluded via the belt-region validity mask.

The method is not a general optical-flow tracker or a full geometric camera calibration pipeline. Its purpose is to exploit repeated belt texture to form a clean background model, subtract that model, and obtain interpretable particle detections and belt-relative velocities.
