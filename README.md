# BeltMap

[![CI](https://github.com/IPS-Stuttgart/BeltMap/actions/workflows/ci.yml/badge.svg)](https://github.com/IPS-Stuttgart/BeltMap/actions/workflows/ci.yml)
[![Smoke test BeltMap image driver](https://github.com/IPS-Stuttgart/BeltMap/actions/workflows/smoke-beltmap-driver.yml/badge.svg)](https://github.com/IPS-Stuttgart/BeltMap/actions/workflows/smoke-beltmap-driver.yml)

Tools for reconstructing conveyor-belt background maps and using them to improve particle localization.

The first implemented piece is belt phase estimation:

- predict the belt phase from a signed constant-speed model
- render the expected clean belt crop for a frame
- refine the predicted phase by robustly registering the observed frame against the belt map

The image-sequence driver writes `phase_estimates.csv` with one row per
processed image. It reports the predicted phase, the registration correction,
the corrected phase in belt-map pixels, the normalized phase fraction, and the
equivalent phase angle in radians.

When building `belt_map.npy`, the driver can iteratively mask bright particles
from the map accumulation. It first builds a provisional belt map, renders each
sampled frame at its belt phase, detects bright residual components, expands
their bounding boxes by `MAP_PARTICLE_MASK_MARGIN_PX`, and rebuilds the map by
averaging only unmasked pixels. With repeated belt revolutions, particle-covered
observations are therefore treated as missing data instead of contaminating the
clean belt estimate.

The public clean-belt renderer is `render_expected_clean_belt`. It can render a
full-frame expected background with a validity mask so subtraction ignores the
camera background outside the belt crop.

Residual images for particle localization are generated with
`generate_residual_image` or the convenience wrapper
`render_clean_belt_residual`:

```python
residual = render_clean_belt_residual(
    image=frame,
    belt_map=belt_map,
    frame_index=t,
    motion_model=model,
    belt_region=(top, left, height, width),
)
z_image = residual.normalized
```

The normalized image is
`(image - expected_background) / local_noise`. The local noise is estimated
robustly from the residual image, and invalid non-belt pixels are masked.

Bright brick particles on a dark belt can then be detected by thresholding the
normalized residual:

```python
particle_mask = detect_particles_from_residual(residual, threshold=5.0)
```

Particle velocities can be extracted from a sequence of particle masks and
compared to the signed belt image velocity:

```python
velocities = extract_particle_velocities_vs_belt(
    particle_masks,
    belt_image_velocity_px_per_frame=model.image_velocity_px_per_frame,
)
for velocity in velocities:
    print(velocity.velocity_y_px_per_frame, velocity.velocity_ratio_y)
```

`velocity_ratio_y` is `particle_velocity_y / belt_velocity_y`. Particles moving
in the belt direction but slower than the belt therefore have ratios between
0 and 1.

Datasets are intentionally not stored in this repository.
