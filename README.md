# BeltMap

Tools for reconstructing conveyor-belt background maps and using them to improve particle localization.

The first implemented piece is belt phase estimation:

- predict the belt phase from a signed constant-speed model
- render the expected clean belt crop for a frame
- refine the predicted phase by robustly registering the observed frame against the belt map

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

Datasets are intentionally not stored in this repository.
