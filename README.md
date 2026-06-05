# BeltMap

[![CI](https://github.com/IPS-Stuttgart/BeltMap/actions/workflows/ci.yml/badge.svg)](https://github.com/IPS-Stuttgart/BeltMap/actions/workflows/ci.yml)
[![Smoke test BeltMap image driver](https://github.com/IPS-Stuttgart/BeltMap/actions/workflows/smoke-beltmap-driver.yml/badge.svg)](https://github.com/IPS-Stuttgart/BeltMap/actions/workflows/smoke-beltmap-driver.yml)

Tools for reconstructing conveyor-belt background maps and using them to improve particle localization.

For a concise description of the implemented coordinate convention, phase model,
registration step, belt-map reconstruction, residual normalization, particle
detection, and tracking assumptions, see the [algorithm note](docs/algorithm.md).
For a file-by-file and column-by-column description of driver outputs, see the
[output schema reference](docs/outputs.md).
For all runtime configuration keys, environment variables, CLI flags, defaults,
and units, see the [configuration reference](docs/configuration.md).
For a practical sequence of result-improvement experiments for the Brick 10 g/s
case and similar real conveyor data, see the
[result-improvement guide](docs/result_improvements.md).

Try the self-contained synthetic sequence example without downloading external
data:

```bash
python -m pip install -e ".[test]"
bash examples/synthetic_sequence/run.sh
```

The example generates a small moving-belt image sequence, runs `beltmap-apply`,
writes diagnostic outputs with `beltmap-validate`, computes synthetic
ground-truth metrics with `beltmap-benchmark`, and validates the expected
outputs. See [`examples/synthetic_sequence`](examples/synthetic_sequence) for
details.

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

## Command-line use

Install the package from a checkout and run the image-sequence driver with the
`beltmap-apply` command:

```bash
python -m pip install -e ".[test]"
beltmap-apply \
  --image-dir data/images \
  --output-dir outputs \
  --belt-region 0,220,1330,1800 \
  --belt-velocity-px-per-frame 59.3 \
  --belt-period-px 14723
```

When processing every Nth input frame with `--frame-stride`, manually supplied
belt velocities must declare their frame unit. Use
`--belt-velocity-frame-unit source_frame` when the velocity was measured between
adjacent original input frames; the driver multiplies it by the stride before
phase prediction. Use `selected_frame` when the velocity is already expressed per
processed/selected frame. Automatic velocity estimation uses selected-frame pairs.

The CLI keeps compatibility with the original environment variables. Runtime
configuration is resolved in this order, with later sources taking precedence:

1. values from `--config`
2. existing environment variables such as `BELT_REGION` or `MAX_FRAMES`
3. explicit CLI flags

The resolved values passed to the driver are written to
`outputs/config_resolved.json` before processing starts. The driver still writes
its run metadata to `outputs/metadata.json`.

Generate a TOML template with:

```bash
beltmap-apply --write-config-template beltmap.toml
```

A minimal config file can be flat or sectioned. For example:

```toml
[paths]
image_dir = "data/images"
output_dir = "outputs"

[belt]
region = [0, 220, 1330, 1800]
velocity_px_per_frame = 59.3
# If frames.stride > 1 and this velocity is measured between original input
# frames, also set: velocity_frame_unit = "source_frame"
# Use "selected_frame" if the value already refers to processed frames.
period_px = 14723

[detection]
threshold = 5.0
min_area_px = 4

[map]
sample_frames = 120
mask_iterations = 1

[progress]
interval_frames = 25
partial_output_interval_frames = 250
```

Run it with:

```bash
beltmap-apply --config beltmap.toml
```

Use `beltmap-apply --dry-run --config beltmap.toml` to print the resolved
settings without running the image driver.

After a run, create a Markdown validation report, scalar diagnostics, and visual
QC overlays with:

```bash
beltmap-validate --output-dir outputs
```

The validation command reads the standard driver outputs and writes:

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
  detections_overlay_sample_000001.png
  detections_overlay_sample_000002.png
  tracks_overlay_sample_000000.png
  tracks_overlay_sample_000001.png
  tracks_overlay_sample_000002.png
```

The overlay images are intended for manual sanity checks on real conveyor data:
use them to verify whether detections are actual particles and whether track
segments connect the correct components.

The Brick 10g/s GitHub Actions workflow runs this validation step automatically
after a successful `beltmap-apply` job, so downloaded workflow artifacts include
the report, plots, and overlay samples.

Compare several output directories, for example detection-only threshold sweeps, with:

The visual contact sheet can only show frames that have residual preview PNGs in
each compared output directory. Before running the compared jobs, set
`DEBUG_RESIDUAL_PREVIEW_INTERVAL_FRAMES` or
`--debug-residual-preview-interval-frames` so the frames passed to `--frames`
are actually saved.

```bash
beltmap-compare \
  --run T4.0=outputs/T4p0 \
  --run T3.5=outputs/T3p5 \
  --run T3.0=outputs/T3p0 \
  --frames 0,248,496,744,992 \
  --report-dir outputs/threshold_comparison
```

This writes `comparison_report.md`, `summary.csv`, detection-count and
velocity-ratio plots, and a side-by-side residual preview contact sheet with
detection boxes.

For real conveyor data, first create a sparse but adversarial labeling plan from
one representative run:

```bash
beltmap-suggest-label-frames \
  --output-dir outputs/T12_area50 \
  --frames 60 \
  --empty-frames 12 \
  --output labels/brick_validation_plan.csv \
  --template-output labels/brick_validation_boxes.csv
```

The plan deliberately mixes detection spikes, particle-free or low-detection
candidate frames, poor registration frames, large phase-correction frames,
recurrent-artifact-heavy frames, photometric outliers, and regular controls.
Fill one template row per particle box; keep a blank row only when the whole
frame has been inspected and is intentionally scored as empty.

When this small real-data validation subset has been labeled, pass it to the same
comparison command to rank variants by detection precision, recall, and F1 on
the labeled frames rather than by proxy metrics alone:

```bash
beltmap-compare \
  --run T4.0=outputs/T4p0 \
  --run T3.5=outputs/T3p5 \
  --truth-path labels/brick_validation_boxes.csv \
  --truth-iou-threshold 0.25 \
  --frames 0,248,496,744,992 \
  --report-dir outputs/threshold_comparison
```

The label file may be a CSV with `frame_index`, `bbox_top`, `bbox_left`,
`bbox_bottom`, and `bbox_right` columns, or a JSON object/list with equivalent
`top`, `left`, `bottom`, and `right` fields. Coordinates are crop-local and use
the same half-open bounding-box convention as `detections.csv`. To include
labeled empty frames, leave CSV rows containing only `frame_index`, or use a JSON
object with `scored_frames`. Detections outside the scored frame set are ignored
by the labeled metrics.

Post-process track velocities with conservative physical gates using:

```bash
beltmap-filter-tracks \
  --output-dir outputs/T3p5 \
  --min-track-length 5 \
  --min-velocity-ratio-y 0.0 \
  --max-velocity-ratio-y 1.1
```

This leaves the raw `velocities.csv` untouched and writes `track_scores.csv`
plus `filtered_velocities.csv`; when `tracks.csv` is available, it also writes
`filtered_tracks.csv` for trajectory-level overlays and downstream analysis. The
same filter is also applied automatically at the end of `beltmap-apply`.

If the input sequence was generated by the synthetic example, compute quantitative
ground-truth metrics with:

```bash
beltmap-benchmark \
  --output-dir outputs \
  --truth-path data/images/synthetic_metadata.json
```

This writes:

```text
outputs/
  benchmark_metrics.json
  benchmark_report.md
```

`beltmap-validate` is an operational health check for any run; `beltmap-benchmark`
requires synthetic ground truth and reports phase error, cyclic belt-map RMSE,
detection precision/recall/F1, velocity error, runtime, and peak memory.

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

For image-fixed residual structures, enable static residual-noise learning. This
estimates `static_noise(y, x) = 1.4826 * MAD_t(residual_t(y, x))` from sampled
belt-subtracted residuals and detects with
`residual / max(local_noise, static_noise)`:

```toml
[static_noise]
sample_frames = 500
mask_threshold = 4.0
```

Bright brick particles on a dark belt can then be detected by thresholding the
normalized residual:

```python
particle_mask = detect_particles_from_residual(residual, threshold=5.0)
```

For dark particles or mixed-polarity residual artifacts, select the residual
polarity explicitly. The optional low threshold enables hysteresis growing from
strong seed pixels into adjacent weaker particle shoulders:

```python
dark_mask = detect_particles_from_residual(residual, threshold=5.0, mode="negative")
mixed_mask = detect_particles_from_residual(residual, threshold=5.0, mode="absolute", low_threshold=2.0)
```

Connected-component extraction is implemented with a pure NumPy fallback. For
large residual masks, install the optional acceleration dependencies:

```bash
python -m pip install ".[speed]"
```

When available, BeltMap uses `scipy.ndimage.label` first, then
`skimage.measure.label`, and falls back to the pure NumPy implementation when
neither optional backend is installed.

For scratch-heavy residuals, reject line-like components before tracking with
the optional shape gates:

```toml
[detection]
threshold = 3.5
min_area_px = 4
min_bbox_width_px = 3
min_bbox_height_px = 3
max_bbox_aspect_ratio = 4.0
min_bbox_extent = 0.15
```

To suppress belt-fixed scratches or map ghosts that recur at the same belt
phase, enable recurrent artifact filtering:

```toml
[recurrent_artifact]
min_revolutions = 3
margin_px = 2
max_overlap_fraction = 0.3
mode = "hard"
soft_penalty_weight = 1.0
```

For detection-only sweeps, a previously built artifact map can be reused:

```toml
[reuse]
recurrent_artifact_map_path = "previous/recurrent_artifact_map.npy"
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

## Citation

If you use BeltMap in academic work, please cite the software using the metadata
in [`CITATION.cff`](CITATION.cff).

## License

BeltMap is released under the MIT License. See [`LICENSE`](LICENSE) for the full
license text.
