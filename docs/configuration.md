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
  the string `"auto"`. Positive velocity means the belt texture moves downward
  in image coordinates.
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
| `frames.max_frames` | `MAX_FRAMES` | `--max-frames` | `0` | frames | Maximum number of selected frames to process after sorting and striding. `0` means process all selected frames. |
| `frames.stride` | `FRAME_STRIDE` | `--frame-stride` | `1` | frames | Process every Nth frame after natural filename sorting. Must be at least 1. |
| `belt.region` | `BELT_REGION` | `--belt-region` | full frame | px | Belt crop as `top,left,height,width`. Coordinates are full-frame image coordinates. Omit only when the full frame is belt texture. |
| `belt.velocity_px_per_frame` | `BELT_VELOCITY_PX_PER_FRAME` | `--belt-velocity-px-per-frame` | `auto` | px/frame | Signed vertical belt texture velocity. Use `auto` to estimate from frame-to-frame vertical correlation shifts. |
| `belt.period_px` | `BELT_PERIOD_PX` | `--belt-period-px` | unset | px | Belt circumference/period in belt-map pixels. If unset or non-positive, the driver builds a finite map covering the selected sequence phase range. |
| `detection.threshold` | `DETECTION_THRESHOLD` | `--detection-threshold` | `5.0` | z | Threshold on normalized residuals for final bright-particle detection. |
| `detection.min_area_px` | `MIN_AREA_PX` | `--min-area-px` | `4` | px | Minimum connected-component area for final particle detections. Must be at least 1. |
| `tracking.min_track_length` | `MIN_TRACK_LENGTH` | `--min-track-length` | `2` | detections | Minimum number of detections required before a particle track contributes a velocity row. Must be at least 1 at driver parsing and at least 2 for velocity estimation. |
| `tracking.max_match_distance_px` | `MAX_MATCH_DISTANCE_PX` | `--max-match-distance-px` | `max(5, 1.5 * abs(belt_velocity))` | px | Maximum frame-to-frame nearest-neighbor association distance for tracking. Leave unset to derive it from the belt speed. |
| `map.sample_frames` | `MAP_SAMPLE_FRAMES` | `--map-sample-frames` | `120` | frames | Number of frames sampled across the selected sequence to reconstruct the belt map. Must be at least 1. |
| `map.mask_iterations` | `MAP_MASK_ITERATIONS` | `--map-mask-iterations` | `1` | passes | Number of particle-masked belt-map refinement passes after the initial provisional map. `0` disables particle masking during map reconstruction. |
| `map.particle_mask_threshold` | `MAP_PARTICLE_MASK_THRESHOLD` | `--map-particle-mask-threshold` | `detection.threshold` | z | Strong residual threshold used to seed particle masks while building the clean belt map. |
| `map.particle_mask_mode` | `MAP_PARTICLE_MASK_MODE` | `--map-particle-mask-mode` | `positive` | mode | Map-building particle-mask mode. Valid values are `positive`, `absolute`, and `hysteresis_abs`. |
| `map.particle_mask_grow_threshold` | `MAP_PARTICLE_MASK_GROW_THRESHOLD` | `--map-particle-mask-grow-threshold` | `2.0` | z | Lower absolute-residual threshold used to grow `hysteresis_abs` map masks from strong seeds. Ignored by `positive` and `absolute`. |
| `map.particle_mask_dilation_px` | `MAP_PARTICLE_MASK_DILATION_PX` | `--map-particle-mask-dilation-px` | `0` | px | Morphological dilation radius for `hysteresis_abs` map masks before applying the rectangular safety margin. `0` disables dilation. |
| `map.particle_mask_margin_px` | `MAP_PARTICLE_MASK_MARGIN_PX` | `--map-particle-mask-margin-px` | `8` | px | Safety margin added around detected or grown particle regions during map reconstruction. |
| `map.particle_mask_min_area_px` | `MAP_PARTICLE_MASK_MIN_AREA_PX` | `--map-particle-mask-min-area-px` | `detection.min_area_px` | px | Minimum component area used for particle masking during map reconstruction. Must be at least 1. |
| `auto_velocity.search_radius_px` | `VELOCITY_SEARCH_RADIUS_PX` | `--velocity-search-radius-px` | `50` | px | Maximum vertical shift searched for each adjacent-frame pair during automatic belt-velocity estimation. Increase if the belt moves farther than this between frames. |
| `auto_velocity.estimation_pairs` | `VELOCITY_ESTIMATION_PAIRS` | `--velocity-estimation-pairs` | `100` | pairs | Number of adjacent-frame pairs used for automatic belt-velocity estimation, capped by the available sequence length. |
| `auto_velocity.min_abs_px_per_frame` | `AUTO_VELOCITY_MIN_ABS_PX_PER_FRAME` | `--auto-velocity-min-abs-px-per-frame` | `0.25` | px/frame | Minimum accepted absolute value of the auto-estimated belt velocity. Helps reject static-background-dominated crops. |
| `auto_velocity.max_edge_fraction` | `AUTO_VELOCITY_MAX_EDGE_FRACTION` | `--auto-velocity-max-edge-fraction` | `0.2` | fraction | Maximum accepted fraction of adjacent-frame shifts that land near the search-radius edge. Must be in `[0, 1]`. |
| `auto_velocity.allow_full_frame` | `ALLOW_FULL_FRAME_AUTO_VELOCITY` | `--allow-full-frame-auto-velocity` / `--no-allow-full-frame-auto-velocity` | `false` | bool | Allow `belt.velocity_px_per_frame = "auto"` when `belt.region` is the full frame. Keep this false unless the full frame really contains only belt texture. |
| `registration.search_radius_px` | `REGISTRATION_SEARCH_RADIUS_PX` | `--registration-search-radius-px` | `8.0` | px | Local phase-registration search radius around the constant-speed prediction. |
| `registration.search_step_px` | `REGISTRATION_SEARCH_STEP_PX` | `--registration-search-step-px` | `0.5` | px | Local phase-registration candidate spacing. Must be positive. |
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
