# Synthetic sequence example

This example is self-contained and does not require any external dataset. It
creates a small grayscale conveyor-belt image sequence with:

- a periodic belt texture moving downward by 2 px/frame;
- one bright square particle moving downward by 1 px/frame, slower than the belt;
- 12 frames of size 64 × 48 pixels.

The example is intentionally tiny so it can run in CI and on a laptop while
exercising BeltMap's map reconstruction, residual rendering, particle detection,
tracking, and velocity output path.

## Run from an installed checkout

From the repository root:

```bash
python -m pip install -e ".[test]"
bash examples/synthetic_sequence/run.sh
```

The script generates frames below `data/images`, runs `beltmap-apply` with
`examples/synthetic_sequence/beltmap.toml`, and validates the expected outputs.

## Run step by step

```bash
python examples/synthetic_sequence/generate.py --output-dir data/images --frames 12
beltmap-apply --config examples/synthetic_sequence/beltmap.toml --print-config
python examples/synthetic_sequence/validate_outputs.py --output-dir outputs
```

Expected generated inputs:

```text
data/images/
  frame_000.png
  frame_001.png
  ...
  frame_011.png
  synthetic_metadata.json
```

Expected BeltMap outputs:

```text
outputs/
  belt_map.npy
  belt_map.png
  config_resolved.json
  detections.csv
  detections_per_frame.csv
  metadata.json
  phase_estimates.csv
  progress.jsonl
  progress_latest.json
  residual_frame_000000.png
  residual_frame_000001.png
  residual_frame_000002.png
  velocities.csv
```

## Configuration

The example config fixes the known synthetic motion parameters:

```toml
[belt]
velocity_px_per_frame = 2.0
period_px = 64
```

The detector threshold and minimum area are intentionally permissive because the
synthetic particle is small:

```toml
[detection]
threshold = 3.0
min_area_px = 2
```

Use this example as a smoke test for the full image-sequence driver before
running BeltMap on real conveyor-belt data.
