# Brick 10 g/s Zenodo example

This example configuration runs BeltMap on the Brick 10 g/s conveyor-belt image
sequence used by the GitHub Actions workflow.

The image dataset is not stored in this repository. The workflow downloads or
reuses a local cache of the Zenodo archive:

```text
https://zenodo.org/records/7802579/files/images_Brick_10gpers.zip?download=1
```

## Expected input layout

After extracting the dataset, make the image directory available as:

```text
data/images/
```

For example:

```bash
mkdir -p data
ln -s /path/to/extracted/images_Brick_10gpers data/images
```

## Run

From the repository root:

```bash
python -m pip install -e ".[test]"
beltmap-apply --config examples/brick_10gpers/beltmap.toml
beltmap-validate --output-dir outputs
```

The run writes the standard BeltMap outputs below `outputs/`, including:

```text
belt_map.npy
belt_map.png
config_resolved.json
metadata.json
phase_estimates.csv
detections.csv
detections_per_frame.csv
velocities.csv
progress.jsonl
validation_report.md
```

See `docs/outputs.md` for the output schema and `docs/validation.md` for
diagnostic checks.

## Configuration notes

The default crop is the known Brick belt region:

```toml
[belt]
region = [0, 220, 1330, 1800]
```

The default signed belt image velocity is:

```toml
velocity_px_per_frame = 59.3
```

The default belt period is:

```toml
period_px = 14723
```

Set `frames.max_frames` to a positive value for a shorter test run:

```toml
[frames]
max_frames = 200
```

Set `frames.stride` above 1 to subsample the sequence:

```toml
[frames]
stride = 5
```

The GitHub workflow exposes most of these values as workflow-dispatch inputs so
that long real-data runs can be adjusted without editing the committed config.
The committed config is the reference default, while workflow inputs generate an
equivalent temporary config for CI or runner use.
