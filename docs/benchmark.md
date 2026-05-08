# Synthetic ground-truth benchmark

BeltMap includes two complementary checks for an image-sequence run:

- `beltmap-validate` is an operational sanity check. It reads normal driver
  outputs and creates diagnostic plots plus a Markdown validation report.
- `beltmap-benchmark` is a quantitative ground-truth check for synthetic runs.
  It compares the run against known latent truth written by the synthetic
  sequence generator.

The benchmark does not require manually annotated real data. It is deliberately
based on simulated conveyor-belt sequences, where the true belt phase, clean belt
map, particle boxes, particle velocity, and velocity ratio are known.

## Run the benchmark

From a repository checkout:

```bash
python -m pip install -e ".[test]"
python examples/synthetic_sequence/generate.py --output-dir data/images --frames 12
beltmap-apply --config examples/synthetic_sequence/beltmap.toml
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

The all-in-one example script also runs the benchmark:

```bash
bash examples/synthetic_sequence/run.sh
```

## Metrics

`benchmark_metrics.json` contains these sections:

- `phase`: circular phase errors in belt-period pixels.
- `belt_map`: cyclic-shift-invariant RMSE and mean absolute error between the
  reconstructed belt map and the true synthetic belt map.
- `detections`: greedy per-frame IoU matching against synthetic particle boxes,
  with precision, recall, F1, IoU, and centroid-error statistics.
- `velocity`: representative-track vertical velocity and velocity-ratio errors.
- `runtime`: elapsed time, throughput, and peak resident memory when available.

Phase errors are circular because the belt coordinate wraps modulo the belt
period. Belt-map RMSE is minimized over cyclic vertical shifts so a constant
phase offset in the reconstructed map is not counted as a texture-reconstruction
error.

## Why synthetic first?

Ground-truth real conveyor-belt data would require manual annotation and still
would not directly reveal the true clean belt texture or true phase. Synthetic
and semi-synthetic cases make these latent variables observable, which makes them
well suited for regression tests and algorithm comparisons.

Recommended future cases include:

- weak belt texture;
- incorrect supplied belt velocity;
- faint particles;
- large or merged particles;
- many particles per frame;
- negative belt velocity;
- particles moving faster or slower than the belt;
- nonuniform illumination;
- camera noise and blur;
- particles at crop boundaries.

A later real-data benchmark can be added as a small curated sanity check, but the
synthetic benchmark should remain the reproducible baseline.
