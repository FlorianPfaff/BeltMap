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
- `tracks` and `filtered_tracks`: PyRecEst track continuity statistics,
  including mean/median track length and single-frame track counts before and
  after final track filtering.
- `events`: track/event precision, recall, F1, temporal coverage, and
  `track_fragmentation`, the number of extra predicted event fragments per
  truth event, plus birth false-positive and missed-event rates.
- `velocity`: representative-track vertical velocity, velocity-ratio errors,
  and all-row velocity mean absolute error, bias, standard deviation, and
  variance.
- `runtime`: elapsed time, throughput, and peak resident memory when available.

Phase errors are circular because the belt coordinate wraps modulo the belt
period. Belt-map RMSE is minimized over cyclic vertical shifts so a constant
phase offset in the reconstructed map is not counted as a texture-reconstruction
error.

## Threshold sweeps

Use `beltmap-sweep` with `--benchmark-truth-path` to turn a parameter grid into
curve data. For detection thresholds, each row is one operating point:

```bash
beltmap-sweep \
  --base-config outputs/synthetic_suite/faint_particles/beltmap.toml \
  --param detection.threshold=1.5,2.0,2.5,3.0,3.5,4.0 \
  --output-root outputs/threshold_sweep/faint_particles \
  --execute \
  --benchmark-truth-path outputs/synthetic_suite/faint_particles/synthetic_metadata.json
```

This writes:

```text
outputs/threshold_sweep/faint_particles/
  sweep_manifest.json
  sweep_metrics.csv
  sweep_metrics.json
  sweep_froc_curve.svg
  sweep_report.md
```

`sweep_metrics.csv` includes detection precision/recall/F1, false positives per
frame, event F1, filtered event F1, mean/median track length, single-frame
tracks, track fragmentation, birth false-positive rate, missed-event rate,
velocity mean absolute error, velocity bias, phase RMSE, and map RMSE. Plot
these columns as precision-recall, F1-vs-threshold,
false-positives-vs-recall, track-length-vs-threshold,
fragmentation-vs-threshold, and velocity-bias-vs-threshold curves. This is
stronger evidence than reporting one F1 value at one arbitrary threshold.
The Markdown report embeds `sweep_froc_curve.svg` as the ready-made detection
FROC view.

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
