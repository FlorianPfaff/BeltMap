# Advanced result-improvement patch notes

This patch adds opt-in tooling for the larger result-improvement ideas that are
better evaluated before becoming driver defaults.

Implemented as concrete utilities or CLIs:

- robust per-frame photometric gain/offset fitting;
- sub-grid phase-offset estimation from a sampled loss curve;
- diagnostic two-dimensional registration shifts;
- phase/velocity smoothing for registration sequences;
- belt-map uncertainty and seam-discontinuity helpers;
- robust Theil-Sen velocity fitting and continuous track-confidence scoring;
- sparse real-data label evaluation with IoU matching;
- parameter sweep generation/execution;
- synthetic stress-test generation for weak texture, illumination drift, faint
  particles, high density, and negative velocity;
- provenance and failure-mode reporting.

Commands:

```bash
beltmap-advanced-report --output-dir outputs --image-dir data/images
beltmap-evaluate-real --output-dir outputs --labels labels.json
beltmap-sweep --base-config examples/brick_10gpers/beltmap.toml \
  --param detection.threshold=3.5,4.0,4.5 \
  --param map.sample_frames=120,500
beltmap-synthetic-suite --output-root outputs/synthetic_suite --execute
```

For synthetic cases, add `--benchmark-truth-path` to `beltmap-sweep` to write
`sweep_metrics.csv`, `sweep_metrics.json`, and `sweep_report.md`. Those files
turn threshold grids into precision-recall, F1-threshold, false-positive,
PyRecEst track-length, single-frame-track, fragmentation, missed-event, and
velocity-bias curve data instead of isolated single-threshold scores.

The helpers are intentionally not all enabled in `beltmap-apply` by default.
They are meant to make the next round of experiments reproducible and to reduce
risk: once a helper is shown to improve the Brick sequence and the synthetic
stress suite, it can be promoted into the main driver path with a small follow-up
patch.
