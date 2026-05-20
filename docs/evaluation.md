# BeltMap evaluation and ablations

Use `beltmap-evaluate` after running `beltmap-apply` on one or more configurations. The command reads the standard output files produced by the driver and writes a compact comparison table in JSON, CSV, and Markdown form.

Example:

```bash
beltmap-apply --config baseline.toml --output-dir outputs/baseline
beltmap-apply --config phase_feedback.toml --output-dir outputs/phase_feedback
beltmap-evaluate \
  --run baseline=outputs/baseline \
  --run phase_feedback=outputs/phase_feedback \
  --output-dir outputs/evaluation
```

The generated artifacts are:

- `evaluation_summary.json` for scripts and dashboards,
- `evaluation_summary.csv` for spreadsheets,
- `evaluation_summary.md` for review in pull requests or experiment logs.

The summary intentionally uses only files that are already produced by `beltmap-apply`: `metadata.json`, `progress.jsonl`, `phase_estimates.csv`, `detections.csv`, `detections_per_frame.csv`, and `velocities.csv`. This makes it suitable for quick ablations before a fully labeled benchmark is available.

Recommended ablations:

- baseline vs. phase feedback enabled,
- mean belt map vs. particle-masked belt map,
- static residual background/noise learning off vs. on,
- detection-threshold sweeps,
- tracker-parameter sweeps.

Interpret the proxy metrics conservatively. Lower registration scores and smaller absolute phase corrections usually indicate a cleaner phase model, but detection-count and velocity-ratio changes should still be checked against residual previews and the experiment physics.
