# Post-run result-improvement patch notes

This patch adds a second, opt-in layer around the core BeltMap driver.  It is
intended for result review, parameter sweeps, CI checks, and sparse real-data
label planning.  None of these features changes the default detector or tracker;
they consume standard output files and write additional diagnostics.

## New commands

```bash
beltmap-postrun-audit --output-dir outputs --frame-rate-hz 100
beltmap-map-uncertainty --output-dir outputs --write-full-counts
beltmap-suggest-label-frames --output-dir outputs --frames 50 --output labels/label_plan.csv
beltmap-quality-contract --output-dir outputs
beltmap-quality-contract --write-template quality_contract.json
beltmap-quality-contract --write-synthetic-template synthetic_contract.json
```

`beltmap-postrun-audit` writes a compact report bundle under
`outputs/postrun_audit`:

- `quality_flags.json` and `quality_flags.md`
- worst-frame CSVs for registration, detection spikes, recurrent-artifact
  rejections, and photometric RMSE when available
- `belt_map_row_counts.npy`, `belt_map_row_uncertainty.npy`, and preview PNGs
- optional full `belt_map_counts.npy` and `belt_map_uncertainty.npy`
- `seam_diagnostics.json` and `seam_discontinuity_profile.png`
- `detection_confidence.csv`
- `adaptive_map_frame_plan.csv`
- `label_plan.csv`
- `flux_summary.json`
- `quality_contract.json` and `quality_contract.md`
- `short_horizon_track_diagnostics.csv`

## Trimmed-mean map reconstruction safety

The generated configuration template now defaults
`map.reconstruction_trim_fraction` to `0.0`.  The previous template value of
`0.1` can require a dense sample-by-map stack during map construction.  For large
periodic maps and hundreds of samples this can become very memory-heavy.  Prefer
Huber aggregation for large real-data runs:

```toml
[map]
reconstruction_trim_fraction = 0.0
aggregation = "huber"
robust_iterations = 1
robust_huber_delta = 3.0
robust_min_scale = 1.0
```

## Map coverage and uncertainty

`beltmap-map-uncertainty` estimates belt-coordinate row exposure from
`phase_estimates.csv`, `metadata.json`, and the belt-map shape.  This is an
approximation: it measures which belt rows were visible in rendered frames, not
which individual pixels survived every particle mask during map reconstruction.
It is still useful for spotting seam and low-coverage regions.  Use
`--write-full-counts` only when the full 2-D arrays are acceptable for the data
size.

## Worst-frame review and label planning

Worst-frame tables are built from existing run CSVs.  They intentionally select
frames that are likely to expose failure modes rather than only the first few
frames.  The label-plan command combines detection spikes, empty/low-detection
frames, low registration score, large phase correction, recurrent-artifact
rejection, and photometric-fit RMSE when available.

## Detection confidence

`detection_confidence.csv` adds a post-hoc confidence score combining peak and
mean residual signal, component shape extent, registration score, and recurrent
artifact overlap/probability.  It is not a calibrated probability.  It is a
ranking signal for review, sweeps, and precision/recall curves once sparse labels
exist.

## Seam diagnostics and rolling

`seam_diagnostics.json` reports the current row-0 seam jump and the belt-map row
with the smallest adjacent-row jump.  The patch does not automatically roll the
map.  Treat the suggested row as a diagnostic first; rolling the map also requires
consistent phase/reference metadata updates.

## Geometry, masks, color residuals, and tracking diagnostics

The new `beltmap.postrun_improvements` module also contains reusable helpers for
belt-edge ignore masks, loading and applying ignore masks, RGB residual scoring,
perspective warping with a homography, and short-horizon track-fragmentation
diagnostics.  These are intentionally API utilities rather than default driver
behavior because they need dataset-specific validation.

## Quality contracts

`beltmap-quality-contract` evaluates a small pass/fail contract for operational
runs.  It can return a nonzero exit status, which makes it suitable for CI or
large sweeps.  The synthetic contract template is a starting point for CI-level
benchmark thresholds; fill in project-specific tolerances before enforcing it.
