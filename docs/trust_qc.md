# Trust and evidence workflow

`beltmap-trust` adds preflight and result-audit checks that are deliberately
separate from the core detector. The goal is to make a run easier to trust before
using its counts, tracks, or velocities in an experiment report.

## Full report

After a normal run and optional visual validation:

```bash
beltmap-trust report \
  --image-dir data/images \
  --output-dir outputs \
  --frame-rate-hz 100 \
  --expected-mass-flux-g-s 10.0 \
  --particle-mass-g 0.02 \
  --epoch-count 4
```

This writes:

```text
outputs/
  sequence_qc.json
  frame_quality_summary.json
  speed_consistency.json
  run_drift_summary.json
  physical_validation.json
  detection_edge_audit.csv
  detection_confidence.csv
  events.csv
  rejection_audit.csv
  map_epoch_plan.csv
  minimum_evidence_report.md
  trust_artifacts.json
```

The report is intended to answer the operational questions that visual overlays
alone cannot answer:

- Are image numbers missing or duplicated?
- Are frames saturated, clipped, or motion blurred?
- Does the supplied belt speed agree with registration corrections?
- Are detections drifting over time, suggesting contamination or stale maps?
- Do particle counts or mass-flux estimates agree with independent measurements?
- Which detections or tracks were rejected and why?
- Which detections touch crop edges and should be treated as truncated?

## Sequence and image-quality preflight

Before tuning algorithms, check whether the input sequence itself is sane:

```bash
beltmap-trust check-sequence --image-dir data/images --output outputs/sequence_qc.json
beltmap-trust quality \
  --image-dir data/images \
  --belt-region 0,220,1330,1800 \
  --output outputs/frame_quality_summary.json
```

The sequence check reports missing frame numbers, duplicate frame numbers,
duplicate image files, and sampled image dimensions. The quality check reports
Laplacian blur, gradient energy, saturation, dark clipping, and robust intensity
range.

## Belt-speed audit

The core phase model is sensitive to the signed belt velocity. A linear trend in
registration corrections usually indicates a small velocity or timing mismatch:

```bash
beltmap-trust speed-audit --output-dir outputs
```

The audit estimates a correction slope and an inferred belt velocity:

```text
inferred_belt_velocity_px_per_frame =
    configured_belt_velocity_px_per_frame - correction_slope_px_per_frame
```

Large correction trends, low registration scores, or many boundary corrections
should be treated as warnings before interpreting particle velocities.

## Physical validation

If an experiment has independent mass-flow, particle-count, or feed-rate
measurements, compare them against the image-derived event table:

```bash
beltmap-trust physical-validation \
  --output-dir outputs \
  --frame-rate-hz 100 \
  --expected-mass-flux-g-s 10.0 \
  --particle-mass-g 0.02
```

This does not replace object-level evaluation, but it is a valuable sanity check:
a visually plausible detector can still overcount or undercount badly at the run
level.

## Edge-aware particle counting

`detection_edge_audit.csv` adds:

```text
touches_top_edge
touches_bottom_edge
touches_left_edge
touches_right_edge
is_truncated
```

Use these flags to exclude truncated detections from particle-size summaries or
to diagnose crop-boundary misses.

## Event-level aggregation and confidence

`events.csv` aggregates `tracks.csv` or `filtered_tracks.csv` into one row per
countable event. `detection_confidence.csv` adds a simple confidence score based
on peak signal, area, recurrent-artifact overlap, and edge truncation.

The confidence is intentionally a heuristic until labeled or semi-synthetic data
are available for calibration.

## Rejection audit

`rejection_audit.csv` combines available recurrent-artifact rejections and track
filter failures into one table. It helps answer whether candidate particles are
lost because of recurrent-artifact filtering, short tracks, velocity-ratio gates,
or lateral-velocity gates.

## Multi-epoch map planning

`map_epoch_plan.csv` suggests frame ranges for multi-epoch belt maps. It does
not build those maps automatically; it gives a reproducible plan for splitting
long runs when drift diagnostics indicate that one static belt map may be stale.

## Calibration target

Pixel-to-mm scale can be estimated from two clicked calibration-target points:

```bash
beltmap-trust calibrate-scale \
  --point-a 100,200 \
  --point-b 100,700 \
  --known-distance-mm 50 \
  --output calibration.json
```

The output contains `px_per_mm` and `mm_per_px`, which can be used by downstream
analysis to convert pixel velocities and sizes into physical units.

## Cost-sensitive profiles

Write a configuration overlay for a common operating point:

```bash
beltmap-trust write-profile high_precision --output high_precision.toml
beltmap-trust write-profile high_recall --output high_recall.toml
beltmap-trust write-profile velocity_quality --output velocity_quality.toml
beltmap-trust write-profile map_quality --output map_quality.toml
beltmap-trust write-profile fast_screening --output fast_screening.toml
```

Profiles are intentionally small overlays. They should be merged with a dataset
specific base config rather than replacing it wholesale.

## Cross-run comparability

Before comparing outputs from several runs, check whether their key run settings
match:

```bash
beltmap-trust compare-runs \
  --run outputs/baseline \
  --run outputs/tuned \
  --output outputs/comparability.json
```

The command compares crop, image shape, velocity, frame stride, map height,
threshold, minimum area, and phase-estimate source.
