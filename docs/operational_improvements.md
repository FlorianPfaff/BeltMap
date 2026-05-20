# Operational result-improvement toolkit

This patch adds opt-in utilities for calibration, quality control, reproducibility,
and reporting around the core BeltMap driver.  The tools are intentionally small
and NumPy/Pillow-only so they can be used before a full run, after validation, or
inside future driver integrations.

## Added commands

| Command | Purpose |
|---|---|
| `beltmap-detect-roi` | Suggest a belt crop from temporal motion energy. |
| `beltmap-estimate-period` | Estimate a belt period from a reconstructed `belt_map.npy`. |
| `beltmap-suggest-threshold` | Recommend a residual threshold from empirical residual tails or FDR. |
| `beltmap-manifest` | Write a hashed image-directory manifest for reproducibility. |
| `beltmap-flux-summary` | Export particle-flux and velocity-distribution summaries. |
| `beltmap-review` | Build a lightweight HTML review page from overlay/residual previews. |
| `beltmap-write-templates` | Write Snakemake, Nextflow, Docker, Apptainer, ruff, and pre-commit templates. |
| `beltmap-stream-snapshot` | Track newly arrived frames for online-processing prototypes. |

## Added API areas

All helpers live in `beltmap.operational_improvements`.

### Calibration and geometry

- `suggest_belt_region_from_frames` estimates the moving belt region from frame-to-frame motion energy.
- `estimate_homography` and `warp_perspective` provide a lightweight projective rectification path.
- `estimate_period_from_belt_map` and `estimate_period_from_profile` estimate periodic texture length.
- `select_adaptive_map_frames` chooses map-build samples that improve phase coverage.

### Detection, masks, and residual statistics

- `load_ignore_mask`, `apply_ignore_mask`, and `belt_edge_ignore_mask` support fixed ignore regions and belt-edge margins.
- `recommend_threshold`, `empirical_p_values`, and `fdr_threshold_from_p_values` support empirical thresholding and false-discovery-rate control.
- `particle_density_score` and `rank_frames_by_particle_density` help exclude heavily contaminated frames during map construction.
- `split_merged_components` provides a dependency-free projection-gap splitter for large merged masks.
- `particle_descriptor_from_mask` adds calibrated-ish shape descriptors: equivalent diameter, axes, orientation, integrated signal, and extent.

### Uncertainty, timing, and event-level outputs

- `estimate_centroid_uncertainty` and `robust_velocity_fit` provide approximate uncertainty estimates.
- `load_timestamps_csv` and `TimestampTable` support irregular frame-time workflows.
- `classify_event` distinguishes loose-particle, belt-fixed-artifact, and map-uncertainty cases with rule-based reasons.
- `summarize_flux` and `write_science_exports` create experiment-level analysis products.

### Reproducibility and deployment

- `dataset_manifest` records sorted image paths, sizes, hashes, dimensions, and modified times.
- `runtime_provenance` records Python, platform, dependency, and BeltMap environment context.
- `write_workflow_templates`, `write_container_templates`, and `write_quality_tooling_templates` create starting points for Snakemake/Nextflow, Docker/Apptainer, and ruff/pre-commit.

### Larger-feature scaffolding

- `discover_new_stream_frames` and `incremental_update_map` are the minimal online-processing building blocks.
- `load_detector_plugin` and `run_detector_plugin` define a simple `module:function` plugin interface for learned detectors.
- `randomize_synthetic_frame` supports domain-randomized synthetic stress tests.
- `stitch_multicamera_events` greedily groups detections across cameras by time and belt phase.
- `run_pipeline_stage` records elapsed time and provenance for pure pipeline stages.

## Example use

```bash
beltmap-detect-roi --image-dir data/images --output belt_region_suggestion.json
beltmap-manifest --image-dir data/images --output data_manifest.json
beltmap-apply --config examples/brick_10gpers/beltmap.toml
beltmap-validate --output-dir outputs
beltmap-review --output-dir outputs
beltmap-flux-summary --velocities outputs/filtered_velocities.csv --output-dir outputs/science --frame-rate-hz 100
```

## Scope and limitations

This patch makes every roadmap item concrete, but several features are deliberately
implemented as safe opt-in utilities rather than invasive changes to the core
image driver.  In particular, homography correction, streaming map updates,
learned detectors, and multi-camera event stitching need dataset-specific
validation before they should be enabled by default.
