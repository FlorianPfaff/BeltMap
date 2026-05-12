# Visual quality-control artifacts

`beltmap-validate` writes scalar diagnostics and visual QC artifacts after a
standard `beltmap-apply` run. The visual QC outputs are designed for manual
inspection on real conveyor data, where scalar summaries alone are often not
enough to decide whether the result is credible.

## Generated files

| File | Meaning |
|---|---|
| `residual_histogram.png` | Histogram over saved residual preview PNG intensities. This is a quick sanity check for unusual clipping, saturation, or nearly empty residual previews. |
| `belt_map_coverage.png` | Nominal belt-map observation coverage induced by the recovered phase trajectory and crop geometry. This is a QC proxy, not the exact driver accumulation mask. |
| `overlay_contact_sheet.png` | Side-by-side sheet pairing detection and track overlay samples by frame. |
| `detections_overlay_sample_*.png` | Residual preview images with current-frame detection boxes and centroids overlaid. |
| `tracks_overlay_sample_*.png` | Residual preview images with simple nearest-neighbor track polylines overlaid for sanity checking association quality. |

## Interpretation

Use the detection overlays to answer whether the residual threshold is actually
selecting particle components. Typical warning signs are boxes on belt texture,
boxes on crop boundaries, repeated false positives at the same belt texture
feature, or missing obvious bright particles.

Use the track overlays to answer whether association links are plausible. Typical
warning signs are trajectories jumping between nearby components, tracks crossing
implausibly, or tracks that connect detections across long visual gaps.

Use the coverage plot to identify weakly observed belt-map rows. Poor coverage
can explain residual ghosts, interpolation bands, and unstable detections.

## Limitations

The visual overlays are generated as a post-processing validation step. They use
saved residual preview PNGs and detections written by the driver. Track overlays
currently reconstruct a simple nearest-neighbor association for visualization;
they should be treated as a sanity check, not as a replacement for quantitative
tracking evaluation.

`residual_histogram.png` summarizes display-scaled residual preview intensities,
not raw normalized residual arrays. Use the CSV outputs or direct Python API
calls for quantitative analysis.
