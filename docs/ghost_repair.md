# GhostRepair mode policy

GhostRepair localizes learned-map ghosts by projecting map-only negative-control
tracks into belt-map coordinates. The generated defect mask is useful for
analysis and for repair experiments, but the available repair modes have
different paper status.

## Mode status

| Mode | Role | Paper-facing status |
| --- | --- | --- |
| `original` | Baseline learned map | Reference only. |
| `local_inpaint` | Prototype/control map-space fill | Diagnostic only. Do not use as the final repaired map without texture-plausibility checks and rerun real-label metrics. |
| `rebuild_masked` | Raw-frame rebuild with defect coordinates excluded from map accumulation | Preferred implemented repair direction, but still requires map-only and labeled-quality reruns before claims. |

## Why `local_inpaint` is quarantined

`local_inpaint` edits the learned `belt_map.npy` directly by filling the defect
mask from nearby map pixels. It can suppress map-only false detections, but it
may also create artificial gray or high-frequency texture patches. Passing the
map-only negative control is therefore necessary but not sufficient: a repair
also needs texture-plausibility checks and real-frame/labeled-quality reruns.

The CLI still writes `repaired_belt_map.npy` as a legacy alias for the local
inpaint output to avoid breaking old scripts. New code and paper notes should
refer to `local_inpaint_repaired_belt_map.npy` and should treat it as a
prototype control.

## Preferred direction

For paper-facing GhostRepair claims, prefer `rebuild_masked` or a future
clean-observation/revolution rebuild that recomputes the affected belt
coordinates from raw observations of the same physical belt location. A repaired
map should preserve local mean, high-pass energy, texture statistics, and map
boundary continuity, not merely suppress the detector.

## Evidence policy

Cite a GhostRepair result only when all relevant checks are rerun for the exact
map being discussed:

- map-only false detections / false long tracks / false accepted tracks,
- texture-plausibility diagnostics for the repaired region,
- real-frame or reviewed-label metrics if making detection-quality claims.

Do not cite `local_inpaint` as an improved detector or final repair solely
because it reaches zero map-only ghosts.
