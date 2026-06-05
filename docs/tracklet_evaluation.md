# Sparse short-tracklet evaluation

Use this workflow when a small real-data validation subset has manual particle
identities over short time windows. It complements frame-wise labeled detection
metrics by measuring whether the PyRecEst-backed tracker keeps the same particle
identity over consecutive frames and whether ghost detections become coherent
false tracks.

## Annotation CSV

The minimal CSV schema is:

```csv
tracklet_id,frame_index,bbox_top,bbox_left,bbox_bottom,bbox_right
p01,120,42,510,58,529
p01,121,45,511,61,530
p02,121,90,714,105,731
```

Coordinates are crop-local and use the same half-open bounding-box convention as
`detections.csv` and `tracks.csv`.

A row with only `frame_index` marks an explicitly scored empty frame. Predictions
on such frames count as false positives, which makes this useful for ghost-track
stress testing:

```csv
tracklet_id,frame_index,bbox_top,bbox_left,bbox_bottom,bbox_right
,248,,,,
```

Accepted identifier aliases are `tracklet_id`, `particle_id`, `event_id`,
`track_id`, and `id`. Accepted box aliases are `bbox_top`/`bbox_left`/
`bbox_bottom`/`bbox_right`, `top`/`left`/`bottom`/`right`,
`y_min`/`x_min`/`y_max`/`x_max`, and `y1`/`x1`/`y2`/`x2`.

## Annotation JSON

The evaluator also accepts flat JSON rows:

```json
{
  "scored_frames": [120, 121, 122],
  "tracklets": [
    {
      "tracklet_id": "p01",
      "boxes": [
        {"frame_index": 120, "top": 42, "left": 510, "bottom": 58, "right": 529},
        {"frame_index": 121, "top": 45, "left": 511, "bottom": 61, "right": 530}
      ]
    }
  ]
}
```

## Running the evaluator

By default, the command scores `filtered_tracks.csv` when it exists and otherwise
falls back to `tracks.csv`:

```bash
beltmap-evaluate-tracklets \
  --output-dir outputs/T12_area50 \
  --truth-path labels/brick_short_tracklets.csv \
  --iou-threshold 0.25
```

To compare raw and filtered PyRecEst trajectories explicitly, run the command
twice with different `--prediction-path` values:

```bash
beltmap-evaluate-tracklets \
  --output-dir outputs/T12_area50 \
  --truth-path labels/brick_short_tracklets.csv \
  --prediction-path outputs/T12_area50/tracks.csv \
  --metrics-path outputs/T12_area50/raw_tracklet_metrics.json \
  --report-path outputs/T12_area50/raw_tracklet_report.md \
  --matches-path outputs/T12_area50/raw_tracklet_matches.csv
```

The command writes:

```text
outputs/
  tracklet_metrics.json
  tracklet_report.md
  tracklet_matches.csv
```

## Metrics

The JSON summary includes frame-level detection precision/recall/F1 and a local
HOTA-style decomposition:

- `det_a`: `TP / (TP + FP + FN)` over manually scored frames.
- `ass_a`: average association accuracy for matched truth/predicted identity
  pairs. It penalizes identity switches, split truth tracklets, merged predicted
  tracks, missed truth boxes, and false positive predicted boxes.
- `loc_a`: mean IoU of matched boxes.
- `hota`: `sqrt(det_a * ass_a)`.

These values are designed for sparse real labels and are not byte-for-byte
TrackEval output. They are intentionally lightweight so the metric can run in CI
without adding a heavy dependency. Use `tracklet_matches.csv` to audit every
true positive, false positive, and false negative assignment.
