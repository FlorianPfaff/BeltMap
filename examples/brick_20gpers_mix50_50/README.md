# Brick 20 g/s 50:50 Zenodo example

This example targets Zenodo record `7801882`, the high-flow 20 g/s brick/sand-lime brick sequence with a 50:50 mixture ratio.

The dataset is not stored in this repository. Prepare it with the reusable Zenodo helper:

```bash
python -m pip install -e ".[test]"
beltmap-prepare-zenodo \
  --record-id 7801882 \
  --record-file-glob 'images_*.zip' \
  --dataset-name images_BrickandSandLimeBrick_50vs50_20gpers \
  --image-link data/images \
  --zip-link data/images_BrickandSandLimeBrick_50vs50_20gpers.zip \
  --manifest-path outputs/dataset_manifest.json
```

Then run:

```bash
beltmap-apply --config examples/brick_20gpers_mix50_50/beltmap.toml
beltmap-validate --output-dir outputs
```

The configuration uses stricter detection defaults than the Brick 10 g/s example. A 20-frame smoke run on this dataset over-detected badly with the lower 10 g/s threshold, while `threshold = 12.0`, `low_threshold = 0.0`, and `min_area_px = 50` produced a more conservative PyRecEst-tracked run. This removes tiny speckles while preserving the long-track count seen with `min_area_px = 8`; `min_area_px = 100` was also tractable but dropped more velocity tracks. Treat these as starting defaults, then tune only after recording raw, static-average, and BeltMap baseline evidence.
