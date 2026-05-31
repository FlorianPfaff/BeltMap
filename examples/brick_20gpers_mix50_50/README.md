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

The configuration intentionally starts from the existing Brick 10 g/s BeltMap settings. The dataset is meant as a same-domain, higher-density holdout; tune thresholds only after recording raw, static-average, and BeltMap baseline evidence.
