# Brick 20 g/s 50:50 Zenodo example

This example targets Zenodo record `7801882`, the high-flow 20 g/s brick/sand-lime brick sequence with a 50:50 mixture ratio.

The dataset is not stored in this repository. Use the companion GitHub Actions workflow, or prepare the data manually so the extracted image directory is available as:

```text
data/images/
```

Then run:

```bash
python -m pip install -e ".[test]"
beltmap-apply --config examples/brick_20gpers_mix50_50/beltmap.toml
beltmap-validate --output-dir outputs
```

The configuration intentionally starts from the existing Brick 10 g/s BeltMap settings. The dataset is meant as a same-domain, higher-density holdout; tune thresholds only after recording raw, static-average, and BeltMap baseline evidence.
