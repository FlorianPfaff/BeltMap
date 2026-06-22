# Script taxonomy

This directory is for lightweight helper scripts that are useful during development or reproducibility work but are **not** installed as supported command-line interfaces.

Supported CLIs live under `beltmap/cli/` and are registered in `[project.scripts]` in `pyproject.toml`. A command in `beltmap/cli/` should be treated as part of the supported user-facing surface: it needs a `--help` path, tests, stable inputs/outputs, and documentation.

Paper-specific or dataset-specific experiment drivers should live under `scripts/paper_experiments/`. They may encode concrete dataset names, frame ranges, paper evidence paths, or manuscript-specific decisions, but should not be imported by the package.

One-off debugging or archived development helpers should live under `scripts/dev_or_archive/`. These scripts are allowed to be less polished, but they should be clearly labeled as non-citable / non-supported.

## Current root-level helpers

The root of `scripts/` is kept only for generic helpers that do not encode a particular paper result:

- `apply_beltmap_to_images.py`
- `compare_raw_baselines.py`
- `filter_run_by_detection_overlap.py`
- `sweep_detection_overlap_filter.py`
- `check_script_taxonomy.py`

If a new script name contains a dataset, paper figure, one-off run ID, or local experiment label such as `brick20g`, `specificity50`, `yolo11`, `ghost_repair`, or `paper`, put it under `scripts/paper_experiments/` instead of the root directory.

Run the taxonomy check with:

```bash
python scripts/check_script_taxonomy.py
```
