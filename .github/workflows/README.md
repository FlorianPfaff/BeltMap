# Workflow organization

This repository keeps GitHub Actions in three practical classes:

1. **Core CI**
   - `ci.yml`
   - `mega-linter.yml`
   - `smoke-beltmap-driver.yml`

   These workflows are safe to run automatically and should stay lightweight.

2. **Experiment smoke tests**
   - `experiment-smoke.yml`

   This workflow replaces the temporary file-triggered YOLO autosmokes. It runs focused, data-free tests for experiment utilities such as YOLO prediction export and recurrence scoring. Add future lightweight experiment utility checks here instead of adding new one-off trigger workflows.

3. **Manual heavy experiments**
   - `yolo11-beltmap-pilot.yml`
   - `compare-beltmap-brick-configs.yml`
   - `compare-brick-raw-baselines.yml`
   - `apply-beltmap-brick-10gpers.yml`

   These workflows may download large Zenodo archives, train models, run long BeltMap jobs, or depend on self-hosted runners. Keep them `workflow_dispatch`-only unless there is a deliberate reason to spend compute on every push.

Temporary trigger files under `.github/triggers/` were used only to force early GitHub Actions smoke runs after adding new workflows. They should not be used as a permanent workflow trigger mechanism.
