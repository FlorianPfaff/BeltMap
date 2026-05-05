# BeltMap

Tools for reconstructing conveyor-belt background maps and using them to improve particle localization.

The first implemented piece is belt phase estimation:

- predict the belt phase from a signed constant-speed model
- render the expected clean belt crop for a frame
- refine the predicted phase by robustly registering the observed frame against the belt map

Datasets are intentionally not stored in this repository.
