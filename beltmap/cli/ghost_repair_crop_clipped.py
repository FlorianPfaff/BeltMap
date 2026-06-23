from __future__ import annotations

# Import for side effect before importing the original CLI: make GhostRepair
# defect-map projection clip bbox margins to the visible crop height.
import beltmap.ghost_repair_crop_clip_patch  # noqa: F401
from beltmap.cli.ghost_repair import main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
