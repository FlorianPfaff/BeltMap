from __future__ import annotations

import json
from pathlib import Path

from beltmap.cli.ghost_repair import (
    LOCAL_INPAINT_WARNING,
    REPAIR_MODE_POLICY,
    annotate_mode_row,
    write_mode_policy,
)



def test_local_inpaint_policy_marks_prototype_control() -> None:
    policy = REPAIR_MODE_POLICY["local_inpaint"]

    assert policy["repair_role"] == "prototype_control"
    assert "not_final_repair" in policy["paper_status"]
    assert "not_texture_preserving" in policy["texture_plausibility_status"]
    assert "prototype/control" in LOCAL_INPAINT_WARNING



def test_summary_rows_are_annotated_with_mode_policy() -> None:
    row = annotate_mode_row(
        {
            "map_variant": "local_inpaint",
            "belt_map_path": "repaired_belt_map.npy",
            "map_only_false_detections": 0,
        },
        "local_inpaint",
    )

    assert row["repair_role"] == "prototype_control"
    assert row["paper_status"] == "diagnostic_only_not_final_repair"
    assert row["texture_plausibility_status"] == "not_texture_preserving_without_extra_checks"



def test_write_mode_policy_outputs_warning_files(tmp_path: Path) -> None:
    write_mode_policy(
        tmp_path,
        local_repaired_path=tmp_path / "local_inpaint_repaired_belt_map.npy",
        legacy_repaired_alias_path=tmp_path / "repaired_belt_map.npy",
        rebuild_mask_path=tmp_path / "ghost_defect_mask.npy",
        rebuild_map_path=None,
    )

    policy = json.loads((tmp_path / "ghost_repair_mode_policy.json").read_text(encoding="utf-8"))
    assert policy["modes"]["local_inpaint"]["repair_role"] == "prototype_control"
    assert policy["modes"]["rebuild_masked"]["repair_role"] == "preferred_raw_frame_rebuild"
    assert "Map-only 0/0/0" in " ".join(policy["paper_guidance"])

    readme = (tmp_path / "local_inpaint_PROTOTYPE_README.md").read_text(encoding="utf-8")
    assert "legacy alias" in readme
    assert "not cite" in readme.lower()
