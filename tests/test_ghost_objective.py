import csv
import json
from pathlib import Path

import pytest

from beltmap.cli import ghost_objective as cli_ghost_objective
from beltmap.ghost_objective import (
    GhostObjectiveWeights,
    merge_evidence,
    score_variant,
    selected_variant_from_rows,
    select_winner,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_score_variant_uses_all_penalty_terms():
    score, missing = score_variant(
        {
            "variant": "candidate",
            "labeled_f1": 0.9,
            "fp_per_frame": 2.0,
            "map_only_false_detections": 3,
            "map_only_false_long_tracks": 1,
            "map_only_false_accepted_tracks": 0,
            "small_accepted_tracks": 5,
            "n_tracks": 10,
            "masked_pixel_fraction": 0.2,
        },
        GhostObjectiveWeights(),
    )

    assert missing == []
    assert score == pytest.approx(
        0.9 - 0.01 * 2.0 - 0.05 * 3 - 1.0 - 0.1 * 0.5 - 0.1 * 0.2
    )


def test_missing_map_only_terms_make_variant_ineligible():
    rows = merge_evidence(
        labeled={
            "candidate": {
                "variant": "candidate",
                "labeled_f1": 0.9,
                "fp_per_frame": 0.1,
            }
        },
        map_only={},
        weights=GhostObjectiveWeights(),
    )

    assert rows[0]["eligible_for_selection"] is False
    assert "map_only_false_detections" in rows[0]["missing_score_terms"]
    assert select_winner(rows) is None


def test_selected_variant_from_rows_uses_objective_winner_not_fixed_label():
    rows = [
        {
            "variant": "posmask_iter2",
            "eligible_for_selection": True,
            "ghost_objective_score": 0.5,
        },
        {
            "variant": "clean",
            "eligible_for_selection": True,
            "ghost_objective_score": 0.8,
        },
    ]

    assert selected_variant_from_rows(rows) == "clean"


def test_brick20g_regression_table_selects_posmask_iter2(tmp_path):
    summary_csv = tmp_path / "summary.csv"
    map_only_csv = tmp_path / "map_only.csv"
    output_dir = tmp_path / "objective"
    write_csv(
        summary_csv,
        [
            {
                "label": "current_pre_fix",
                "labeled_precision": 0.9719887955182073,
                "labeled_recall": 0.9830028328611898,
                "labeled_f1": 0.9774647887323944,
                "labeled_false_positives_per_frame": 0.60,
                "labeled_froc_auc_fp_per_frame_le_1": 0.9482058545797922,
                "small_accepted_tracks_lt_50": 0,
                "n_tracks": 579,
            },
            {
                "label": "posmask_iter2",
                "labeled_precision": 0.9818181818181818,
                "labeled_recall": 0.9943342776203966,
                "labeled_f1": 0.988036593947924,
                "labeled_false_positives_per_frame": 0.39,
                "labeled_froc_auc_fp_per_frame_le_1": 0.9628517469310671,
                "small_accepted_tracks_lt_50": 0,
                "n_tracks": 560,
            },
            {
                "label": "raw_robust_zscore",
                "labeled_precision": 0.3234458819816977,
                "labeled_recall": 0.9678942398489141,
                "labeled_f1": 0.4848628192999054,
                "labeled_false_positives_per_frame": 42.88,
                "labeled_froc_auc_fp_per_frame_le_1": 0.8079627006610008,
                "small_accepted_tracks_lt_50": 829,
                "n_tracks": 4172,
            },
        ],
    )
    write_csv(
        map_only_csv,
        [
            {
                "run": "current_pre_fix",
                "map_only_false_detections": 23,
                "map_only_false_long_tracks": 1,
                "map_only_false_accepted_tracks": 1,
            },
            {
                "run": "posmask_iter2",
                "map_only_false_detections": 0,
                "map_only_false_long_tracks": 0,
                "map_only_false_accepted_tracks": 0,
            },
        ],
    )

    assert (
        cli_ghost_objective.main(
            [
                "--summary-csv",
                str(summary_csv),
                "--map-only-summary-csv",
                str(map_only_csv),
                "--output-dir",
                str(output_dir),
                "--quiet",
            ]
        )
        == 0
    )

    selection = json.loads((output_dir / "config_selection.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((output_dir / "ghost_objective_table.csv").open(newline="", encoding="utf-8")))
    assert selection["selected_variant"] == "posmask_iter2"
    assert rows[0]["variant"] == "posmask_iter2"
    assert rows[1]["variant"] == "current_pre_fix"
    assert "not_a_beltmap_config" in rows[2]["missing_score_terms"]


def write_minimal_run(path: Path, *, false_positive: bool) -> None:
    path.mkdir(parents=True)
    (path / "metadata.json").write_text(
        json.dumps({"n_images": 1, "n_tracks": 1, "n_detections": 2 if false_positive else 1}),
        encoding="utf-8",
    )
    detections = [
        {
            "frame_index": 0,
            "label": 1,
            "y": 5,
            "x": 5,
            "area_px": 16,
            "peak_signal": 10,
            "bbox_top": 0,
            "bbox_left": 0,
            "bbox_bottom": 10,
            "bbox_right": 10,
        }
    ]
    if false_positive:
        detections.append(
            {
                "frame_index": 0,
                "label": 2,
                "y": 55,
                "x": 55,
                "area_px": 16,
                "peak_signal": 9,
                "bbox_top": 50,
                "bbox_left": 50,
                "bbox_bottom": 60,
                "bbox_right": 60,
            }
        )
    write_csv(path / "detections.csv", detections)
    write_csv(path / "detections_per_frame.csv", [{"frame_index": 0, "n_detections": len(detections)}])


def test_cli_scores_direct_truth_and_run_inputs(tmp_path):
    truth_path = tmp_path / "truth.json"
    current = tmp_path / "current"
    clean = tmp_path / "clean"
    map_only_csv = tmp_path / "map_only.csv"
    output_dir = tmp_path / "objective"
    truth_path.write_text(
        json.dumps(
            {
                "status": "reviewed_ground_truth",
                "scored_frames": [0],
                "particles": [
                    {
                        "frame_index": 0,
                        "top": 0,
                        "left": 0,
                        "bottom": 10,
                        "right": 10,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    write_minimal_run(current, false_positive=True)
    write_minimal_run(clean, false_positive=False)
    write_csv(
        map_only_csv,
        [
            {
                "run": "current",
                "map_only_false_detections": 4,
                "map_only_false_long_tracks": 1,
                "map_only_false_accepted_tracks": 1,
            },
            {
                "run": "clean",
                "map_only_false_detections": 0,
                "map_only_false_long_tracks": 0,
                "map_only_false_accepted_tracks": 0,
            },
        ],
    )

    assert (
        cli_ghost_objective.main(
            [
                "--truth-path",
                str(truth_path),
                "--run",
                f"current={current}",
                "--run",
                f"clean={clean}",
                "--map-only-summary-csv",
                str(map_only_csv),
                "--output-dir",
                str(output_dir),
                "--quiet",
            ]
        )
        == 0
    )

    selection = json.loads((output_dir / "config_selection.json").read_text(encoding="utf-8"))
    assert selection["selected_variant"] == "clean"
