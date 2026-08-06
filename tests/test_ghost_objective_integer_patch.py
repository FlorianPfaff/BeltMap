from __future__ import annotations

import importlib

import pytest

import beltmap  # noqa: F401 - imports side-effect patches
import beltmap.ghost_objective as ghost_objective
import beltmap.ghost_objective_integer_patch as integer_patch


def test_ghost_objective_integer_patch_is_autoloaded() -> None:
    assert getattr(
        ghost_objective.finite_integer,
        "_beltmap_exact_ghost_objective_integer_patched",
        False,
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        (2, 2),
        (3.0, 3),
        ("4", 4),
        ("5.0", 5),
        ("6e0", 6),
        (0.6, None),
        ("1.5", None),
        (-1, None),
        ("-2.0", None),
        (True, None),
        (False, None),
        ("nan", None),
        (None, None),
    ],
)
def test_finite_integer_accepts_only_exact_nonnegative_counts(
    value: object,
    expected: int | None,
) -> None:
    assert ghost_objective.finite_integer(value) == expected


def test_fractional_map_only_count_makes_variant_ineligible() -> None:
    map_only_row = ghost_objective.map_only_evidence_from_row(
        "candidate",
        {
            "map_only_false_detections": "0.6",
            "map_only_false_long_tracks": "0",
            "map_only_false_accepted_tracks": "0",
        },
        source="map-only.csv",
    )

    rows = ghost_objective.merge_evidence(
        labeled={
            "candidate": {
                "variant": "candidate",
                "labeled_f1": 0.9,
                "fp_per_frame": 0.1,
            }
        },
        map_only={"candidate": map_only_row},
        weights=ghost_objective.GhostObjectiveWeights(),
    )

    assert map_only_row["map_only_false_detections"] is None
    assert rows[0]["eligible_for_selection"] is False
    assert "map_only_false_detections" in rows[0]["missing_score_terms"]


def test_integer_patch_reload_is_idempotent() -> None:
    importlib.reload(integer_patch)
    importlib.reload(integer_patch)

    assert ghost_objective.finite_integer("2.0") == 2
    assert ghost_objective.finite_integer("2.1") is None
