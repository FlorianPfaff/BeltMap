from __future__ import annotations

import numpy as np

import beltmap  # noqa: F401 - imports compatibility patches
from beltmap import ghost_objective


def test_ghost_objective_count_parser_rejects_fractional_and_boolean_values() -> None:
    assert ghost_objective.finite_integer("2") == 2
    assert ghost_objective.finite_integer(2.0) == 2
    assert ghost_objective.finite_integer("0.6") is None
    assert ghost_objective.finite_integer(True) is None
    assert ghost_objective.finite_integer(np.bool_(False)) is None


def test_fractional_map_only_count_makes_candidate_ineligible() -> None:
    row = {
        "variant": "candidate",
        "labeled_f1": 0.9,
        "fp_per_frame": 0.1,
        "map_only_false_detections": ghost_objective.finite_integer("0.6"),
        "map_only_false_long_tracks": ghost_objective.finite_integer("0"),
        "map_only_false_accepted_tracks": ghost_objective.finite_integer("0"),
    }

    score, missing = ghost_objective.score_variant(
        row,
        ghost_objective.GhostObjectiveWeights(),
    )

    assert score is None
    assert missing == ["map_only_false_detections"]
