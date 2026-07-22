from __future__ import annotations

import pytest

from beltmap import reused_period_state


@pytest.mark.parametrize(
    "metadata",
    [
        {"model_period_px": 120.0, "belt_period_known": False},
        {"model_period_px": 120.0, "belt_map_periodic": False},
        {"belt_model_period_px": 120.0, "model_period_px": None},
    ],
)
def test_reused_period_state_rejects_period_conflicting_with_finite_metadata(
    metadata: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="provides a physical belt period"):
        reused_period_state(
            map_height_px=120,
            supplied_period_px=None,
            metadata=metadata,
        )


def test_reused_period_state_still_accepts_consistent_periodic_metadata() -> None:
    state = reused_period_state(
        map_height_px=120,
        supplied_period_px=None,
        metadata={
            "belt_map_height_px": 120,
            "model_period_px": 120.0,
            "belt_period_known": True,
            "belt_map_periodic": True,
        },
    )

    assert state.model_period_px == 120.0
    assert state.period_known
    assert state.periodic
    assert state.source == "metadata"
