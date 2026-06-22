import pytest
import numpy as np

from beltmap import (
    BeltPeriodState,
    fresh_period_state,
    metadata_fields,
    phase_fraction_and_radians,
    require_period_known,
    reused_period_state,
)


def test_fresh_period_state_preserves_supplied_period():
    state = fresh_period_state(map_height_px=96, model_period_px=96.0)

    assert state == BeltPeriodState(
        map_height_px=96,
        model_period_px=96.0,
        period_known=True,
        source="supplied",
    )
    assert state.periodic


def test_fresh_period_state_marks_inferred_finite_strip():
    state = fresh_period_state(map_height_px=123, model_period_px=None)

    assert state.model_period_px is None
    assert not state.period_known
    assert not state.periodic
    assert state.source == "inferred_finite_strip"


def test_fresh_period_state_rejects_fractional_and_boolean_dimensions():
    with pytest.raises(ValueError, match="map_height_px must be a positive integer"):
        fresh_period_state(map_height_px=12.5, model_period_px=None)
    with pytest.raises(ValueError, match="map_height_px must be finite"):
        fresh_period_state(map_height_px=True, model_period_px=None)
    with pytest.raises(ValueError, match="map_height_px must be finite"):
        fresh_period_state(map_height_px=np.bool_(True), model_period_px=None)


def test_fresh_period_state_rejects_boolean_period():
    with pytest.raises(ValueError, match="model_period_px must be finite"):
        fresh_period_state(map_height_px=96, model_period_px=True)


def test_reused_period_state_trusts_explicit_metadata_period():
    state = reused_period_state(
        map_height_px=120,
        supplied_period_px=None,
        metadata={
            "belt_map_height_px": 120,
            "model_period_px": 120.0,
            "belt_period_known": True,
        },
    )

    assert state.model_period_px == 120.0
    assert state.period_known
    assert state.source == "metadata"


def test_reused_period_state_rejects_periodic_metadata_without_period():
    with pytest.raises(ValueError, match="declares a periodic belt map"):
        reused_period_state(
            map_height_px=120,
            supplied_period_px=None,
            metadata={"belt_period_known": True},
        )


def test_reused_period_state_rejects_periodic_map_metadata_without_period():
    with pytest.raises(ValueError, match="declares a periodic belt map"):
        reused_period_state(
            map_height_px=120,
            supplied_period_px=None,
            metadata={"belt_map_periodic": True},
        )


def test_reused_period_state_rejects_incompatible_metadata_height():
    with pytest.raises(ValueError, match="belt_map_height_px must match"):
        reused_period_state(
            map_height_px=120,
            supplied_period_px=None,
            metadata={"belt_map_height_px": 96},
        )


def test_reused_period_state_rejects_fractional_metadata_height():
    with pytest.raises(ValueError, match="positive integer"):
        reused_period_state(
            map_height_px=120,
            supplied_period_px=None,
            metadata={"belt_map_height_px": 120.5},
        )


def test_reused_period_state_rejects_incompatible_metadata_period():
    with pytest.raises(ValueError, match="must match reused belt-map height"):
        reused_period_state(
            map_height_px=120,
            supplied_period_px=None,
            metadata={"model_period_px": 96.0, "belt_period_known": True},
        )


def test_reused_period_state_preserves_metadata_finite_strip():
    state = reused_period_state(
        map_height_px=120,
        supplied_period_px=120,
        metadata={"model_period_px": None, "belt_period_known": False},
    )

    assert state.model_period_px is None
    assert not state.period_known
    assert state.source == "metadata_finite_strip"


def test_reused_period_state_only_trusts_matching_supplied_period_for_legacy_map():
    matching = reused_period_state(
        map_height_px=120,
        supplied_period_px=120,
        metadata={},
    )
    mismatching = reused_period_state(
        map_height_px=120,
        supplied_period_px=96,
        metadata={},
    )

    assert matching.model_period_px == 120.0
    assert matching.source == "supplied_matching_reuse_height"
    assert mismatching.model_period_px is None
    assert mismatching.source == "legacy_reuse_unknown"


def test_phase_fraction_and_radians_are_empty_without_physical_period():
    assert phase_fraction_and_radians(12.0, None) == ("", "")

    fraction, radians = phase_fraction_and_radians(12.0, 48.0)

    assert fraction == pytest.approx(0.25)
    assert radians == pytest.approx(0.5 * 3.141592653589793)


def test_phase_fraction_and_radians_rejects_nonfinite_phase():
    with pytest.raises(ValueError, match="phase_px must be finite"):
        phase_fraction_and_radians(float("nan"), 48.0)


def test_require_period_known_rejects_finite_strip():
    state = fresh_period_state(map_height_px=120, model_period_px=None)

    with pytest.raises(ValueError, match="requires a known physical BELT_PERIOD_PX"):
        require_period_known(state, feature="recurrent artifact filtering")


def test_metadata_fields_round_trip_period_state():
    state = fresh_period_state(map_height_px=120, model_period_px=None)

    fields = metadata_fields(state)

    assert fields == {
        "belt_map_height_px": 120,
        "model_period_px": None,
        "belt_period_known": False,
        "belt_map_periodic": False,
        "belt_period_state_source": "inferred_finite_strip",
    }
