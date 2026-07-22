import importlib

from beltmap import _driver_map
from beltmap import driver_period_state_patch as patch


def test_driver_period_state_patch_reload_keeps_true_originals(monkeypatch):
    original_callables = {
        "build_belt_map_result": patch._original_build_belt_map_result,
        "accumulate_belt_map": patch._original_accumulate_belt_map,
        "belt_motion_model": patch._original_belt_motion_model,
        "phase_drift_filter": patch._original_phase_drift_filter,
        "phase_estimate_row": patch._original_phase_estimate_row,
        "texture_phase_velocity_summary": (
            patch._original_texture_phase_velocity_summary
        ),
        "score_recurrent_artifact_detections": (
            patch._original_score_recurrent_artifact_detections
        ),
        "driver_main": patch._original_driver_main,
    }

    reloaded = importlib.reload(patch)
    reloaded = importlib.reload(reloaded)

    assert reloaded._original_build_belt_map_result is original_callables[
        "build_belt_map_result"
    ]
    assert reloaded._original_accumulate_belt_map is original_callables[
        "accumulate_belt_map"
    ]
    assert reloaded._original_belt_motion_model is original_callables[
        "belt_motion_model"
    ]
    assert reloaded._original_phase_drift_filter is original_callables[
        "phase_drift_filter"
    ]
    assert reloaded._original_phase_estimate_row is original_callables[
        "phase_estimate_row"
    ]
    assert reloaded._original_texture_phase_velocity_summary is original_callables[
        "texture_phase_velocity_summary"
    ]
    assert (
        reloaded._original_score_recurrent_artifact_detections
        is original_callables["score_recurrent_artifact_detections"]
    )
    assert reloaded._original_driver_main is original_callables["driver_main"]

    calls = []

    def fake_accumulate_belt_map(*args, **kwargs):
        calls.append((args, kwargs))
        return "accumulated"

    monkeypatch.setattr(
        reloaded,
        "_original_accumulate_belt_map",
        fake_accumulate_belt_map,
    )

    result = _driver_map.accumulate_belt_map(model_period=None)

    assert result == "accumulated"
    assert calls == [((), {"model_period": None})]
