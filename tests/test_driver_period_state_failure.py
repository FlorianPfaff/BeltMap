from __future__ import annotations

import json

import pytest

from beltmap import driver_period_state_patch as patch


def test_failed_driver_does_not_rewrite_existing_metadata(tmp_path, monkeypatch):
    metadata = {
        "belt_map_height_px": 120,
        "model_period_px": None,
        "belt_period_known": False,
        "belt_map_periodic": False,
        "belt_period_state_source": "previous-successful-run",
    }
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    monkeypatch.setenv("BELTMAP_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("BELT_PERIOD_PX", "120")
    monkeypatch.delenv("REUSE_BELT_MAP_PATH", raising=False)
    monkeypatch.delenv("REUSE_RECURRENT_ARTIFACT_MAP_PATH", raising=False)
    monkeypatch.delenv("RECURRENT_ARTIFACT_MIN_REVOLUTIONS", raising=False)

    def fail_driver(*args, **kwargs):
        raise RuntimeError("synthetic driver failure")

    monkeypatch.setattr(patch, "_original_driver_main", fail_driver)
    previous_period = patch._DRIVER_MODEL_PERIOD_PX[0]
    patch._DRIVER_MODEL_PERIOD_PX[0] = 37.0
    try:
        with pytest.raises(RuntimeError, match="synthetic driver failure"):
            patch._patched_main()
        assert patch._DRIVER_MODEL_PERIOD_PX[0] == 37.0
    finally:
        patch._DRIVER_MODEL_PERIOD_PX[0] = previous_period

    assert json.loads(metadata_path.read_text(encoding="utf-8")) == metadata
