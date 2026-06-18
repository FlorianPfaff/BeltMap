from __future__ import annotations

import pytest

from scripts import apply_beltmap_to_images as legacy_apply


@pytest.mark.parametrize("value", ["nan", "inf", "+inf", "-inf"])
def test_legacy_env_float_rejects_nonfinite_values(monkeypatch, value):
    monkeypatch.setenv("LEGACY_FLOAT", value)

    with pytest.raises(ValueError, match="LEGACY_FLOAT must be finite"):
        legacy_apply.env_float("LEGACY_FLOAT", 1.0)


def test_legacy_env_optional_float_rejects_nonfinite_values(monkeypatch):
    monkeypatch.setenv("LEGACY_OPTIONAL_FLOAT", "nan")

    with pytest.raises(ValueError, match="LEGACY_OPTIONAL_FLOAT must be finite"):
        legacy_apply.env_optional_float("LEGACY_OPTIONAL_FLOAT")


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_legacy_resolve_supplied_velocity_rejects_nonfinite_values(monkeypatch, value):
    monkeypatch.delenv("BELT_VELOCITY_FRAME_UNIT", raising=False)

    with pytest.raises(ValueError, match="BELT_VELOCITY_PX_PER_FRAME must be finite"):
        legacy_apply.resolve_supplied_velocity(value, frame_stride=1)
