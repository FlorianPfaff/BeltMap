from __future__ import annotations

import pytest

from beltmap._driver_motion import resolve_supplied_velocity


def test_supplied_source_frame_velocity_is_rescaled_by_stride(monkeypatch):
    monkeypatch.setenv("BELT_VELOCITY_FRAME_UNIT", "source_frame")

    velocity, frame_unit, raw_velocity = resolve_supplied_velocity(
        "2.5",
        frame_stride=4,
    )

    assert raw_velocity == 2.5
    assert frame_unit == "source_frame"
    assert velocity == 10.0


def test_supplied_selected_frame_velocity_is_not_rescaled(monkeypatch):
    monkeypatch.setenv("BELT_VELOCITY_FRAME_UNIT", "selected_frame")

    velocity, frame_unit, raw_velocity = resolve_supplied_velocity(
        "2.5",
        frame_stride=4,
    )

    assert raw_velocity == 2.5
    assert frame_unit == "selected_frame"
    assert velocity == 2.5


def test_supplied_velocity_defaults_to_selected_frame_when_unstrided(monkeypatch):
    monkeypatch.delenv("BELT_VELOCITY_FRAME_UNIT", raising=False)

    velocity, frame_unit, raw_velocity = resolve_supplied_velocity(
        "2.5",
        frame_stride=1,
    )

    assert raw_velocity == 2.5
    assert frame_unit == "selected_frame"
    assert velocity == 2.5


def test_supplied_velocity_requires_unit_when_frames_are_strided(monkeypatch):
    monkeypatch.delenv("BELT_VELOCITY_FRAME_UNIT", raising=False)

    with pytest.raises(ValueError, match="BELT_VELOCITY_FRAME_UNIT"):
        resolve_supplied_velocity("2.5", frame_stride=4)
