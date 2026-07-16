import json

import pytest

from beltmap import map_only_negative_control as map_only
from beltmap import map_only_period_state_patch as patch


def test_map_only_period_state_guard_is_autoloaded():
    assert (
        map_only.generate_map_only_negative_control_report
        is patch.period_safe_generate_map_only_negative_control_report
    )


def test_map_only_negative_control_rejects_inferred_finite_strip(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "belt_map_height_px": 40,
                "model_period_px": None,
                "belt_period_known": False,
                "belt_map_periodic": False,
                "belt_period_state_source": "inferred_finite_strip",
            }
        ),
        encoding="utf-8",
    )
    called = False

    def fake_generate(**kwargs):
        nonlocal called
        called = True
        return kwargs

    monkeypatch.setattr(
        patch,
        "_original_generate_map_only_negative_control_report",
        fake_generate,
    )

    with pytest.raises(ValueError, match="known physical BELT_PERIOD_PX"):
        map_only.generate_map_only_negative_control_report(output_dir=output_dir)

    assert not called


def test_map_only_negative_control_preserves_periodic_and_legacy_runs(
    tmp_path,
    monkeypatch,
):
    periodic_dir = tmp_path / "periodic"
    periodic_dir.mkdir()
    (periodic_dir / "metadata.json").write_text(
        json.dumps(
            {
                "belt_map_height_px": 40,
                "model_period_px": 40.0,
                "belt_period_known": True,
                "belt_map_periodic": True,
            }
        ),
        encoding="utf-8",
    )
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return kwargs["output_dir"]

    monkeypatch.setattr(
        patch,
        "_original_generate_map_only_negative_control_report",
        fake_generate,
    )

    assert (
        map_only.generate_map_only_negative_control_report(output_dir=periodic_dir)
        == periodic_dir
    )
    assert (
        map_only.generate_map_only_negative_control_report(output_dir=legacy_dir)
        == legacy_dir
    )
    assert [call["output_dir"] for call in calls] == [periodic_dir, legacy_dir]


def test_custom_belt_map_uses_its_sibling_metadata(tmp_path, monkeypatch):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    (output_dir / "metadata.json").write_text(
        json.dumps({"belt_map_periodic": True, "model_period_px": 40.0}),
        encoding="utf-8",
    )
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    custom_map = custom_dir / "belt_map.npy"
    (custom_dir / "metadata.json").write_text(
        json.dumps({"belt_map_periodic": False, "model_period_px": None}),
        encoding="utf-8",
    )
    called = False

    def fake_generate(**kwargs):
        nonlocal called
        called = True
        return kwargs

    monkeypatch.setattr(
        patch,
        "_original_generate_map_only_negative_control_report",
        fake_generate,
    )

    with pytest.raises(ValueError, match="opposite map boundary"):
        map_only.generate_map_only_negative_control_report(
            output_dir=output_dir,
            belt_map_path=custom_map,
        )

    assert not called
