from __future__ import annotations

from beltmap.cli import apply


def test_config_passes_velocity_frame_unit_to_driver_environment(tmp_path):
    config_path = tmp_path / "beltmap.toml"
    config_path.write_text(
        "[belt]\n"
        "velocity_px_per_frame = 2.5\n"
        "velocity_frame_unit = \"source_frame\"\n",
        encoding="utf-8",
    )
    parser = apply.build_parser()
    args = parser.parse_args(["--config", str(config_path), "--dry-run"])

    env_updates, report = apply.resolve_driver_env(args, environ={})

    assert env_updates["BELT_VELOCITY_PX_PER_FRAME"] == "2.5"
    assert env_updates["BELT_VELOCITY_FRAME_UNIT"] == "source_frame"
    assert report["options"]["belt_velocity_frame_unit"]["value"] == "source_frame"
