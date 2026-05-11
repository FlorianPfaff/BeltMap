import json

import pytest

from beltmap.cli import apply as cli


def parse_args(*args: str):
    return cli.build_parser().parse_args(list(args))


def test_resolve_driver_env_applies_config_environment_and_cli_precedence(tmp_path):
    config_path = tmp_path / "beltmap.toml"
    config_path.write_text(
        """
[paths]
image_dir = "config-images"
output_dir = "config-outputs"

[belt]
velocity_px_per_frame = 1.5

[detection]
threshold = 3.25
""".strip(),
        encoding="utf-8",
    )
    environ = {
        "BELT_VELOCITY_PX_PER_FRAME": "2.5",
        "DETECTION_THRESHOLD": "4.0",
        "MAX_FRAMES": "20",
    }
    args = parse_args(
        "--config",
        str(config_path),
        "--belt-velocity-px-per-frame",
        "7.5",
        "--belt-region",
        "1, 2, 3, 4",
        "--allow-full-frame-auto-velocity",
    )

    env_updates, report = cli.resolve_driver_env(args, environ=environ)

    assert env_updates == {
        "ALLOW_FULL_FRAME_AUTO_VELOCITY": "1",
        "BELTMAP_IMAGE_DIR": "config-images",
        "BELTMAP_OUTPUT_DIR": "config-outputs",
        "BELT_REGION": "1,2,3,4",
        "BELT_VELOCITY_PX_PER_FRAME": "7.5",
        "DETECTION_THRESHOLD": "4",
        "MAX_FRAMES": "20",
    }
    assert report["precedence"] == ["config", "environment", "cli"]
    assert report["options"]["image_dir"]["source"].startswith("config:")
    assert report["options"]["detection_threshold"]["source"] == "env:DETECTION_THRESHOLD"
    assert report["options"]["belt_velocity_px_per_frame"]["source"] == "cli"
    assert report["options"]["allow_full_frame_auto_velocity"]["value"] == "1"


def test_sectioned_toml_config_is_resolved_to_driver_environment(tmp_path):
    config_path = tmp_path / "sectioned.toml"
    config_path.write_text(
        """
[paths]
image_dir = "data/images"
output_dir = "outputs"

[reuse]
belt_map_path = "previous/belt_map.npy"
phase_estimates_path = "previous/phase_estimates.csv"

[frames]
stride = 3

[belt]
region = [10, 20, 30, 40]
velocity_px_per_frame = "auto"
period_px = 64

[detection]
threshold = 5.5
min_area_px = 4

[map]
particle_mask_mode = "hysteresis_abs"
particle_mask_threshold = 4.0
particle_mask_grow_threshold = 1.5
particle_mask_dilation_px = 24
particle_mask_margin_px = 16
particle_mask_min_area_px = 8

[auto_velocity]
allow_full_frame = true
""".strip(),
        encoding="utf-8",
    )

    env_updates, report = cli.resolve_driver_env(
        parse_args("--config", str(config_path)),
        environ={},
    )

    assert env_updates == {
        "ALLOW_FULL_FRAME_AUTO_VELOCITY": "1",
        "BELTMAP_IMAGE_DIR": "data/images",
        "BELTMAP_OUTPUT_DIR": "outputs",
        "BELT_PERIOD_PX": "64",
        "BELT_REGION": "10,20,30,40",
        "BELT_VELOCITY_PX_PER_FRAME": "auto",
        "DETECTION_THRESHOLD": "5.5",
        "FRAME_STRIDE": "3",
        "MAP_PARTICLE_MASK_DILATION_PX": "24",
        "MAP_PARTICLE_MASK_GROW_THRESHOLD": "1.5",
        "MAP_PARTICLE_MASK_MARGIN_PX": "16",
        "MAP_PARTICLE_MASK_MIN_AREA_PX": "8",
        "MAP_PARTICLE_MASK_MODE": "hysteresis_abs",
        "MAP_PARTICLE_MASK_THRESHOLD": "4",
        "MIN_AREA_PX": "4",
        "REUSE_BELT_MAP_PATH": "previous/belt_map.npy",
        "REUSE_PHASE_ESTIMATES_PATH": "previous/phase_estimates.csv",
    }
    assert report["driver_environment"] == env_updates


def test_flat_json_config_is_resolved_to_driver_environment(tmp_path):
    config_path = tmp_path / "flat.json"
    config_path.write_text(
        json.dumps(
            {
                "image_dir": "images-json",
                "output_dir": "outputs-json",
                "belt_region": [5, 6, 7, 8],
                "belt_velocity_px_per_frame": 12.25,
                "detection_threshold": 3,
                "allow_full_frame_auto_velocity": False,
            }
        ),
        encoding="utf-8",
    )

    env_updates, _report = cli.resolve_driver_env(
        parse_args("--config", str(config_path)),
        environ={},
    )

    assert env_updates == {
        "ALLOW_FULL_FRAME_AUTO_VELOCITY": "0",
        "BELTMAP_IMAGE_DIR": "images-json",
        "BELTMAP_OUTPUT_DIR": "outputs-json",
        "BELT_REGION": "5,6,7,8",
        "BELT_VELOCITY_PX_PER_FRAME": "12.25",
        "DETECTION_THRESHOLD": "3",
    }


def test_unknown_config_option_raises_useful_error(tmp_path):
    config_path = tmp_path / "unknown.toml"
    config_path.write_text("[belt]\nspeed = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Unknown config option 'belt\.speed'"):
        cli.values_from_config(config_path)


def test_duplicate_config_aliases_raise_useful_error(tmp_path):
    config_path = tmp_path / "duplicate.toml"
    config_path.write_text(
        """
image_dir = "flat-images"

[paths]
image_dir = "section-images"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"Config option 'image_dir' was specified more than once",
    ):
        cli.values_from_config(config_path)


def test_empty_environment_values_are_ignored():
    args = parse_args()
    environ = {
        "BELT_PERIOD_PX": "",
        "MAX_MATCH_DISTANCE_PX": "   ",
        "BELTMAP_IMAGE_DIR": "data/images",
    }

    env_updates, report = cli.resolve_driver_env(args, environ=environ)

    assert env_updates == {"BELTMAP_IMAGE_DIR": "data/images"}
    assert set(report["options"]) == {"image_dir"}


def test_normalize_value_accepts_expected_shapes_and_rejects_bad_values():
    assert cli.normalize_value("belt_region", [1, 2, 3, 4]) == "1,2,3,4"
    assert cli.normalize_value("belt_region", "1, 2, 3, 4") == "1,2,3,4"
    assert cli.normalize_value("belt_velocity_px_per_frame", "auto") == "auto"
    assert cli.normalize_value("belt_velocity_px_per_frame", 2.0) == "2"
    assert cli.normalize_value("map_particle_mask_mode", "hysteresis_abs") == "hysteresis_abs"
    assert cli.normalize_value("allow_full_frame_auto_velocity", "yes") == "1"
    assert cli.normalize_value("allow_full_frame_auto_velocity", "no") == "0"

    with pytest.raises(ValueError, match="belt_region must contain exactly four values"):
        cli.normalize_value("belt_region", [1, 2, 3])
    with pytest.raises(ValueError, match="detection_threshold must be a finite number"):
        cli.normalize_value("detection_threshold", "nan")
    with pytest.raises(ValueError, match="min_area_px must be an integer"):
        cli.normalize_value("min_area_px", 1.5)


def test_write_config_template_writes_valid_toml(tmp_path):
    config_path = tmp_path / "nested" / "beltmap.toml"

    cli.write_config_template(config_path)

    parsed = cli.load_config_file(config_path)
    assert parsed["paths"]["image_dir"] == "data/images"
    assert parsed["paths"]["output_dir"] == "outputs"
    assert parsed["belt"]["region"] == [0, 220, 1330, 1800]
    assert parsed["belt"]["velocity_px_per_frame"] == "auto"
    assert parsed["map"]["particle_mask_mode"] == "positive"
    assert parsed["map"]["particle_mask_grow_threshold"] == 2.0
    assert parsed["map"]["particle_mask_dilation_px"] == 0


def test_main_write_config_template_exits_without_running_driver(tmp_path, monkeypatch):
    config_path = tmp_path / "beltmap.toml"

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("run_driver should not be called")

    monkeypatch.setattr(cli, "run_driver", fail_if_called)

    assert cli.main(["--write-config-template", str(config_path)]) == 0
    assert config_path.exists()


def test_dry_run_prints_report_and_does_not_run_driver(monkeypatch, capsys):
    for _name, env_var, *_rest in cli.OPTION_SPECS:
        monkeypatch.delenv(env_var, raising=False)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("run_driver should not be called during --dry-run")

    monkeypatch.setattr(cli, "run_driver", fail_if_called)

    exit_code = cli.main(
        [
            "--dry-run",
            "--image-dir",
            "data/images",
            "--belt-region",
            "1,2,3,4",
            "--map-particle-mask-mode",
            "hysteresis_abs",
            "--map-particle-mask-grow-threshold",
            "1.5",
            "--map-particle-mask-dilation-px",
            "24",
            "--allow-full-frame-auto-velocity",
        ]
    )

    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["driver_environment"] == {
        "ALLOW_FULL_FRAME_AUTO_VELOCITY": "1",
        "BELTMAP_IMAGE_DIR": "data/images",
        "BELT_REGION": "1,2,3,4",
        "MAP_PARTICLE_MASK_DILATION_PX": "24",
        "MAP_PARTICLE_MASK_GROW_THRESHOLD": "1.5",
        "MAP_PARTICLE_MASK_MODE": "hysteresis_abs",
    }
    assert report["options"]["image_dir"]["source"] == "cli"


def test_main_reports_invalid_config_as_argparse_error(tmp_path, capsys):
    config_path = tmp_path / "invalid.toml"
    config_path.write_text("[belt]\nspeed = 1\n", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--dry-run", "--config", str(config_path)])

    assert excinfo.value.code == 2
    assert "Unknown config option 'belt.speed'" in capsys.readouterr().err
