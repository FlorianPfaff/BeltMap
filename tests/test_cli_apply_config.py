import pytest

from beltmap.cli.apply import build_parser, resolve_driver_env, write_config_template


def parse_args(arguments):
    return build_parser().parse_args(arguments)


def test_nested_toml_config_maps_to_driver_environment(tmp_path):
    config = tmp_path / "beltmap.toml"
    config.write_text(
        """
        [paths]
        image_dir = "images"
        output_dir = "out"

        [belt]
        region = [1, 2, 30, 40]
        velocity_px_per_frame = 59.3
        period_px = 14723

        [detection]
        threshold = 4.5
        min_area_px = 6

        [residual]
        noise_exclusion_sigma = 3.5
        noise_exclusion_radius_px = 1

        [map]
        sampling_strategy = "adaptive_phase_coverage"

        [auto_velocity]
        allow_full_frame = true
        """,
        encoding="utf-8",
    )

    env, report = resolve_driver_env(parse_args(["--config", str(config)]), environ={})

    assert env["BELTMAP_IMAGE_DIR"] == "images"
    assert env["BELTMAP_OUTPUT_DIR"] == "out"
    assert env["BELT_REGION"] == "1,2,30,40"
    assert env["BELT_VELOCITY_PX_PER_FRAME"] == "59.3"
    assert env["BELT_PERIOD_PX"] == "14723"
    assert env["DETECTION_THRESHOLD"] == "4.5"
    assert env["MIN_AREA_PX"] == "6"
    assert env["RESIDUAL_NOISE_EXCLUSION_SIGMA"] == "3.5"
    assert env["RESIDUAL_NOISE_EXCLUSION_RADIUS_PX"] == "1"
    assert env["MAP_SAMPLING_STRATEGY"] == "adaptive_phase_coverage"
    assert env["ALLOW_FULL_FRAME_AUTO_VELOCITY"] == "1"
    assert report["options"]["belt_region"]["source"].startswith("config:")


def test_static_residual_config_accepts_auto_sample_frames(tmp_path):
    config = tmp_path / "beltmap.toml"
    config.write_text(
        """
        [static_noise]
        sample_frames = "auto"

        [static_background]
        sample_frames = "auto"
        """,
        encoding="utf-8",
    )

    env, report = resolve_driver_env(parse_args(["--config", str(config)]), environ={})

    assert env["STATIC_NOISE_SAMPLE_FRAMES"] == "auto"
    assert env["STATIC_BACKGROUND_SAMPLE_FRAMES"] == "auto"
    assert report["options"]["static_noise_sample_frames"]["value"] == "auto"
    assert report["options"]["static_background_sample_frames"]["value"] == "auto"


def test_cli_overrides_environment_and_environment_overrides_config(tmp_path):
    config = tmp_path / "beltmap.toml"
    config.write_text(
        """
        detection_threshold = 3.0
        max_frames = 10
        """,
        encoding="utf-8",
    )

    env, report = resolve_driver_env(
        parse_args([
            "--config",
            str(config),
            "--detection-threshold",
            "7.5",
            "--allow-full-frame-auto-velocity",
        ]),
        environ={"DETECTION_THRESHOLD": "6.0"},
    )

    assert env["DETECTION_THRESHOLD"] == "7.5"
    assert env["MAX_FRAMES"] == "10"
    assert env["ALLOW_FULL_FRAME_AUTO_VELOCITY"] == "1"
    assert report["options"]["detection_threshold"]["source"] == "cli"
    assert report["options"]["max_frames"]["source"].startswith("config:")


def test_config_rejects_unknown_keys(tmp_path):
    config = tmp_path / "beltmap.toml"
    config.write_text(
        """
        [belt]
        velocit_px_per_frame = 12.0
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown config option"):
        resolve_driver_env(parse_args(["--config", str(config)]), environ={})


def test_config_template_writer(tmp_path):
    path = tmp_path / "beltmap.toml"

    write_config_template(path)

    text = path.read_text(encoding="utf-8")
    assert "[paths]" in text
    assert "velocity_px_per_frame" in text
    assert "max_bbox_aspect_ratio" in text
    assert "reconstruction_trim_fraction" in text
    assert "sampling_strategy" in text
    assert "particle_mask_margin_px" in text
    assert "noise_exclusion_sigma" in text
