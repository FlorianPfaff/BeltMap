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
static_noise_path = "previous/static_noise.npy"
static_background_path = "previous/static_background.npy"
recurrent_artifact_map_path = "previous/recurrent_artifact_map.npy"

[frames]
stride = 3

[belt]
region = [10, 20, 30, 40]
velocity_px_per_frame = "auto"
period_px = 64

[detection]
threshold = 5.5
min_area_px = 4
max_area_px = 5000
min_bbox_width_px = 3
min_bbox_height_px = 3
max_bbox_aspect_ratio = 4.0
min_bbox_extent = 0.15

[photometric]
enabled = true
trim_fraction = 0.1
max_iterations = 4
min_pixels = 512

[tracking]
min_track_length = 2
max_match_distance_px = 75
max_frame_gap = 2
assignment_method = "global"
area_cost_weight_px = 2.5
signal_cost_weight_px = 0.5
lateral_cost_weight = 1.0
max_area_ratio = 3.0

[track_filter]
min_length = 5
min_velocity_ratio_y = 0.1
max_velocity_ratio_y = 1.05
max_abs_x_velocity_px_per_frame = 12.5

[map]
particle_mask_mode = "hysteresis_abs"
particle_mask_threshold = 4.0
particle_mask_grow_threshold = 1.5
particle_mask_dilation_px = 24
particle_mask_margin_px = 16
particle_mask_min_area_px = 8
aggregation = "huber"
robust_iterations = 2
robust_huber_delta = 2.5
robust_min_scale = 0.75

[static_noise]
sample_frames = 250
min_scale = 0.25
mask_threshold = 4.0
mask_margin_px = 12
mask_min_area_px = 6

[static_background]
sample_frames = 200
mask_threshold = 3.5
mask_margin_px = 10
mask_min_area_px = 5

[recurrent_artifact]
min_revolutions = 3
margin_px = 4
max_overlap_fraction = 0.35
min_recurrence_probability = 0.25
mode = "soft"
soft_penalty_weight = 1.5

[auto_velocity]
allow_full_frame = true

[registration]
subpixel_refinement = false
robust_normalization = true

[phase_drift]
enabled = true
smoothing_alpha = 0.2
min_score = 0.1
max_abs_residual_correction_px = 3
max_abs_px = 6
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
        "DETECTION_MAX_AREA_PX": "5000",
        "DETECTION_MAX_BBOX_ASPECT_RATIO": "4",
        "DETECTION_MIN_BBOX_EXTENT": "0.15",
        "DETECTION_MIN_BBOX_HEIGHT_PX": "3",
        "DETECTION_MIN_BBOX_WIDTH_PX": "3",
        "PHOTOMETRIC_ENABLED": "1",
        "PHOTOMETRIC_MAX_ITERATIONS": "4",
        "PHOTOMETRIC_MIN_PIXELS": "512",
        "PHOTOMETRIC_TRIM_FRACTION": "0.1",
        "FRAME_STRIDE": "3",
        "MAP_PARTICLE_MASK_DILATION_PX": "24",
        "MAP_PARTICLE_MASK_GROW_THRESHOLD": "1.5",
        "MAP_PARTICLE_MASK_MARGIN_PX": "16",
        "MAP_PARTICLE_MASK_MIN_AREA_PX": "8",
        "MAP_PARTICLE_MASK_MODE": "hysteresis_abs",
        "MAP_PARTICLE_MASK_THRESHOLD": "4",
        "MAP_AGGREGATION": "huber",
        "MAP_ROBUST_HUBER_DELTA": "2.5",
        "MAP_ROBUST_ITERATIONS": "2",
        "MAP_ROBUST_MIN_SCALE": "0.75",
        "MAX_MATCH_DISTANCE_PX": "75",
        "MIN_AREA_PX": "4",
        "MIN_TRACK_LENGTH": "2",
        "REUSE_BELT_MAP_PATH": "previous/belt_map.npy",
        "REUSE_PHASE_ESTIMATES_PATH": "previous/phase_estimates.csv",
        "REUSE_RECURRENT_ARTIFACT_MAP_PATH": "previous/recurrent_artifact_map.npy",
        "REUSE_STATIC_BACKGROUND_PATH": "previous/static_background.npy",
        "REUSE_STATIC_NOISE_PATH": "previous/static_noise.npy",
        "RECURRENT_ARTIFACT_MARGIN_PX": "4",
        "RECURRENT_ARTIFACT_MAX_OVERLAP_FRACTION": "0.35",
        "RECURRENT_ARTIFACT_MIN_RECURRENCE_PROBABILITY": "0.25",
        "RECURRENT_ARTIFACT_MIN_REVOLUTIONS": "3",
        "RECURRENT_ARTIFACT_MODE": "soft",
        "RECURRENT_ARTIFACT_SOFT_PENALTY_WEIGHT": "1.5",
        "STATIC_BACKGROUND_MASK_MARGIN_PX": "10",
        "STATIC_BACKGROUND_MASK_MIN_AREA_PX": "5",
        "STATIC_BACKGROUND_MASK_THRESHOLD": "3.5",
        "STATIC_BACKGROUND_SAMPLE_FRAMES": "200",
        "STATIC_NOISE_MASK_MARGIN_PX": "12",
        "STATIC_NOISE_MASK_MIN_AREA_PX": "6",
        "STATIC_NOISE_MASK_THRESHOLD": "4",
        "STATIC_NOISE_MIN_SCALE": "0.25",
        "STATIC_NOISE_SAMPLE_FRAMES": "250",
        "TRACKING_AREA_COST_WEIGHT_PX": "2.5",
        "TRACKING_ASSIGNMENT_METHOD": "global",
        "TRACKING_MAX_FRAME_GAP": "2",
        "TRACKING_LATERAL_COST_WEIGHT": "1",
        "TRACKING_MAX_AREA_RATIO": "3",
        "TRACKING_SIGNAL_COST_WEIGHT_PX": "0.5",
        "TRACK_FILTER_MAX_ABS_X_VELOCITY_PX_PER_FRAME": "12.5",
        "TRACK_FILTER_MAX_VELOCITY_RATIO_Y": "1.05",
        "TRACK_FILTER_MIN_LENGTH": "5",
        "TRACK_FILTER_MIN_VELOCITY_RATIO_Y": "0.1",
        "REGISTRATION_ROBUST_NORMALIZATION": "1",
        "REGISTRATION_SUBPIXEL_REFINEMENT": "0",
        "PHASE_DRIFT_ENABLED": "1",
        "PHASE_DRIFT_MAX_ABS_PX": "6",
        "PHASE_DRIFT_MAX_ABS_RESIDUAL_CORRECTION_PX": "3",
        "PHASE_DRIFT_MIN_SCORE": "0.1",
        "PHASE_DRIFT_SMOOTHING_ALPHA": "0.2",
    }
    assert report["driver_environment"] == env_updates


def test_legacy_result_improvement_aliases_are_resolved(tmp_path):
    config_path = tmp_path / "aliases.toml"
    config_path.write_text(
        """
[detection]
method = "absolute"
grow_threshold = 2.0

[tracking]
matching_strategy = "greedy"
""".strip(),
        encoding="utf-8",
    )

    env_updates, _report = cli.resolve_driver_env(
        parse_args("--config", str(config_path)),
        environ={},
    )

    assert env_updates == {
        "DETECTION_LOW_THRESHOLD": "2",
        "DETECTION_MODE": "absolute",
        "TRACKING_ASSIGNMENT_METHOD": "greedy",
    }


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


def test_legacy_detection_and_tracking_config_aliases_are_resolved(tmp_path):
    config_path = tmp_path / "legacy.toml"
    config_path.write_text(
        """
[detection]
method = "hysteresis_abs"
grow_threshold = 2.0

[tracking]
matching_strategy = "greedy"
""".strip(),
        encoding="utf-8",
    )

    env_updates, _report = cli.resolve_driver_env(
        parse_args("--config", str(config_path)),
        environ={},
    )

    assert env_updates == {
        "DETECTION_LOW_THRESHOLD": "2",
        "DETECTION_MODE": "absolute",
        "TRACKING_ASSIGNMENT_METHOD": "greedy",
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


def test_map_sampling_strategy_aliases_prefer_canonical_within_same_layer(tmp_path):
    config_path = tmp_path / "sampling_aliases.toml"
    config_path.write_text(
        """
[map]
sampling_strategy = "adaptive_phase_coverage"
sample_strategy = "uniform"
""".strip(),
        encoding="utf-8",
    )

    env_updates, report = cli.resolve_driver_env(
        parse_args("--config", str(config_path)),
        environ={},
    )

    assert env_updates == {"MAP_SAMPLING_STRATEGY": "adaptive_phase_coverage"}
    assert "MAP_SAMPLE_STRATEGY" not in env_updates
    assert report["options"]["map_sampling_strategy"] == {
        "env_var": "MAP_SAMPLING_STRATEGY",
        "value": "adaptive_phase_coverage",
        "source": f"config:{config_path}",
    }


def test_map_sampling_strategy_aliases_preserve_layer_precedence(tmp_path):
    config_path = tmp_path / "sampling_precedence.toml"
    config_path.write_text(
        """
[map]
sampling_strategy = "uniform"
""".strip(),
        encoding="utf-8",
    )
    environ = {"MAP_SAMPLE_STRATEGY": "adaptive_phase_coverage"}

    env_updates, report = cli.resolve_driver_env(
        parse_args("--config", str(config_path)),
        environ=environ,
    )

    assert env_updates == {"MAP_SAMPLING_STRATEGY": "adaptive_phase_coverage"}
    assert "MAP_SAMPLE_STRATEGY" not in env_updates
    assert report["options"]["map_sampling_strategy"] == {
        "env_var": "MAP_SAMPLING_STRATEGY",
        "value": "adaptive_phase_coverage",
        "source": "env:MAP_SAMPLE_STRATEGY",
    }


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
    assert parsed["map"]["aggregation"] == "mean"
    assert parsed["map"]["robust_iterations"] == 1
    assert parsed["map"]["robust_huber_delta"] == 3.0
    assert parsed["map"]["robust_min_scale"] == 1.0
    assert parsed["static_noise"]["sample_frames"] == 0
    assert parsed["static_noise"]["mask_margin_px"] == 8
    assert parsed["static_background"]["sample_frames"] == 0
    assert parsed["static_background"]["mask_margin_px"] == 8
    assert parsed["recurrent_artifact"]["min_revolutions"] == 0
    assert parsed["recurrent_artifact"]["max_overlap_fraction"] == 0.3
    assert parsed["recurrent_artifact"]["min_recurrence_probability"] == 0.0
    assert parsed["recurrent_artifact"]["mode"] == "hard"
    assert parsed["recurrent_artifact"]["soft_penalty_weight"] == 1.0
    assert parsed["tracking"]["assignment_method"] == "global"
    assert parsed["tracking"]["area_cost_weight_px"] == 0.0
    assert parsed["tracking"]["signal_cost_weight_px"] == 0.0
    assert parsed["tracking"]["lateral_cost_weight"] == 0.0
    assert parsed["tracking"]["max_area_ratio"] == 0.0


def test_write_config_template_omits_legacy_map_sample_strategy(tmp_path):
    config_path = tmp_path / "beltmap.toml"

    cli.write_config_template(config_path)
    parsed = cli.load_config_file(config_path)

    assert parsed["map"]["sampling_strategy"] == "uniform"
    assert "sample_strategy" not in parsed["map"]


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
