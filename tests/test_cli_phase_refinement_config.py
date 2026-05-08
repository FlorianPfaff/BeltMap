from beltmap.cli.apply import build_parser, resolve_driver_env, write_config_template


def parse_args(arguments):
    return build_parser().parse_args(arguments)


def test_phase_refinement_toml_config_maps_to_driver_environment(tmp_path):
    config = tmp_path / "beltmap.toml"
    config.write_text(
        """
        [phase_refinement]
        iterations = 2
        min_score = 0.15
        max_abs_correction_px = 6.5
        smoothing_window_frames = 31
        """,
        encoding="utf-8",
    )

    env, report = resolve_driver_env(parse_args(["--config", str(config)]), environ={})

    assert env["PHASE_REFINEMENT_ITERATIONS"] == "2"
    assert env["PHASE_REFINEMENT_MIN_SCORE"] == "0.15"
    assert env["PHASE_REFINEMENT_MAX_ABS_CORRECTION_PX"] == "6.5"
    assert env["PHASE_REFINEMENT_SMOOTHING_WINDOW_FRAMES"] == "31"
    assert report["options"]["phase_refinement_iterations"]["source"].startswith("config:")


def test_phase_refinement_cli_overrides_environment():
    env, report = resolve_driver_env(
        parse_args(["--phase-refinement-iterations", "3"]),
        environ={"PHASE_REFINEMENT_ITERATIONS": "1"},
    )

    assert env["PHASE_REFINEMENT_ITERATIONS"] == "3"
    assert report["options"]["phase_refinement_iterations"]["source"] == "cli"


def test_config_template_includes_phase_refinement_section(tmp_path):
    path = tmp_path / "beltmap.toml"

    write_config_template(path)

    text = path.read_text(encoding="utf-8")
    assert "[phase_refinement]" in text
    assert "smoothing_window_frames" in text
