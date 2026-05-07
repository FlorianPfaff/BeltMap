from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib


TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class ConfigOption:
    name: str
    env_var: str
    kind: str
    help: str
    config_keys: tuple[tuple[str, ...], ...]
    metavar: str | None = None


OPTIONS: tuple[ConfigOption, ...] = (
    ConfigOption(
        "image_dir",
        "BELTMAP_IMAGE_DIR",
        "path",
        "Directory containing input images.",
        (("image_dir",), ("paths", "image_dir")),
        "DIR",
    ),
    ConfigOption(
        "output_dir",
        "BELTMAP_OUTPUT_DIR",
        "path",
        "Directory where BeltMap outputs are written.",
        (("output_dir",), ("paths", "output_dir")),
        "DIR",
    ),
    ConfigOption(
        "belt_region",
        "BELT_REGION",
        "region",
        "Belt crop as top,left,height,width. Omit to use the full frame.",
        (("belt_region",), ("belt", "region")),
        "TOP,LEFT,HEIGHT,WIDTH",
    ),
    ConfigOption(
        "belt_velocity_px_per_frame",
        "BELT_VELOCITY_PX_PER_FRAME",
        "velocity",
        "Signed belt image velocity in pixels per frame, or 'auto'.",
        (("belt_velocity_px_per_frame",), ("belt", "velocity_px_per_frame")),
        "PX_PER_FRAME|auto",
    ),
    ConfigOption(
        "belt_period_px",
        "BELT_PERIOD_PX",
        "int",
        "Optional belt circumference/period in pixels.",
        (("belt_period_px",), ("belt", "period_px")),
        "PX",
    ),
    ConfigOption(
        "detection_threshold",
        "DETECTION_THRESHOLD",
        "float",
        "Threshold on normalized residuals for bright particles.",
        (("detection_threshold",), ("detection", "threshold")),
        "Z",
    ),
    ConfigOption(
        "min_area_px",
        "MIN_AREA_PX",
        "int",
        "Minimum connected-component area for detections.",
        (("min_area_px",), ("detection", "min_area_px")),
        "PX",
    ),
    ConfigOption(
        "min_track_length",
        "MIN_TRACK_LENGTH",
        "int",
        "Minimum detections per track for velocity estimates.",
        (("min_track_length",), ("tracking", "min_track_length")),
        "N",
    ),
    ConfigOption(
        "max_match_distance_px",
        "MAX_MATCH_DISTANCE_PX",
        "float",
        "Optional tracking match distance. Omit to derive it from belt speed.",
        (("max_match_distance_px",), ("tracking", "max_match_distance_px")),
        "PX",
    ),
    ConfigOption(
        "max_frames",
        "MAX_FRAMES",
        "int",
        "Maximum number of selected frames to process. Use 0 for all frames.",
        (("max_frames",), ("frames", "max_frames")),
        "N",
    ),
    ConfigOption(
        "frame_stride",
        "FRAME_STRIDE",
        "int",
        "Process every Nth frame after natural sorting.",
        (("frame_stride",), ("frames", "stride")),
        "N",
    ),
    ConfigOption(
        "map_sample_frames",
        "MAP_SAMPLE_FRAMES",
        "int",
        "Number of frames sampled to build the belt map.",
        (("map_sample_frames",), ("map", "sample_frames")),
        "N",
    ),
    ConfigOption(
        "map_mask_iterations",
        "MAP_MASK_ITERATIONS",
        "int",
        "Particle-mask refinement iterations while building the belt map.",
        (("map_mask_iterations",), ("map", "mask_iterations")),
        "N",
    ),
    ConfigOption(
        "map_particle_mask_threshold",
        "MAP_PARTICLE_MASK_THRESHOLD",
        "float",
        "Threshold used for particle masking during map building.",
        (("map_particle_mask_threshold",), ("map", "particle_mask_threshold")),
        "Z",
    ),
    ConfigOption(
        "map_particle_mask_margin_px",
        "MAP_PARTICLE_MASK_MARGIN_PX",
        "int",
        "Safety margin around detected particle boxes during map building.",
        (("map_particle_mask_margin_px",), ("map", "particle_mask_margin_px")),
        "PX",
    ),
    ConfigOption(
        "map_particle_mask_min_area_px",
        "MAP_PARTICLE_MASK_MIN_AREA_PX",
        "int",
        "Minimum component area for map-building particle masks.",
        (("map_particle_mask_min_area_px",), ("map", "particle_mask_min_area_px")),
        "PX",
    ),
    ConfigOption(
        "velocity_search_radius_px",
        "VELOCITY_SEARCH_RADIUS_PX",
        "int",
        "Max vertical shift searched during automatic belt-velocity estimation.",
        (("velocity_search_radius_px",), ("auto_velocity", "search_radius_px")),
        "PX",
    ),
    ConfigOption(
        "velocity_estimation_pairs",
        "VELOCITY_ESTIMATION_PAIRS",
        "int",
        "Number of adjacent frame pairs used for automatic velocity estimation.",
        (("velocity_estimation_pairs",), ("auto_velocity", "estimation_pairs")),
        "N",
    ),
    ConfigOption(
        "auto_velocity_min_abs_px_per_frame",
        "AUTO_VELOCITY_MIN_ABS_PX_PER_FRAME",
        "float",
        "Minimum accepted absolute auto-estimated belt velocity.",
        (("auto_velocity_min_abs_px_per_frame",), ("auto_velocity", "min_abs_px_per_frame")),
        "PX_PER_FRAME",
    ),
    ConfigOption(
        "auto_velocity_max_edge_fraction",
        "AUTO_VELOCITY_MAX_EDGE_FRACTION",
        "float",
        "Maximum accepted fraction of auto-velocity shifts that hit the search edge.",
        (("auto_velocity_max_edge_fraction",), ("auto_velocity", "max_edge_fraction")),
        "FRACTION",
    ),
    ConfigOption(
        "allow_full_frame_auto_velocity",
        "ALLOW_FULL_FRAME_AUTO_VELOCITY",
        "bool",
        "Allow automatic belt-velocity estimation on a full-frame belt region.",
        (("allow_full_frame_auto_velocity",), ("auto_velocity", "allow_full_frame")),
    ),
    ConfigOption(
        "registration_search_radius_px",
        "REGISTRATION_SEARCH_RADIUS_PX",
        "float",
        "Phase-registration search radius in pixels.",
        (("registration_search_radius_px",), ("registration", "search_radius_px")),
        "PX",
    ),
    ConfigOption(
        "registration_search_step_px",
        "REGISTRATION_SEARCH_STEP_PX",
        "float",
        "Phase-registration search step in pixels.",
        (("registration_search_step_px",), ("registration", "search_step_px")),
        "PX",
    ),
    ConfigOption(
        "progress_interval_frames",
        "PROGRESS_INTERVAL_FRAMES",
        "int",
        "Print progress every N frames during long stages.",
        (("progress_interval_frames",), ("progress", "interval_frames")),
        "N",
    ),
    ConfigOption(
        "partial_output_interval_frames",
        "PARTIAL_OUTPUT_INTERVAL_FRAMES",
        "int",
        "Write partial CSV outputs every N processed frames. Use 0 for final only.",
        (("partial_output_interval_frames",), ("progress", "partial_output_interval_frames")),
        "N",
    ),
    ConfigOption(
        "debug_residual_preview_frames",
        "DEBUG_RESIDUAL_PREVIEW_FRAMES",
        "int",
        "Save residual PNG previews for the first N frames.",
        (("debug_residual_preview_frames",), ("debug", "residual_preview_frames")),
        "N",
    ),
    ConfigOption(
        "debug_residual_preview_interval_frames",
        "DEBUG_RESIDUAL_PREVIEW_INTERVAL_FRAMES",
        "int",
        "Also save residual previews every N frames. Use 0 to disable.",
        (("debug_residual_preview_interval_frames",), ("debug", "residual_preview_interval_frames")),
        "N",
    ),
)

OPTION_BY_NAME = {option.name: option for option in OPTIONS}
OPTION_BY_ENV = {option.env_var: option for option in OPTIONS}
CONFIG_KEY_TO_OPTION = {
    key: option for option in OPTIONS for key in option.config_keys
}

CONFIG_TEMPLATE = """# BeltMap image-sequence driver configuration.
# CLI flags override environment variables, and environment variables override
# values from this file.

[paths]
image_dir = "data/images"
output_dir = "outputs"

[frames]
max_frames = 0
stride = 1

[belt]
# Crop format is [top, left, height, width]. Omit this key to use the full frame.
region = [0, 220, 1330, 1800]
# Use a number for a known signed belt velocity, or "auto" to estimate it.
velocity_px_per_frame = "auto"
# period_px = 14723

[detection]
threshold = 5.0
min_area_px = 4

[tracking]
min_track_length = 2
# max_match_distance_px = 90.0

[map]
sample_frames = 120
mask_iterations = 1
particle_mask_threshold = 5.0
particle_mask_margin_px = 8
particle_mask_min_area_px = 4

[auto_velocity]
search_radius_px = 90
estimation_pairs = 100
min_abs_px_per_frame = 0.25
max_edge_fraction = 0.2
allow_full_frame = false

[registration]
search_radius_px = 8.0
search_step_px = 0.5

[progress]
interval_frames = 25
partial_output_interval_frames = 250

[debug]
residual_preview_frames = 3
residual_preview_interval_frames = 0
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beltmap-apply",
        description="Apply BeltMap to an image sequence and write residual, detection, tracking, and velocity outputs.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="TOML or JSON config file. Values can be flat or grouped into sections.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved driver environment and exit without running the image driver.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved driver environment before running the image driver.",
    )
    parser.add_argument(
        "--write-config-template",
        type=Path,
        metavar="PATH",
        help="Write a TOML config template and exit.",
    )

    for option in OPTIONS:
        flag = f"--{option.name.replace('_', '-')}"
        help_text = f"{option.help} [env: {option.env_var}]"
        if option.kind == "bool":
            parser.add_argument(
                flag,
                dest=option.name,
                action=argparse.BooleanOptionalAction,
                default=None,
                help=help_text,
            )
        else:
            parser.add_argument(
                flag,
                dest=option.name,
                metavar=option.metavar,
                help=help_text,
            )
    return parser


def load_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    elif suffix in {".toml", ".tml"}:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported config file extension {suffix!r}; use .toml or .json")
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a top-level object/table")
    return data


def flatten_config(data: Mapping[str, Any], prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    flattened: dict[tuple[str, ...], Any] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise ValueError(f"Config keys must be strings, got {key!r}")
        path = prefix + (key,)
        if isinstance(value, Mapping):
            flattened.update(flatten_config(value, path))
        else:
            flattened[path] = value
    return flattened


def values_from_config(path: Path | None) -> tuple[dict[str, str], dict[str, str]]:
    if path is None:
        return {}, {}

    values: dict[str, str] = {}
    sources: dict[str, str] = {}
    for key_path, raw_value in flatten_config(load_config_file(path)).items():
        option = CONFIG_KEY_TO_OPTION.get(key_path)
        if option is None:
            dotted = ".".join(key_path)
            raise ValueError(f"Unknown config option {dotted!r}")
        normalized = normalize_value(option, raw_value)
        if normalized is None:
            continue
        if option.name in values:
            raise ValueError(f"Config option {option.name!r} was specified more than once")
        values[option.name] = normalized
        sources[option.name] = f"config:{path}"
    return values, sources


def values_from_environment(environ: Mapping[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    values: dict[str, str] = {}
    sources: dict[str, str] = {}
    for option in OPTIONS:
        raw_value = environ.get(option.env_var)
        if raw_value is None or raw_value.strip() == "":
            continue
        normalized = normalize_value(option, raw_value)
        if normalized is None:
            continue
        values[option.name] = normalized
        sources[option.name] = f"env:{option.env_var}"
    return values, sources


def values_from_args(namespace: argparse.Namespace) -> tuple[dict[str, str], dict[str, str]]:
    values: dict[str, str] = {}
    sources: dict[str, str] = {}
    for option in OPTIONS:
        raw_value = getattr(namespace, option.name)
        if raw_value is None:
            continue
        normalized = normalize_value(option, raw_value)
        if normalized is None:
            continue
        values[option.name] = normalized
        sources[option.name] = "cli"
    return values, sources


def resolve_driver_env(
    namespace: argparse.Namespace,
    environ: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Resolve config-file, environment, and CLI values into driver env vars."""

    current_environ = os.environ if environ is None else environ
    merged: dict[str, str] = {}
    sources: dict[str, str] = {}

    for layer_values, layer_sources in (
        values_from_config(namespace.config),
        values_from_environment(current_environ),
        values_from_args(namespace),
    ):
        merged.update(layer_values)
        sources.update(layer_sources)

    env_updates = {OPTION_BY_NAME[name].env_var: value for name, value in merged.items()}
    report = {
        "precedence": ["config", "environment", "cli"],
        "options": {
            name: {
                "env_var": OPTION_BY_NAME[name].env_var,
                "value": value,
                "source": sources[name],
            }
            for name, value in sorted(merged.items())
        },
        "driver_environment": dict(sorted(env_updates.items())),
    }
    return env_updates, report


def normalize_value(option: ConfigOption, value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None

    if option.kind == "bool":
        return "1" if parse_bool(value, option.name) else "0"
    if option.kind == "int":
        return str(parse_int(value, option.name))
    if option.kind == "float":
        return format_float(parse_float(value, option.name))
    if option.kind == "velocity":
        if isinstance(value, str) and value.lower() == "auto":
            return "auto"
        return format_float(parse_float(value, option.name))
    if option.kind == "region":
        return format_region(value, option.name)
    if option.kind in {"path", "str"}:
        return str(value)
    raise ValueError(f"Unsupported option kind {option.kind!r} for {option.name}")


def parse_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in TRUE_VALUES:
            return True
        if lowered in FALSE_VALUES:
            return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def parse_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{name} must be an integer, got {value!r}")
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def parse_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return parsed


def format_float(value: float) -> str:
    return f"{value:.15g}"


def format_region(value: Any, name: str) -> str:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        raise ValueError(f"{name} must be a comma-separated string or a four-item list")

    if len(parts) != 4:
        raise ValueError(f"{name} must contain exactly four values: top,left,height,width")
    return ",".join(str(parse_int(part, name)) for part in parts)


def apply_driver_env(env_updates: Mapping[str, str]) -> None:
    for key, value in env_updates.items():
        os.environ[key] = value


def write_config_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(CONFIG_TEMPLATE, encoding="utf-8")


def write_resolved_config(report: Mapping[str, Any], env_updates: Mapping[str, str]) -> Path:
    output_dir = Path(env_updates.get("BELTMAP_OUTPUT_DIR") or os.getenv("BELTMAP_OUTPUT_DIR", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "config_resolved.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


def run_driver(env_updates: Mapping[str, str], report: Mapping[str, Any]) -> None:
    apply_driver_env(env_updates)
    write_resolved_config(report, env_updates)

    from scripts import apply_beltmap_to_images as driver

    # The legacy driver stores these two paths as module globals at import time.
    # Reassign them as a safeguard if the module was imported before this CLI.
    driver.DATA = Path(os.getenv("BELTMAP_IMAGE_DIR", "data/images"))
    driver.OUT = Path(os.getenv("BELTMAP_OUTPUT_DIR", "outputs"))
    driver.main()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.write_config_template is not None:
        write_config_template(args.write_config_template)
        return 0

    try:
        env_updates, report = resolve_driver_env(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    if args.print_config or args.dry_run:
        print(json.dumps(report, indent=2), flush=True)

    if args.dry_run:
        return 0

    run_driver(env_updates, report)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
