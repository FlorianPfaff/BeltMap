from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from beltmap.map_only_negative_control import (
    MapOnlyNegativeControlConfig,
    generate_map_only_negative_control_report,
)

TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beltmap-map-only-negative-control",
        description=(
            "Run a map-only negative-control ghost benchmark. The command renders "
            "belt_map.npy as a high-pass pseudo-residual sequence and runs the "
            "normal detection plus PyRecEst tracking pipeline. Every detection and "
            "track is therefore a false positive."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="BeltMap output directory. Default: outputs",
    )
    parser.add_argument(
        "--belt-map-path",
        type=Path,
        help="Path to belt_map.npy. Default: OUTPUT_DIR/belt_map.npy",
    )
    parser.add_argument(
        "--phase-estimates-path",
        type=Path,
        help="Path to phase_estimates.csv. Default: OUTPUT_DIR/phase_estimates.csv",
    )
    parser.add_argument("--metrics-path", type=Path, help="Metrics JSON output path.")
    parser.add_argument("--report-path", type=Path, help="Markdown report output path.")
    parser.add_argument("--detections-path", type=Path, help="Detection CSV output path.")
    parser.add_argument("--detections-per-frame-path", type=Path, help="Per-frame CSV output path.")
    parser.add_argument("--tracks-path", type=Path, help="Track membership CSV output path.")
    parser.add_argument("--velocities-path", type=Path, help="Velocity CSV output path.")
    parser.add_argument("--track-scores-path", type=Path, help="Track-score CSV output path.")

    parser.add_argument("--threshold", type=float, help="Detection threshold in high-pass z units.")
    parser.add_argument(
        "--mode",
        "--detection-mode",
        dest="mode",
        help="Detection residual polarity: positive, negative, or absolute.",
    )
    parser.add_argument(
        "--low-threshold",
        type=float,
        help="Optional hysteresis low threshold. Values <= 0 disable hysteresis.",
    )
    parser.add_argument("--min-area-px", type=int, help="Minimum connected-component area.")
    parser.add_argument("--max-area-px", type=int, help="Optional maximum connected-component area. 0 disables.")
    parser.add_argument("--min-bbox-width-px", type=int, help="Optional minimum bounding-box width. 0 disables.")
    parser.add_argument("--min-bbox-height-px", type=int, help="Optional minimum bounding-box height. 0 disables.")
    parser.add_argument("--max-bbox-aspect-ratio", type=float, help="Optional maximum bounding-box aspect ratio. 0 disables.")
    parser.add_argument("--min-bbox-extent", type=float, help="Optional minimum area/bbox-area extent. 0 disables.")
    parser.add_argument(
        "--split-merged-components",
        dest="split_merged_components",
        action="store_true",
        default=None,
        help="Split connected components at projection valleys.",
    )
    parser.add_argument(
        "--no-split-merged-components",
        dest="split_merged_components",
        action="store_false",
        help="Disable merged-component splitting.",
    )
    parser.add_argument("--split-min-projection-gap-px", type=int, help="Projection-valley gap for splitting.")
    parser.add_argument("--split-min-component-area-px", type=int, help="Minimum area for split pieces. 0 inherits min_area_px.")

    parser.add_argument(
        "--highpass-radius-px",
        type=int,
        default=15,
        help="Local mean radius used to high-pass belt_map.npy. Default: 15",
    )
    parser.add_argument(
        "--highpass-min-scale-gray",
        type=float,
        default=1e-6,
        help="Minimum robust high-pass scale. Default: 1e-6",
    )
    parser.add_argument("--noise-sigma", type=float, default=0.0, help="Optional Gaussian z-noise added to pseudo-residual frames.")
    parser.add_argument("--random-seed", type=int, default=0, help="Random seed for --noise-sigma. Default: 0")

    parser.add_argument("--crop-height-px", type=int, help="Rendered crop height. Defaults to BELT_REGION height, then map height.")
    parser.add_argument(
        "--frame-count",
        type=int,
        help="Number of phase rows/pseudo-frames to render. 0 means all phase rows; without phase CSV, default is 100.",
    )
    parser.add_argument(
        "--belt-velocity-px-per-frame",
        type=float,
        help="Belt velocity for generated phases and velocity ratios. Inferred from phase_estimates.csv when possible.",
    )
    parser.add_argument("--period-px", type=float, help="Belt period for phase unwrapping. Defaults to map height.")

    parser.add_argument("--max-match-distance-px", type=float, help="PyRecEst tracking match distance. Default derives from belt speed.")
    parser.add_argument("--tracking-max-frame-gap", type=float, help="Maximum selected-frame gap for track association.")
    parser.add_argument("--min-track-length", type=int, help="Minimum detections per velocity row.")
    parser.add_argument("--velocity-fit-method", choices=["linear", "theil_sen"], help="Velocity slope estimator.")
    parser.add_argument("--track-filter-min-length", type=int, help="Minimum detections per accepted filtered track.")
    parser.add_argument("--track-filter-min-velocity-ratio-y", type=float, help="Minimum accepted vertical velocity ratio.")
    parser.add_argument("--track-filter-max-velocity-ratio-y", type=float, help="Maximum accepted vertical velocity ratio.")
    parser.add_argument("--track-filter-max-abs-x-velocity-px-per-frame", type=float, help="Optional lateral-velocity gate. 0 disables.")
    parser.add_argument("--long-track-length", type=int, help="Track length counted as a long false track.")
    parser.add_argument("--quiet", action="store_true", help="Do not print artifact paths and summary metrics as JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output_config = _read_json_object(args.output_dir / "config_resolved.json")
        metadata = _read_json_object(args.output_dir / "metadata.json")
        config = _build_config(args, output_config, metadata)
        result = generate_map_only_negative_control_report(
            output_dir=args.output_dir,
            config=config,
            belt_map_path=args.belt_map_path,
            phase_estimates_path=args.phase_estimates_path,
            metrics_path=args.metrics_path,
            report_path=args.report_path,
            detections_path=args.detections_path,
            detections_per_frame_path=args.detections_per_frame_path,
            tracks_path=args.tracks_path,
            velocities_path=args.velocities_path,
            track_scores_path=args.track_scores_path,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    if not args.quiet:
        metrics = result.metrics
        print(
            json.dumps(
                {
                    "metrics": str(result.artifacts.metrics),
                    "report": str(result.artifacts.report),
                    "detections": str(result.artifacts.detections),
                    "tracks": str(result.artifacts.tracks),
                    "velocities": str(result.artifacts.velocities),
                    "false_detections": metrics["detections"]["false_detections"],
                    "false_tracks": metrics["tracks"]["false_tracks"],
                    "false_long_tracks": metrics["tracks"]["false_long_tracks"],
                    "false_accepted_tracks": metrics["velocities"]["false_accepted_tracks"],
                },
                indent=2,
            ),
            flush=True,
        )
    return 0


def _build_config(
    args: argparse.Namespace,
    output_config: dict[str, Any],
    metadata: dict[str, Any],
) -> MapOnlyNegativeControlConfig:
    threshold = _float_option(args.threshold, output_config, "detection_threshold", ("detection", "threshold"), 5.0)
    mode = args.mode or _string_option(output_config, "detection_mode", ("detection", "mode"), "positive")
    low_threshold = _float_option(args.low_threshold, output_config, "detection_low_threshold", ("detection", "low_threshold"), 0.0)
    low_threshold = None if low_threshold is None or low_threshold <= 0 else low_threshold
    split_min_component_area = _none_if_nonpositive_int(
        _int_option(
            args.split_min_component_area_px,
            output_config,
            "detection_split_min_component_area_px",
            ("detection", "split_min_component_area_px"),
            0,
        )
    )
    return MapOnlyNegativeControlConfig(
        threshold=threshold,
        mode=mode,
        low_threshold=low_threshold,
        min_area_px=_int_option(args.min_area_px, output_config, "min_area_px", ("detection", "min_area_px"), 4),
        max_area_px=_none_if_nonpositive_int(_int_option(args.max_area_px, output_config, "detection_max_area_px", ("detection", "max_area_px"), 0)),
        min_bbox_width_px=_none_if_nonpositive_int(_int_option(args.min_bbox_width_px, output_config, "detection_min_bbox_width_px", ("detection", "min_bbox_width_px"), 0)),
        min_bbox_height_px=_none_if_nonpositive_int(_int_option(args.min_bbox_height_px, output_config, "detection_min_bbox_height_px", ("detection", "min_bbox_height_px"), 0)),
        max_bbox_aspect_ratio=_none_if_nonpositive_float(_float_option(args.max_bbox_aspect_ratio, output_config, "detection_max_bbox_aspect_ratio", ("detection", "max_bbox_aspect_ratio"), 0.0)),
        min_bbox_extent=_none_if_nonpositive_float(_float_option(args.min_bbox_extent, output_config, "detection_min_bbox_extent", ("detection", "min_bbox_extent"), 0.0)),
        split_merged_components=_bool_option(args.split_merged_components, output_config, "detection_split_merged_components", ("detection", "split_merged_components"), False),
        split_min_projection_gap_px=_int_option(args.split_min_projection_gap_px, output_config, "detection_split_min_projection_gap_px", ("detection", "split_min_projection_gap_px"), 2),
        split_min_component_area_px=split_min_component_area,
        highpass_radius_px=args.highpass_radius_px,
        highpass_min_scale_gray=args.highpass_min_scale_gray,
        crop_height_px=args.crop_height_px if args.crop_height_px is not None else _crop_height_from_defaults(output_config, metadata),
        frame_count=None if args.frame_count is None or args.frame_count <= 0 else args.frame_count,
        belt_velocity_px_per_frame=_belt_velocity_from_defaults(args, output_config, metadata),
        period_px=_period_from_defaults(args, output_config, metadata),
        noise_sigma=args.noise_sigma,
        random_seed=args.random_seed,
        max_match_distance_px=(
            args.max_match_distance_px
            if args.max_match_distance_px is not None
            else _none_if_nonpositive_float(
                _float_option(
                    None,
                    output_config,
                    "max_match_distance_px",
                    ("tracking", "max_match_distance_px"),
                    0.0,
                )
            )
        ),
        tracking_max_frame_gap=args.tracking_max_frame_gap if args.tracking_max_frame_gap is not None else _float_option(None, output_config, "tracking_max_frame_gap", ("tracking", "max_frame_gap"), 1.0),
        min_track_length=args.min_track_length if args.min_track_length is not None else _int_option(None, output_config, "min_track_length", ("tracking", "min_track_length"), 2),
        velocity_fit_method=args.velocity_fit_method or _string_option(output_config, "tracking_velocity_fit_method", ("tracking", "velocity_fit_method"), "linear"),
        track_filter_min_length=args.track_filter_min_length if args.track_filter_min_length is not None else _int_option(None, output_config, "track_filter_min_length", ("track_filter", "min_length"), 5),
        track_filter_min_velocity_ratio_y=args.track_filter_min_velocity_ratio_y if args.track_filter_min_velocity_ratio_y is not None else _float_option(None, output_config, "track_filter_min_velocity_ratio_y", ("track_filter", "min_velocity_ratio_y"), 0.0),
        track_filter_max_velocity_ratio_y=args.track_filter_max_velocity_ratio_y if args.track_filter_max_velocity_ratio_y is not None else _float_option(None, output_config, "track_filter_max_velocity_ratio_y", ("track_filter", "max_velocity_ratio_y"), 1.1),
        track_filter_max_abs_x_velocity_px_per_frame=_none_if_nonpositive_float(
            args.track_filter_max_abs_x_velocity_px_per_frame
            if args.track_filter_max_abs_x_velocity_px_per_frame is not None
            else _float_option(None, output_config, "track_filter_max_abs_x_velocity_px_per_frame", ("track_filter", "max_abs_x_velocity_px_per_frame"), 0.0)
        ),
        long_track_length=args.long_track_length if args.long_track_length is not None else 10,
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _config_value(config: dict[str, Any], flat_name: str, nested_path: tuple[str, ...]) -> Any:
    options = config.get("options")
    if isinstance(options, dict):
        option = options.get(flat_name)
        if isinstance(option, dict) and not _is_missing(option.get("value")):
            return option.get("value")
        if not isinstance(option, dict) and not _is_missing(option):
            return option
    if not _is_missing(config.get(flat_name)):
        return config.get(flat_name)
    node: Any = config
    for key in nested_path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return None if _is_missing(node) else node


def _string_option(
    config: dict[str, Any],
    flat_name: str,
    nested_path: tuple[str, ...],
    default: str,
) -> str:
    value = _config_value(config, flat_name, nested_path)
    return default if _is_missing(value) else str(value)


def _float_option(
    cli_value: float | None,
    config: dict[str, Any],
    flat_name: str,
    nested_path: tuple[str, ...],
    default: float,
) -> float:
    for value in (cli_value, _config_value(config, flat_name, nested_path), default):
        parsed = _finite_float(value)
        if parsed is not None:
            return parsed
    raise ValueError(f"{flat_name} must be finite")


def _int_option(
    cli_value: int | None,
    config: dict[str, Any],
    flat_name: str,
    nested_path: tuple[str, ...],
    default: int,
) -> int:
    for value in (cli_value, _config_value(config, flat_name, nested_path), default):
        parsed = _finite_float(value)
        if parsed is not None:
            return int(parsed)
    raise ValueError(f"{flat_name} must be an integer")


def _bool_option(
    cli_value: bool | None,
    config: dict[str, Any],
    flat_name: str,
    nested_path: tuple[str, ...],
    default: bool,
) -> bool:
    if cli_value is not None:
        return bool(cli_value)
    value = _config_value(config, flat_name, nested_path)
    if _is_missing(value):
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(f"{flat_name} must be a boolean")


def _belt_velocity_from_defaults(
    args: argparse.Namespace,
    config: dict[str, Any],
    metadata: dict[str, Any],
) -> float | None:
    if args.belt_velocity_px_per_frame is not None:
        return args.belt_velocity_px_per_frame
    for key in (
        "belt_velocity_px_per_frame",
        "belt_velocity_px_per_selected_frame",
        "belt_image_velocity_px_per_frame",
    ):
        parsed = _finite_float(metadata.get(key))
        if parsed is not None:
            return parsed
    value = _config_value(config, "belt_velocity_px_per_frame", ("belt", "velocity_px_per_frame"))
    return _finite_float(value)


def _period_from_defaults(
    args: argparse.Namespace,
    config: dict[str, Any],
    metadata: dict[str, Any],
) -> float | None:
    if args.period_px is not None:
        return args.period_px
    for key in ("belt_period_px", "belt_map_height_px", "map_height_px", "map_height"):
        parsed = _finite_float(metadata.get(key))
        if parsed is not None and parsed > 0:
            return parsed
    parsed = _finite_float(_config_value(config, "belt_period_px", ("belt", "period_px")))
    return parsed if parsed is not None and parsed > 0 else None


def _crop_height_from_defaults(
    config: dict[str, Any],
    metadata: dict[str, Any],
) -> int | None:
    for value in (
        _config_value(config, "belt_region", ("belt", "region")),
        metadata.get("belt_region"),
        metadata.get("region"),
    ):
        height = _region_height(value)
        if height is not None and height > 0:
            return height
    return None


def _region_height(value: Any) -> int | None:
    if isinstance(value, dict):
        parsed = _finite_float(value.get("height"))
        return None if parsed is None else int(parsed)
    if isinstance(value, (list, tuple)) and len(value) == 4:
        parsed = _finite_float(value[2])
        return None if parsed is None else int(parsed)
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(";", ",").split(",")]
        if len(parts) == 4:
            parsed = _finite_float(parts[2])
            return None if parsed is None else int(parsed)
    return None


def _none_if_nonpositive_int(value: int | None) -> int | None:
    return None if value is None or value <= 0 else int(value)


def _none_if_nonpositive_float(value: float | None) -> float | None:
    return None if value is None or value <= 0 else float(value)


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() == "auto":
            return None
        value = stripped
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
