from __future__ import annotations

import argparse
import json
from pathlib import Path

from beltmap.trust import (
    PROFILE_CONFIGS,
    compare_run_metadata,
    parse_region,
    physical_validation_summary,
    quality_report,
    scale_calibration_from_points,
    sequence_report,
    speed_consistency_report,
    write_json,
    write_profile,
    write_run_trust_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beltmap-trust",
        description="Preflight, trust, and evidence tools for BeltMap runs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    report = subparsers.add_parser("report", help="Write a trust/QC artifact bundle for one run.")
    report.add_argument("--output-dir", type=Path, required=True)
    report.add_argument("--image-dir", type=Path)
    report.add_argument("--belt-region", help="Optional region as top,left,height,width for image-quality sampling.")
    report.add_argument("--frame-rate-hz", type=float)
    report.add_argument("--expected-particle-flux-per-s", type=float)
    report.add_argument("--expected-mass-flux-g-s", type=float)
    report.add_argument("--particle-mass-g", type=float)
    report.add_argument("--epoch-count", type=int, default=1)

    seq = subparsers.add_parser("check-sequence", help="Check image sequence numbering and duplicate frames.")
    seq.add_argument("--image-dir", type=Path, required=True)
    seq.add_argument("--output", type=Path)
    seq.add_argument("--skip-hashes", action="store_true", help="Skip SHA-256 duplicate-image checks.")

    quality = subparsers.add_parser("quality", help="Measure blur and dynamic range over sampled frames.")
    quality.add_argument("--image-dir", type=Path, required=True)
    quality.add_argument("--belt-region", help="Optional region as top,left,height,width.")
    quality.add_argument("--sample-limit", type=int, default=100)
    quality.add_argument("--output", type=Path)

    speed = subparsers.add_parser("speed-audit", help="Check registration correction trend versus configured velocity.")
    speed.add_argument("--output-dir", type=Path, required=True)
    speed.add_argument("--output", type=Path)

    physical = subparsers.add_parser("physical-validation", help="Compare image-derived flux with external measurements.")
    physical.add_argument("--output-dir", type=Path, required=True)
    physical.add_argument("--frame-rate-hz", type=float)
    physical.add_argument("--analysis-duration-s", type=float)
    physical.add_argument("--expected-particle-flux-per-s", type=float)
    physical.add_argument("--expected-mass-flux-g-s", type=float)
    physical.add_argument("--particle-mass-g", type=float)
    physical.add_argument("--output", type=Path)

    calibrate = subparsers.add_parser("calibrate-scale", help="Calibrate px/mm from two clicked target points.")
    calibrate.add_argument("--point-a", required=True, help="First point as y,x.")
    calibrate.add_argument("--point-b", required=True, help="Second point as y,x.")
    calibrate.add_argument("--known-distance-mm", type=float, required=True)
    calibrate.add_argument("--output", type=Path, required=True)

    compare = subparsers.add_parser("compare-runs", help="Check whether multiple runs are comparable.")
    compare.add_argument("--run", type=Path, action="append", required=True, help="BeltMap output directory. Repeatable.")
    compare.add_argument("--output", type=Path)

    profile = subparsers.add_parser("write-profile", help="Write a cost-sensitive config profile overlay.")
    profile.add_argument("name", choices=sorted(PROFILE_CONFIGS))
    profile.add_argument("--output", type=Path, required=True)

    return parser


def parse_point(text: str) -> tuple[float, float]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 2:
        raise ValueError("point must be written as y,x")
    return float(parts[0]), float(parts[1])


def print_or_write(payload: dict, path: Path | None) -> None:
    if path is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        write_json(path, payload)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "report":
        artifacts = write_run_trust_artifacts(
            output_dir=args.output_dir,
            image_dir=args.image_dir,
            region=parse_region(args.belt_region),
            frame_rate_hz=args.frame_rate_hz,
            expected_particle_flux_per_s=args.expected_particle_flux_per_s,
            expected_mass_flux_g_s=args.expected_mass_flux_g_s,
            particle_mass_g=args.particle_mass_g,
            epoch_count=args.epoch_count,
        )
        print(json.dumps({key: str(path) for key, path in artifacts.items()}, indent=2, sort_keys=True))
        return 0

    if args.command == "check-sequence":
        payload = sequence_report(args.image_dir, hash_duplicates=not args.skip_hashes)
        print_or_write(payload, args.output)
        return 0

    if args.command == "quality":
        payload = quality_report(
            args.image_dir,
            region=parse_region(args.belt_region),
            sample_limit=args.sample_limit,
        )
        print_or_write(payload, args.output)
        return 0

    if args.command == "speed-audit":
        payload = speed_consistency_report(args.output_dir)
        print_or_write(payload, args.output)
        return 0

    if args.command == "physical-validation":
        payload = physical_validation_summary(
            args.output_dir,
            expected_particle_flux_per_s=args.expected_particle_flux_per_s,
            expected_mass_flux_g_s=args.expected_mass_flux_g_s,
            particle_mass_g=args.particle_mass_g,
            frame_rate_hz=args.frame_rate_hz,
            analysis_duration_s=args.analysis_duration_s,
        )
        print_or_write(payload, args.output)
        return 0

    if args.command == "calibrate-scale":
        calibration = scale_calibration_from_points(
            parse_point(args.point_a),
            parse_point(args.point_b),
            known_distance_mm=args.known_distance_mm,
        )
        write_json(
            args.output,
            {
                "px_per_mm": calibration.px_per_mm,
                "mm_per_px": calibration.mm_per_px,
                "point_a": list(calibration.point_a),
                "point_b": list(calibration.point_b),
                "known_distance_mm": calibration.known_distance_mm,
            },
        )
        return 0

    if args.command == "compare-runs":
        payload = compare_run_metadata(args.run)
        print_or_write(payload, args.output)
        return 0

    if args.command == "write-profile":
        write_profile(args.name, args.output)
        return 0

    parser.error(f"unsupported command {args.command!r}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
