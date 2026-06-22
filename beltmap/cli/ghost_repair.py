from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from beltmap.ghost_repair import (
    build_ghost_defect_maps,
    config_from_map_only_metrics,
    load_json_object,
    load_phase_by_frame,
    local_inpaint_belt_map,
    map_only_metric_row,
    read_csv_rows,
    run_rebuild_masked_apply,
    run_map_only_for_map,
    selected_ghost_track_ids,
    track_rows_by_id,
    write_before_after,
    write_csv_rows,
    write_defect_report,
    write_defect_overlay,
    write_report,
)


SUMMARY_FIELDS = [
    "map_variant",
    "belt_map_path",
    "map_only_false_detections",
    "map_only_false_tracks",
    "map_only_false_long_tracks",
    "map_only_false_accepted_tracks",
    "map_only_proxy_ghost_penalty",
    "full100_labeled_metrics_status",
]

TRACK_FIELDS = [
    "track_id",
    "n_detections",
    "map_y_min",
    "map_y_max",
    "map_x_min",
    "map_x_max",
    "max_signal",
    "belt_y_rms_px",
    "belt_x_std_px",
    "causal_ghost_score",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beltmap-ghost-repair",
        description="Prototype GhostRepair: localize and repair map-only ghost artifacts in belt_map.npy.",
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="BeltMap output directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Separate GhostRepair output directory.")
    parser.add_argument("--belt-map-path", type=Path, help="Default: INPUT_DIR/belt_map.npy")
    parser.add_argument("--phase-estimates-path", type=Path, help="Default: INPUT_DIR/phase_estimates.csv")
    parser.add_argument("--map-only-detections-path", type=Path, help="Map-only detections CSV.")
    parser.add_argument("--map-only-tracks-path", type=Path, help="Map-only track membership CSV.")
    parser.add_argument("--map-only-track-scores-path", type=Path, help="Map-only track scores CSV.")
    parser.add_argument("--map-only-velocities-path", type=Path, help="Map-only velocities CSV.")
    parser.add_argument("--map-only-metrics-path", type=Path, help="Map-only metrics JSON.")
    parser.add_argument("--mask-margin-px", type=int, default=2)
    parser.add_argument("--inpaint-radius-px", type=int, default=16)
    parser.add_argument(
        "--run-rebuild-masked",
        action="store_true",
        help="Run beltmap-apply from INPUT_DIR/config_resolved.json with MAP_EXCLUSION_MASK_PATH set.",
    )
    parser.add_argument(
        "--rebuild-config-resolved-path",
        type=Path,
        help="Resolved beltmap-apply config JSON used by --run-rebuild-masked. Default: INPUT_DIR/config_resolved.json",
    )
    parser.add_argument(
        "--rebuild-masked-output-dir",
        type=Path,
        help="Output directory for --run-rebuild-masked. Default: OUTPUT_DIR/rebuild_masked_apply",
    )
    parser.add_argument(
        "--rebuild-masked-map-path",
        type=Path,
        help="Existing rebuild-masked belt_map.npy to include and score instead of running a rebuild.",
    )
    parser.add_argument("--skip-map-only-rerun", action="store_true", help="Build masks/repair but do not rerun map-only negative control.")
    parser.add_argument("--quiet", action="store_true")
    return parser


def default_map_only_path(output_dir: Path, stem: str, suffix: str) -> Path:
    preferred = output_dir / f"map_only_negative_control_{stem}.{suffix}"
    if preferred.exists():
        return preferred
    fallback = output_dir / f"{stem}.{suffix}"
    return fallback


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        input_dir = args.input_dir
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        belt_map_path = args.belt_map_path or input_dir / "belt_map.npy"
        phase_path = args.phase_estimates_path or input_dir / "phase_estimates.csv"
        metrics_path = args.map_only_metrics_path or default_map_only_path(input_dir, "metrics", "json")
        tracks_path = args.map_only_tracks_path or default_map_only_path(input_dir, "tracks", "csv")
        track_scores_path = args.map_only_track_scores_path or default_map_only_path(input_dir, "track_scores", "csv")
        velocities_path = args.map_only_velocities_path or default_map_only_path(input_dir, "velocities", "csv")

        belt_map = np.load(belt_map_path)
        metrics = load_json_object(metrics_path)
        tracks_by_id = track_rows_by_id(read_csv_rows(tracks_path))
        track_scores = read_csv_rows(track_scores_path)
        velocities = read_csv_rows(velocities_path)
        long_track_length = int(
            metrics.get("tracks", {}).get("long_track_length", 10)
            if isinstance(metrics.get("tracks"), dict)
            else 10
        )
        selected_track_ids = selected_ghost_track_ids(
            tracks_by_id=tracks_by_id,
            track_scores=track_scores,
            velocities=velocities,
            long_track_length=long_track_length,
        )
        mask, counts, probability, track_table = build_ghost_defect_maps(
            belt_map_shape=tuple(belt_map.shape),
            tracks_by_id=tracks_by_id,
            selected_track_ids=selected_track_ids,
            phase_by_frame=load_phase_by_frame(phase_path),
            metrics=metrics,
            margin_px=args.mask_margin_px,
        )
        np.save(output_dir / "ghost_defect_mask.npy", mask)
        np.save(output_dir / "ghost_defect_counts.npy", counts)
        np.save(output_dir / "ghost_defect_probability.npy", probability)
        write_csv_rows(output_dir / "ghost_defect_tracks.csv", track_table, TRACK_FIELDS)
        write_defect_overlay(output_dir / "ghost_defect_overlay.png", belt_map, mask)
        write_defect_report(
            output_dir / "ghost_defect_report.md",
            track_rows=track_table,
            defect_pixels=int(np.count_nonzero(mask)),
            max_count=int(np.max(counts)) if counts.size else 0,
            overlay_path=output_dir / "ghost_defect_overlay.png",
        )

        repaired = local_inpaint_belt_map(
            belt_map,
            mask,
            radius_px=args.inpaint_radius_px,
        )
        local_repaired_path = output_dir / "local_inpaint_repaired_belt_map.npy"
        final_repaired_path = output_dir / "repaired_belt_map.npy"
        np.save(local_repaired_path, repaired)
        np.save(final_repaired_path, repaired)
        write_before_after(output_dir / "ghost_repair_before_after.png", belt_map, repaired, mask)

        rebuild_output_dir = args.rebuild_masked_output_dir or output_dir / "rebuild_masked_apply"
        rebuild_config_path = args.rebuild_config_resolved_path or input_dir / "config_resolved.json"
        rebuild_mask_path = output_dir / "ghost_defect_mask.npy"
        rebuild_map_path = args.rebuild_masked_map_path
        rebuild_status = "available_via_driver_map_exclusion_mask"
        if args.run_rebuild_masked:
            rebuild_map_path = run_rebuild_masked_apply(
                resolved_config_path=rebuild_config_path,
                output_dir=rebuild_output_dir,
                mask_path=rebuild_mask_path,
            )
            rebuild_status = "executed"
        elif rebuild_map_path is not None:
            rebuild_status = "provided_existing_map"

        rebuild_manifest = {
            "status": rebuild_status,
            "ghost_defect_mask": str(rebuild_mask_path),
            "map_exclusion_mask_env": "MAP_EXCLUSION_MASK_PATH",
            "resolved_config_path": str(rebuild_config_path),
            "rebuild_output_dir": str(rebuild_output_dir),
            "rebuild_masked_map_path": "" if rebuild_map_path is None else str(rebuild_map_path),
            "how_to_run": (
                "beltmap-ghost-repair --input-dir INPUT --output-dir OUTPUT "
                "--run-rebuild-masked"
            ),
        }
        (output_dir / "rebuild_masked_manifest.json").write_text(
            json.dumps(rebuild_manifest, indent=2) + "\n",
            encoding="utf-8",
        )

        summary_rows = []
        if args.skip_map_only_rerun:
            summary_rows = [
                {
                    "map_variant": "original",
                    "belt_map_path": str(belt_map_path),
                    "map_only_false_detections": "",
                    "map_only_false_tracks": "",
                    "map_only_false_long_tracks": "",
                    "map_only_false_accepted_tracks": "",
                    "map_only_proxy_ghost_penalty": "",
                    "full100_labeled_metrics_status": "not_rerun",
                },
                {
                    "map_variant": "local_inpaint",
                    "belt_map_path": str(final_repaired_path),
                    "map_only_false_detections": "",
                    "map_only_false_tracks": "",
                    "map_only_false_long_tracks": "",
                    "map_only_false_accepted_tracks": "",
                    "map_only_proxy_ghost_penalty": "",
                    "full100_labeled_metrics_status": "not_rerun",
                },
            ]
            if rebuild_map_path is not None:
                summary_rows.append(
                    {
                        "map_variant": "rebuild_masked",
                        "belt_map_path": str(rebuild_map_path),
                        "map_only_false_detections": "",
                        "map_only_false_tracks": "",
                        "map_only_false_long_tracks": "",
                        "map_only_false_accepted_tracks": "",
                        "map_only_proxy_ghost_penalty": "",
                        "full100_labeled_metrics_status": "not_rerun",
                    }
                )
        else:
            config = config_from_map_only_metrics(metrics)
            original_metrics = run_map_only_for_map(
                label="original",
                output_dir=output_dir,
                base_output_dir=input_dir,
                belt_map_path=belt_map_path,
                phase_estimates_path=phase_path if phase_path.exists() else None,
                config=config,
            )
            repaired_metrics = run_map_only_for_map(
                label="local_inpaint",
                output_dir=output_dir,
                base_output_dir=input_dir,
                belt_map_path=final_repaired_path,
                phase_estimates_path=phase_path if phase_path.exists() else None,
                config=config,
            )
            summary_rows = [
                map_only_metric_row("original", original_metrics, belt_map_path),
                map_only_metric_row("local_inpaint", repaired_metrics, final_repaired_path),
            ]
            if rebuild_map_path is not None:
                rebuild_metrics = run_map_only_for_map(
                    label="rebuild_masked",
                    output_dir=output_dir,
                    base_output_dir=input_dir,
                    belt_map_path=rebuild_map_path,
                    phase_estimates_path=phase_path if phase_path.exists() else None,
                    config=config,
                )
                summary_rows.append(
                    map_only_metric_row("rebuild_masked", rebuild_metrics, rebuild_map_path)
                )
        if rebuild_map_path is None:
            summary_rows.append(
                {
                    "map_variant": "rebuild_masked",
                    "belt_map_path": "",
                    "map_only_false_detections": "",
                    "map_only_false_tracks": "",
                    "map_only_false_long_tracks": "",
                    "map_only_false_accepted_tracks": "",
                    "map_only_proxy_ghost_penalty": "",
                    "full100_labeled_metrics_status": "not_run_available_via_map_exclusion_mask",
                }
            )
        write_csv_rows(output_dir / "ghost_repair_summary.csv", summary_rows, SUMMARY_FIELDS)
        write_report(
            output_dir / "ghost_repair_report.md",
            summary_rows=summary_rows,
            selected_track_ids=selected_track_ids,
            defect_pixels=int(np.count_nonzero(mask)),
            rebuild_status=rebuild_status,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    if not args.quiet:
        print(
            json.dumps(
                {
                    "output_dir": str(args.output_dir),
                    "repaired_belt_map": str(args.output_dir / "repaired_belt_map.npy"),
                    "summary": str(args.output_dir / "ghost_repair_summary.csv"),
                    "report": str(args.output_dir / "ghost_repair_report.md"),
                },
                indent=2,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
