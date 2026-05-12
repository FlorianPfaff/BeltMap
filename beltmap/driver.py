"""Packaged image-sequence driver entry point for BeltMap."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import (
    BeltMotionModel,
    PhaseEstimate,
    ParticleComponentConfig,
    ParticleTrackingConfig,
    PhaseRegistrationConfig,
    ResidualConfig,
    TrackFilterConfig,
    detect_particles_from_residual,
    estimate_particle_velocities_vs_belt,
    extract_particle_detections,
    render_clean_belt_residual,
    score_particle_velocities,
    track_particle_detections,
)
from . import _driver_runtime as rt
from ._driver_map import (
    PHASE_REFINEMENT_FIELDS,
    PhaseFeedbackConfig,
    build_belt_map_result,
)
from ._driver_motion import estimate_velocity, parse_region, validate_auto_velocity_region

DETECTION_FIELDS = [
    "frame_index", "image", "label", "y", "x", "area_px",
    "bbox_top", "bbox_left", "bbox_bottom", "bbox_right",
    "mean_signal", "peak_signal",
]
PHASE_FIELDS = [
    "frame_index", "image", "phase_px", "phase_fraction", "phase_rad",
    "predicted_phase_px", "correction_px", "loss", "score", "method",
]
VELOCITY_FIELDS = [
    "track_id", "n_detections", "frame_start", "frame_end",
    "velocity_y_px_per_frame", "velocity_x_px_per_frame", "speed_px_per_frame",
    "belt_velocity_y_px_per_frame", "velocity_ratio_y",
    "belt_minus_particle_velocity_y_px_per_frame",
]
TRACK_SCORE_FIELDS = [
    "track_id", "n_detections", "frame_start", "frame_end",
    "velocity_y_px_per_frame", "velocity_x_px_per_frame",
    "velocity_ratio_y", "abs_x_velocity_px_per_frame",
    "passes_min_track_length", "passes_velocity_ratio",
    "passes_lateral_velocity", "accepted", "plausibility_score",
]


def optional_positive_int(name: str) -> int | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def optional_positive_float(name: str, default: float = 0.0) -> float | None:
    value = rt.env_float(name, default, minimum=0.0)
    return None if value <= 0 else value


def optional_path(name: str) -> Path | None:
    value = os.getenv(name, "").strip()
    return Path(value) if value else None


def optional_csv_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "").strip()
    return None if value == "" else float(value)


def load_reuse_metadata(belt_map_path: Path) -> tuple[dict, Path | None]:
    metadata_path = belt_map_path.with_name("metadata.json")
    if not metadata_path.exists():
        return {}, None
    return json.loads(metadata_path.read_text(encoding="utf-8")), metadata_path


def load_phase_estimates(path: Path) -> dict[int, PhaseEstimate]:
    estimates: dict[int, PhaseEstimate] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frame_index = int(row["frame_index"])
            if frame_index in estimates:
                raise ValueError(f"duplicate phase estimate for frame {frame_index}")
            estimates[frame_index] = PhaseEstimate(
                phase_px=float(row["phase_px"]),
                frame_index=float(row["frame_index"]),
                predicted_phase_px=float(row["predicted_phase_px"]),
                correction_px=float(row["correction_px"]),
                loss=optional_csv_float(row, "loss"),
                score=optional_csv_float(row, "score"),
                method=row.get("method", "loaded_phase_estimate") or "loaded_phase_estimate",
            )
    if not estimates:
        raise ValueError(f"no phase estimates found in {path}")
    return estimates


def validate_reused_phase_estimates(
    estimates: dict[int, PhaseEstimate],
    *,
    frame_count: int,
) -> None:
    missing = [index for index in range(frame_count) if index not in estimates]
    if missing:
        preview = ", ".join(str(index) for index in missing[:8])
        raise ValueError(
            f"phase estimates are missing {len(missing)} selected frames; first missing: {preview}"
        )


def phase_estimate_row(frame_index: int, path, residual, period_px: float) -> dict:
    if residual.clean_render is None:
        raise ValueError("phase estimates require residuals with a clean belt render")
    estimate = residual.clean_render.phase_estimate
    phase_fraction = estimate.phase_px / period_px
    return {
        "frame_index": frame_index,
        "image": str(path.relative_to(rt.DATA)),
        "phase_px": estimate.phase_px,
        "phase_fraction": phase_fraction,
        "phase_rad": phase_fraction * 2.0 * np.pi,
        "predicted_phase_px": estimate.predicted_phase_px,
        "correction_px": estimate.correction_px,
        "loss": "" if estimate.loss is None else estimate.loss,
        "score": "" if estimate.score is None else estimate.score,
        "method": estimate.method,
    }


def write_detection_outputs(detections_by_frame: list, detection_rows: list[dict]) -> None:
    rt.write_csv(rt.OUT / "detections.csv", detection_rows, DETECTION_FIELDS)
    rt.write_csv(
        rt.OUT / "detections_per_frame.csv",
        [{"frame_index": i, "n_detections": len(dets)} for i, dets in enumerate(detections_by_frame)],
        ["frame_index", "n_detections"],
    )


def write_phase_outputs(phase_rows: list[dict]) -> None:
    rt.write_csv(rt.OUT / "phase_estimates.csv", phase_rows, PHASE_FIELDS)


def write_phase_refinement_outputs(phase_refinement_rows: list[dict]) -> None:
    rt.write_csv(rt.OUT / "phase_refinement.csv", phase_refinement_rows, PHASE_REFINEMENT_FIELDS)


def should_save_residual_preview(frame_index: int, preview_frames: int, preview_interval: int) -> bool:
    return frame_index < preview_frames or (preview_interval > 0 and frame_index % preview_interval == 0)


def main() -> None:
    """Run the BeltMap image-sequence driver."""

    rt.refresh_runtime_paths()
    rt.OUT.mkdir(parents=True, exist_ok=True)
    rt.emit("startup", "starting BeltMap image driver", data_dir=rt.DATA, output_dir=rt.OUT)
    paths, discovered_frame_count, frame_stride = rt.image_paths()
    rt.emit(
        "images",
        "selected image sequence",
        discovered_frames=discovered_frame_count,
        selected_frames=len(paths),
        frame_stride=frame_stride,
        first_image=paths[0],
        last_image=paths[-1],
    )

    first = rt.read_gray(paths[0])
    region = parse_region(first)
    rt.emit(
        "images",
        "loaded first frame and parsed crop region",
        first_image_shape=list(first.shape),
        belt_region={"top": region[0], "left": region[1], "height": region[2], "width": region[3]},
    )

    velocity_spec = os.getenv("BELT_VELOCITY_PX_PER_FRAME", "auto").strip().lower()
    if velocity_spec == "auto":
        validate_auto_velocity_region(region, first.shape)
        belt_velocity, pair_shifts = estimate_velocity(paths, region)
    else:
        belt_velocity, pair_shifts = float(velocity_spec), []
        rt.emit("velocity", "using supplied belt velocity", belt_velocity_px_per_frame=belt_velocity)

    period_px = optional_positive_int("BELT_PERIOD_PX")
    detection_threshold = rt.env_float("DETECTION_THRESHOLD", 5.0)
    min_area_px = rt.env_int("MIN_AREA_PX", 4, minimum=1)
    min_track_length = rt.env_int("MIN_TRACK_LENGTH", 2, minimum=1)
    map_mask_iterations = rt.env_int("MAP_MASK_ITERATIONS", 1, minimum=0)
    map_particle_mask_threshold = rt.env_float("MAP_PARTICLE_MASK_THRESHOLD", detection_threshold, minimum=0.0)
    map_particle_mask_mode = os.getenv("MAP_PARTICLE_MASK_MODE", "positive").strip().lower()
    map_particle_mask_grow_threshold = rt.env_float("MAP_PARTICLE_MASK_GROW_THRESHOLD", 2.0, minimum=0.0)
    map_particle_mask_dilation_px = rt.env_int("MAP_PARTICLE_MASK_DILATION_PX", 0, minimum=0)
    map_particle_mask_margin_px = rt.env_int("MAP_PARTICLE_MASK_MARGIN_PX", 8, minimum=0)
    map_particle_mask_min_area_px = rt.env_int("MAP_PARTICLE_MASK_MIN_AREA_PX", min_area_px, minimum=1)
    reuse_belt_map_path = optional_path("REUSE_BELT_MAP_PATH")
    reuse_phase_estimates_path = optional_path("REUSE_PHASE_ESTIMATES_PATH")
    if reuse_phase_estimates_path is not None and reuse_belt_map_path is None:
        raise ValueError("REUSE_PHASE_ESTIMATES_PATH requires REUSE_BELT_MAP_PATH")
    registration_config = PhaseRegistrationConfig(
        search_radius_px=rt.env_float("REGISTRATION_SEARCH_RADIUS_PX", 8.0, minimum=0.0),
        search_step_px=rt.env_float("REGISTRATION_SEARCH_STEP_PX", 0.5, minimum=1e-9),
    )
    phase_refinement_iterations = rt.env_int("PHASE_REFINEMENT_ITERATIONS", 0, minimum=0)
    phase_refinement_min_score = rt.env_float("PHASE_REFINEMENT_MIN_SCORE", 0.0, minimum=0.0)
    phase_refinement_max_abs_correction_px = optional_positive_float(
        "PHASE_REFINEMENT_MAX_ABS_CORRECTION_PX",
        0.0,
    )
    phase_refinement_smoothing_window_frames = rt.env_int(
        "PHASE_REFINEMENT_SMOOTHING_WINDOW_FRAMES",
        25,
        minimum=0,
    )
    rt.emit(
        "config",
        "runtime parameters",
        belt_velocity_px_per_frame=belt_velocity,
        belt_period_px=period_px,
        detection_threshold=detection_threshold,
        min_area_px=min_area_px,
        min_track_length=min_track_length,
        map_mask_iterations=map_mask_iterations,
        map_particle_mask_threshold=map_particle_mask_threshold,
        map_particle_mask_mode=map_particle_mask_mode,
        map_particle_mask_grow_threshold=map_particle_mask_grow_threshold,
        map_particle_mask_dilation_px=map_particle_mask_dilation_px,
        map_particle_mask_margin_px=map_particle_mask_margin_px,
        map_particle_mask_min_area_px=map_particle_mask_min_area_px,
        reuse_belt_map_path=reuse_belt_map_path,
        reuse_phase_estimates_path=reuse_phase_estimates_path,
        registration_search_radius_px=registration_config.search_radius_px,
        registration_search_step_px=registration_config.search_step_px,
        phase_refinement_iterations=phase_refinement_iterations,
        phase_refinement_min_score=phase_refinement_min_score,
        phase_refinement_max_abs_correction_px=phase_refinement_max_abs_correction_px,
        phase_refinement_smoothing_window_frames=phase_refinement_smoothing_window_frames,
    )

    reuse_metadata: dict = {}
    reuse_metadata_path: Path | None = None
    phase_refinement_rows: list[dict] = []
    if reuse_belt_map_path is not None:
        belt_map = np.load(reuse_belt_map_path)
        if belt_map.ndim != 2:
            raise ValueError("REUSE_BELT_MAP_PATH must point to a 2-D belt_map.npy")
        if belt_map.shape[1] != region[3]:
            raise ValueError(
                "reused belt map width does not match BELT_REGION width: "
                f"{belt_map.shape[1]} != {region[3]}"
            )
        map_height = int(belt_map.shape[0])
        reuse_metadata, reuse_metadata_path = load_reuse_metadata(reuse_belt_map_path)
        reference_phase = float(reuse_metadata.get("reference_phase_px", 0.0))
        if period_px is not None and period_px != map_height:
            rt.emit(
                "belt_map",
                "reused belt-map height differs from supplied BELT_PERIOD_PX; using loaded map height",
                supplied_period_px=period_px,
                belt_map_height_px=map_height,
            )
        write_phase_refinement_outputs(phase_refinement_rows)
        rt.emit(
            "belt_map",
            "loaded reused belt-map outputs",
            source_belt_map_npy=reuse_belt_map_path,
            source_metadata_json=reuse_metadata_path,
            belt_map_shape=list(belt_map.shape),
            reference_phase_px=reference_phase,
        )
    else:
        build_result = build_belt_map_result(
            paths=paths,
            region=region,
            velocity=belt_velocity,
            supplied_period=period_px,
            mask_iterations=map_mask_iterations,
            mask_threshold=map_particle_mask_threshold,
            mask_mode=map_particle_mask_mode,
            mask_grow_threshold=map_particle_mask_grow_threshold,
            mask_dilation_px=map_particle_mask_dilation_px,
            mask_margin_px=map_particle_mask_margin_px,
            mask_min_area_px=map_particle_mask_min_area_px,
            phase_feedback_config=PhaseFeedbackConfig(
                iterations=phase_refinement_iterations,
                min_score=phase_refinement_min_score,
                max_abs_correction_px=phase_refinement_max_abs_correction_px,
                smoothing_window_frames=phase_refinement_smoothing_window_frames,
                registration_config=registration_config,
            ),
        )
        belt_map = build_result.belt_map
        reference_phase = build_result.reference_phase
        map_height = build_result.map_height
        phase_refinement_rows = build_result.phase_refinement_rows
        write_phase_refinement_outputs(phase_refinement_rows)
    np.save(rt.OUT / "belt_map.npy", belt_map)
    rt.save_png(belt_map, rt.OUT / "belt_map.png")
    rt.emit(
        "belt_map",
        "saved belt-map outputs",
        belt_map_shape=list(belt_map.shape),
        belt_map_npy=rt.OUT / "belt_map.npy",
        belt_map_png=rt.OUT / "belt_map.png",
        phase_refinement_csv=rt.OUT / "phase_refinement.csv",
        phase_refinement_rows=len(phase_refinement_rows),
    )

    motion_model = BeltMotionModel(
        image_velocity_px_per_frame=belt_velocity,
        period_px=float(map_height),
        reference_frame=0.0,
        reference_phase_px=reference_phase,
    )
    component_config = ParticleComponentConfig(min_area_px=min_area_px)
    residual_config = ResidualConfig()

    progress_interval = rt.env_int("PROGRESS_INTERVAL_FRAMES", 25, minimum=1)
    partial_output_interval = rt.env_int("PARTIAL_OUTPUT_INTERVAL_FRAMES", 250, minimum=0)
    residual_preview_frames = rt.env_int("DEBUG_RESIDUAL_PREVIEW_FRAMES", 3, minimum=0)
    residual_preview_interval = rt.env_int("DEBUG_RESIDUAL_PREVIEW_INTERVAL_FRAMES", 0, minimum=0)
    rt.emit(
        "detect",
        "starting residual rendering and particle detection",
        selected_frames=len(paths),
        progress_interval_frames=progress_interval,
        partial_output_interval_frames=partial_output_interval,
        residual_preview_frames=residual_preview_frames,
        residual_preview_interval_frames=residual_preview_interval,
    )

    detections_by_frame = []
    detection_rows: list[dict] = []
    phase_rows: list[dict] = []
    reused_phase_estimates = (
        load_phase_estimates(reuse_phase_estimates_path)
        if reuse_phase_estimates_path is not None
        else None
    )
    if reused_phase_estimates is not None:
        validate_reused_phase_estimates(reused_phase_estimates, frame_count=len(paths))
        rt.emit(
            "detect",
            "loaded reused phase estimates",
            source_phase_estimates_csv=reuse_phase_estimates_path,
            phase_estimates=len(reused_phase_estimates),
        )
    detection_start = rt.time.perf_counter()
    for frame_index, path in enumerate(paths):
        frame = rt.crop(rt.read_gray(path), region)
        phase_estimate = (
            reused_phase_estimates[frame_index]
            if reused_phase_estimates is not None
            else None
        )
        residual = render_clean_belt_residual(
            image=frame,
            belt_map=belt_map,
            frame_index=float(frame_index),
            motion_model=motion_model,
            belt_region=None,
            phase_estimate=phase_estimate,
            registration_config=registration_config,
            residual_config=residual_config,
        )
        phase_rows.append(phase_estimate_row(frame_index, path, residual, float(map_height)))
        if should_save_residual_preview(frame_index, residual_preview_frames, residual_preview_interval):
            rt.save_png(residual, rt.OUT / f"residual_frame_{frame_index:06d}.png")
        mask = detect_particles_from_residual(residual, threshold=detection_threshold)
        detections = extract_particle_detections(mask, residual=residual, frame_index=float(frame_index), config=component_config)
        detections_by_frame.append(detections)
        for detection in detections:
            detection_rows.append({field: getattr(detection, field) for field in DETECTION_FIELDS if field != "image"})
            detection_rows[-1]["frame_index"] = frame_index
            detection_rows[-1]["image"] = str(path.relative_to(rt.DATA))
        processed = frame_index + 1
        if partial_output_interval > 0 and (processed == 1 or processed % partial_output_interval == 0):
            write_detection_outputs(detections_by_frame, detection_rows)
            write_phase_outputs(phase_rows)
            rt.emit("detect", "wrote partial detection and phase outputs", processed_frames=processed, total_detections=len(detection_rows), phase_estimates=len(phase_rows))
        if processed == 1 or processed == len(paths) or processed % progress_interval == 0:
            dt = rt.time.perf_counter() - detection_start
            fps = processed / dt if dt > 0 else float("inf")
            remaining = len(paths) - processed
            eta = remaining / fps if fps > 0 else float("inf")
            rt.emit("detect", f"processed {processed}/{len(paths)} frames", processed_frames=processed, remaining_frames=remaining, detections_this_frame=len(detections), total_detections=len(detection_rows), frames_per_second=round(fps, 4), eta_s=round(eta, 1) if np.isfinite(eta) else None, current_image=path)

    write_detection_outputs(detections_by_frame, detection_rows)
    write_phase_outputs(phase_rows)
    rt.emit("detect", "finished residual rendering, phase estimation, and detection", processed_frames=len(paths), total_detections=len(detection_rows), phase_estimates=len(phase_rows))

    max_match = os.getenv("MAX_MATCH_DISTANCE_PX", "").strip()
    tracking_config = ParticleTrackingConfig(
        max_match_distance_px=float(max_match) if max_match else max(5.0, 1.5 * abs(belt_velocity)),
        velocity_prior_y_px_per_frame=0.8 * belt_velocity,
    )
    rt.emit("track", "starting particle tracking", frames=len(detections_by_frame), max_match_distance_px=tracking_config.max_match_distance_px, velocity_prior_y_px_per_frame=tracking_config.velocity_prior_y_px_per_frame)
    tracks = track_particle_detections(detections_by_frame, config=tracking_config, frame_indices=[float(i) for i in range(len(paths))])
    rt.emit("track", "finished particle tracking", tracks=len(tracks))

    velocity_rows = []
    velocity_objects = []
    if abs(belt_velocity) > 1e-9:
        rt.emit("velocity", "estimating particle velocities relative to belt", min_track_length=min_track_length)
        for velocity in estimate_particle_velocities_vs_belt(tracks, belt_image_velocity_px_per_frame=belt_velocity, min_track_length=min_track_length):
            velocity_objects.append(velocity)
            velocity_rows.append(asdict(velocity))
    else:
        rt.emit("velocity", "skipped particle velocity estimation because belt velocity is near zero")
    rt.write_csv(rt.OUT / "velocities.csv", velocity_rows, VELOCITY_FIELDS)
    rt.emit("velocity", "wrote velocity estimates", velocity_estimates=len(velocity_rows))
    track_filter_config = TrackFilterConfig(
        min_track_length=rt.env_int("TRACK_FILTER_MIN_LENGTH", max(5, min_track_length), minimum=1),
        min_velocity_ratio_y=rt.env_float("TRACK_FILTER_MIN_VELOCITY_RATIO_Y", 0.0),
        max_velocity_ratio_y=rt.env_float("TRACK_FILTER_MAX_VELOCITY_RATIO_Y", 1.1),
        max_abs_x_velocity_px_per_frame=optional_positive_float(
            "TRACK_FILTER_MAX_ABS_X_VELOCITY_PX_PER_FRAME",
            0.0,
        ),
    )
    track_scores = score_particle_velocities(
        velocity_objects,
        config=track_filter_config,
    )
    accepted_track_ids = {score.track_id for score in track_scores if score.accepted}
    filtered_velocity_rows = [
        asdict(velocity)
        for velocity in velocity_objects
        if velocity.track_id in accepted_track_ids
    ]
    rt.write_csv(
        rt.OUT / "track_scores.csv",
        [asdict(score) for score in track_scores],
        TRACK_SCORE_FIELDS,
    )
    rt.write_csv(rt.OUT / "filtered_velocities.csv", filtered_velocity_rows, VELOCITY_FIELDS)
    rt.emit(
        "velocity",
        "wrote track-filter outputs",
        track_scores=len(track_scores),
        filtered_velocity_estimates=len(filtered_velocity_rows),
        track_filter_min_length=track_filter_config.min_track_length,
        track_filter_min_velocity_ratio_y=track_filter_config.min_velocity_ratio_y,
        track_filter_max_velocity_ratio_y=track_filter_config.max_velocity_ratio_y,
        track_filter_max_abs_x_velocity_px_per_frame=track_filter_config.max_abs_x_velocity_px_per_frame,
    )

    metadata = {
        "n_images": len(paths),
        "discovered_frame_count": discovered_frame_count,
        "frame_stride": frame_stride,
        "first_image_shape": list(first.shape),
        "belt_region": {"top": region[0], "left": region[1], "height": region[2], "width": region[3]},
        "belt_velocity_px_per_frame": belt_velocity,
        "belt_period_px_input": period_px,
        "belt_map_height_px": map_height,
        "reference_phase_px": reference_phase,
        "detection_threshold": detection_threshold,
        "min_area_px": min_area_px,
        "map_mask_iterations": map_mask_iterations,
        "map_particle_mask_threshold": map_particle_mask_threshold,
        "map_particle_mask_mode": map_particle_mask_mode,
        "map_particle_mask_grow_threshold": map_particle_mask_grow_threshold,
        "map_particle_mask_dilation_px": map_particle_mask_dilation_px,
        "map_particle_mask_margin_px": map_particle_mask_margin_px,
        "map_particle_mask_min_area_px": map_particle_mask_min_area_px,
        "phase_refinement_iterations": phase_refinement_iterations,
        "phase_refinement_min_score": phase_refinement_min_score,
        "phase_refinement_max_abs_correction_px": phase_refinement_max_abs_correction_px,
        "phase_refinement_smoothing_window_frames": phase_refinement_smoothing_window_frames,
        "reused_belt_map": reuse_belt_map_path is not None,
        "reuse_belt_map_path": "" if reuse_belt_map_path is None else str(reuse_belt_map_path),
        "reuse_phase_estimates_path": "" if reuse_phase_estimates_path is None else str(reuse_phase_estimates_path),
        "reuse_metadata_path": "" if reuse_metadata_path is None else str(reuse_metadata_path),
        "phase_estimate_source": "loaded" if reused_phase_estimates is not None else "registration",
        "n_phase_refinement_rows": len(phase_refinement_rows),
        "n_phase_refinement_used": sum(1 for row in phase_refinement_rows if row.get("used_for_refinement")),
        "n_phase_estimates": len(phase_rows),
        "n_detections": len(detection_rows),
        "n_tracks": len(tracks),
        "n_velocity_estimates": len(velocity_rows),
        "n_filtered_velocity_estimates": len(filtered_velocity_rows),
        "track_filter_min_length": track_filter_config.min_track_length,
        "track_filter_min_velocity_ratio_y": track_filter_config.min_velocity_ratio_y,
        "track_filter_max_velocity_ratio_y": track_filter_config.max_velocity_ratio_y,
        "track_filter_max_abs_x_velocity_px_per_frame": track_filter_config.max_abs_x_velocity_px_per_frame,
        "auto_velocity_pair_shifts": pair_shifts,
        "elapsed_s": rt.elapsed_s(),
    }
    metadata_path = rt.OUT / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    rt.emit("done", "finished BeltMap image driver", metadata_json=metadata_path)


if __name__ == "__main__":  # pragma: no cover
    main()
