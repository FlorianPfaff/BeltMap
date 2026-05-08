"""Packaged image-sequence driver entry point for BeltMap."""

from __future__ import annotations

import json
import os
from dataclasses import asdict

import numpy as np

from . import (
    BeltMotionModel,
    ParticleComponentConfig,
    ParticleTrackingConfig,
    PhaseRegistrationConfig,
    ResidualConfig,
    detect_particles_from_residual,
    estimate_particle_velocities_vs_belt,
    extract_particle_detections,
    render_clean_belt_residual,
    track_particle_detections,
)
from . import _driver_runtime as rt
from ._driver_map import build_belt_map
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


def optional_positive_int(name: str) -> int | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


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
    )

    belt_map, reference_phase, map_height = build_belt_map(
        paths,
        region,
        belt_velocity,
        period_px,
        mask_iterations=map_mask_iterations,
        mask_threshold=map_particle_mask_threshold,
        mask_mode=map_particle_mask_mode,
        mask_grow_threshold=map_particle_mask_grow_threshold,
        mask_dilation_px=map_particle_mask_dilation_px,
        mask_margin_px=map_particle_mask_margin_px,
        mask_min_area_px=map_particle_mask_min_area_px,
    )
    np.save(rt.OUT / "belt_map.npy", belt_map)
    rt.save_png(belt_map, rt.OUT / "belt_map.png")
    rt.emit(
        "belt_map",
        "saved belt-map outputs",
        belt_map_shape=list(belt_map.shape),
        belt_map_npy=rt.OUT / "belt_map.npy",
        belt_map_png=rt.OUT / "belt_map.png",
    )

    motion_model = BeltMotionModel(
        image_velocity_px_per_frame=belt_velocity,
        period_px=float(map_height),
        reference_frame=0.0,
        reference_phase_px=reference_phase,
    )
    registration_config = PhaseRegistrationConfig(
        search_radius_px=rt.env_float("REGISTRATION_SEARCH_RADIUS_PX", 8.0, minimum=0.0),
        search_step_px=rt.env_float("REGISTRATION_SEARCH_STEP_PX", 0.5, minimum=1e-9),
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
    detection_start = rt.time.perf_counter()
    for frame_index, path in enumerate(paths):
        frame = rt.crop(rt.read_gray(path), region)
        residual = render_clean_belt_residual(
            image=frame,
            belt_map=belt_map,
            frame_index=float(frame_index),
            motion_model=motion_model,
            belt_region=None,
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
    if abs(belt_velocity) > 1e-9:
        rt.emit("velocity", "estimating particle velocities relative to belt", min_track_length=min_track_length)
        for velocity in estimate_particle_velocities_vs_belt(tracks, belt_image_velocity_px_per_frame=belt_velocity, min_track_length=min_track_length):
            velocity_rows.append(asdict(velocity))
    else:
        rt.emit("velocity", "skipped particle velocity estimation because belt velocity is near zero")
    rt.write_csv(rt.OUT / "velocities.csv", velocity_rows, VELOCITY_FIELDS)
    rt.emit("velocity", "wrote velocity estimates", velocity_estimates=len(velocity_rows))

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
        "n_phase_estimates": len(phase_rows),
        "n_detections": len(detection_rows),
        "n_tracks": len(tracks),
        "n_velocity_estimates": len(velocity_rows),
        "auto_velocity_pair_shifts": pair_shifts,
        "elapsed_s": rt.elapsed_s(),
    }
    metadata_path = rt.OUT / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    rt.emit("done", "finished BeltMap image driver", metadata_json=metadata_path)


if __name__ == "__main__":  # pragma: no cover
    main()
