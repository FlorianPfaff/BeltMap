"""Keep fallback track reconstruction compact for sparse absolute frame IDs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from beltmap.cli import filter_tracks as _filter_tracks

_PATCHED_ATTR = "_beltmap_filter_tracks_sparse_frame_patched"
_ORIGINAL_ATTR = "_beltmap_filter_tracks_original_reconstruct_track_rows"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the original reconstruction function if the patch is reloaded."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_reconstruct_track_rows = _unwrap_patched_callable(
    _filter_tracks.reconstruct_track_rows
)


def reconstruct_sparse_track_rows(output_dir: Path) -> list[dict[str, str]]:
    """Reconstruct memberships without allocating through the largest frame ID.

    Detection CSV files may retain absolute source-frame indices after a subset
    run. Group only the observed frames and pass those indices explicitly to the
    tracker, rather than materializing empty lists for every preceding frame.
    """

    tracks_path = output_dir / "tracks.csv"
    if tracks_path.is_file():
        return _filter_tracks.read_optional_csv_rows(tracks_path)

    detection_rows = _filter_tracks.read_optional_csv_rows(
        output_dir / "detections.csv"
    )
    if not detection_rows:
        return []

    metadata = _filter_tracks.read_json(output_dir / "metadata.json")
    config = _filter_tracks.read_json(output_dir / "config_resolved.json")
    raw_velocity = metadata.get("belt_velocity_px_per_frame")
    velocity = (
        0.0
        if raw_velocity in (None, "")
        else _filter_tracks.parse_required_float(
            {"belt_velocity_px_per_frame": raw_velocity},
            "belt_velocity_px_per_frame",
        )
    )
    max_match_text = _filter_tracks.option_value(config, "max_match_distance_px")
    max_match = (
        _filter_tracks.parse_required_float(
            {"max_match_distance_px": max_match_text},
            "max_match_distance_px",
        )
        if max_match_text is not None
        else max(5.0, 1.5 * abs(velocity))
    )

    detections_by_frame: dict[int, list[_filter_tracks.ParticleDetection]] = {}
    images_by_frame: dict[int, str] = {}
    for row in detection_rows:
        frame_index = _filter_tracks.parse_integral_field(row, "frame_index")
        detections_by_frame.setdefault(frame_index, []).append(
            _filter_tracks.parse_detection(row)
        )
        images_by_frame.setdefault(frame_index, row.get("image", ""))

    frame_indices = sorted(detections_by_frame)
    tracks = _filter_tracks.track_particle_detections(
        [detections_by_frame[frame_index] for frame_index in frame_indices],
        frame_indices=[float(frame_index) for frame_index in frame_indices],
        config=_filter_tracks.ParticleTrackingConfig(
            max_match_distance_px=max_match,
            velocity_prior_y_px_per_frame=0.8 * velocity,
        ),
    )

    rows: list[dict] = []
    for track in tracks:
        for detection_index, detection in enumerate(track.detections):
            frame_index = int(detection.frame_index)
            rows.append(
                {
                    "track_id": track.track_id,
                    "track_detection_index": detection_index,
                    "frame_index": frame_index,
                    "image": images_by_frame.get(frame_index, ""),
                    "label": detection.label,
                    "y": detection.y,
                    "x": detection.x,
                    "area_px": detection.area_px,
                    "bbox_top": detection.bbox_top,
                    "bbox_left": detection.bbox_left,
                    "bbox_bottom": detection.bbox_bottom,
                    "bbox_right": detection.bbox_right,
                    "mean_signal": (
                        "" if detection.mean_signal is None else detection.mean_signal
                    ),
                    "peak_signal": (
                        "" if detection.peak_signal is None else detection.peak_signal
                    ),
                    "recurrent_artifact_overlap_fraction": (
                        ""
                        if detection.recurrent_artifact_overlap_fraction is None
                        else detection.recurrent_artifact_overlap_fraction
                    ),
                    "recurrent_artifact_probability": (
                        ""
                        if detection.recurrent_artifact_probability is None
                        else detection.recurrent_artifact_probability
                    ),
                    "recurrent_artifact_required_peak_signal": (
                        ""
                        if detection.recurrent_artifact_required_peak_signal is None
                        else detection.recurrent_artifact_required_peak_signal
                    ),
                }
            )

    _filter_tracks.write_csv(
        tracks_path,
        rows,
        _filter_tracks.TRACK_DETECTION_FIELDS,
    )
    return rows


setattr(reconstruct_sparse_track_rows, _PATCHED_ATTR, True)
setattr(
    reconstruct_sparse_track_rows,
    _ORIGINAL_ATTR,
    _original_reconstruct_track_rows,
)
_filter_tracks.reconstruct_track_rows = reconstruct_sparse_track_rows
