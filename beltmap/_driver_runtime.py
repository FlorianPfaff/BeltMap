"""Runtime helpers for the packaged BeltMap image driver."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from .residual import ResidualImage

DATA = Path(os.getenv("BELTMAP_IMAGE_DIR", "data/images"))
OUT = Path(os.getenv("BELTMAP_OUTPUT_DIR", "outputs"))
EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
START_TIME = time.perf_counter()

GENERATED_OUTPUT_FILES = {
    "metadata.json",
    "progress.jsonl",
    "progress_latest.json",
    "phase_estimates.csv",
    "registration_quality.csv",
    "phase_refinement.csv",
    "photometric_fits.csv",
    "local_illumination_fits.csv",
    "detections.csv",
    "detections_per_frame.csv",
    "tracks.csv",
    "velocities.csv",
    "track_scores.csv",
    "filtered_tracks.csv",
    "filtered_velocities.csv",
    "recurrent_artifact_detections.csv",
    "cross_map_agreement_detections.csv",
    "belt_map.npy",
    "belt_map.png",
    "cross_map_agreement_map_a.npy",
    "cross_map_agreement_map_a.png",
    "cross_map_agreement_map_b.npy",
    "cross_map_agreement_map_b.png",
    "belt_map_support.npy",
    "belt_map_support.png",
    "belt_map_observed_mask.npy",
    "belt_map_observed_mask.png",
    "belt_map_interpolated_mask.npy",
    "belt_map_interpolated_mask.png",
    "belt_map_low_support_mask.npy",
    "belt_map_low_support_mask.png",
    "belt_map_risk.npy",
    "belt_map_risk.png",
    "map_risk_detections.csv",
    "static_noise.npy",
    "static_noise.png",
    "static_background.npy",
    "static_background.png",
    "recurrent_artifact_map.npy",
    "recurrent_artifact_map.png",
    "recurrent_artifact_counts.npy",
    "recurrent_artifact_counts.png",
    "recurrent_artifact_exposure_counts.npy",
    "recurrent_artifact_probability.npy",
    "recurrent_artifact_probability.png",
    "validation_report.md",
    "validation_summary.json",
    "benchmark_metrics.json",
    "benchmark_report.md",
    "revolution_split_frames.csv",
    "revolution_split_revolutions.csv",
    "revolution_split_detection_summary.csv",
    "revolution_split_score_summary.csv",
    "revolution_split_ghost_detections.csv",
    "revolution_split_summary.json",
    "revolution_split_train_artifact_map.npy",
    "revolution_split_train_artifact_map.png",
    "revolution_split_train_artifact_counts.npy",
    "revolution_split_train_artifact_counts.png",
    "revolution_split_train_artifact_exposure_counts.npy",
    "revolution_split_train_artifact_probability.npy",
    "revolution_split_train_artifact_probability.png",
    "phase_corrections.png",
    "phase_correction_timeseries.png",
    "registration_score.png",
    "detections_per_frame.png",
    "velocity_ratio_histogram.png",
    "track_length_histogram.png",
    "residual_histogram.png",
    "belt_map_coverage.png",
    "overlay_contact_sheet.png",
}

GENERATED_OUTPUT_PATTERNS = (
    "residual_frame_*.png",
    "residual_fixed_frame_*.png",
    "raw_frame_*.png",
    "local_illumination_field_frame_*.png",
    "detections_overlay_sample_*.png",
    "tracks_overlay_sample_*.png",
)


def refresh_runtime_paths() -> None:
    global DATA, OUT, START_TIME
    DATA = Path(os.getenv("BELTMAP_IMAGE_DIR", "data/images"))
    OUT = Path(os.getenv("BELTMAP_OUTPUT_DIR", "outputs"))
    START_TIME = time.perf_counter()


def _safe_resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def clear_generated_outputs(*, protected_paths: Iterable[Path] = ()) -> None:
    """Remove stale BeltMap-generated files from ``OUT`` before starting a run."""

    if not OUT.exists():
        return
    if not OUT.is_dir():
        raise ValueError(f"BELTMAP_OUTPUT_DIR is not a directory: {OUT}")

    protected = {_safe_resolve(path) for path in protected_paths}

    def unlink_if_generated(path: Path) -> None:
        if _safe_resolve(path) in protected:
            return
        if path.is_file() or path.is_symlink():
            path.unlink()

    for name in GENERATED_OUTPUT_FILES:
        unlink_if_generated(OUT / name)
    for pattern in GENERATED_OUTPUT_PATTERNS:
        for path in OUT.glob(pattern):
            unlink_if_generated(path)


def env_int(name: str, default: int, minimum: int | None = None) -> int:
    value = os.getenv(name, "").strip()
    parsed = default if value == "" else int(value)
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name}={parsed} is below minimum {minimum}")
    return parsed


def env_float(name: str, default: float, minimum: float | None = None) -> float:
    value = os.getenv(name, "").strip()
    parsed = default if value == "" else float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite, got {parsed!r}")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name}={parsed} is below minimum {minimum}")
    return parsed


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "").strip().lower()
    if value == "":
        return default
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def elapsed_s() -> float:
    return time.perf_counter() - START_TIME


def _ru_maxrss_to_mib(value: float, *, platform: str | None = None) -> float:
    """Convert ``resource.ru_maxrss`` to MiB for the current platform."""

    current_platform = sys.platform if platform is None else platform
    divisor = 1024.0 ** 2 if current_platform == "darwin" else 1024.0
    return float(value) / divisor


def rss_mb() -> float | None:
    """Return peak resident set size in MiB when available on this platform."""

    try:
        import resource
        return _ru_maxrss_to_mib(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    return value


def emit(stage: str, message: str, **data: Any) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(elapsed_s(), 3),
        "stage": stage,
        "message": message,
    }
    mem = rss_mb()
    if mem is not None:
        payload["rss_mb"] = round(mem, 1)
    payload.update({k: jsonable(v) for k, v in data.items()})
    compact = {k: v for k, v in payload.items() if k not in {"timestamp", "stage", "message"}}
    print(f"[{payload['elapsed_s']:9.1f}s] {stage}: {message} {json.dumps(compact, sort_keys=True)}", flush=True)
    with (OUT / "progress.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    (OUT / "progress_latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def natural_key(path: Path) -> list[int | str]:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(path))]


def _absolute_path(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        _absolute_path(path).relative_to(_absolute_path(parent))
    except ValueError:
        return False
    return True


def _output_dir_to_exclude_from_inputs() -> Path | None:
    data_root = _absolute_path(DATA)
    output_root = _absolute_path(OUT)
    if output_root == data_root:
        raise ValueError(
            "BELTMAP_OUTPUT_DIR must not be the same directory as BELTMAP_IMAGE_DIR; "
            "otherwise generated PNG outputs can be rediscovered as input frames"
        )
    if _path_is_relative_to(output_root, data_root):
        return output_root
    return None


def image_paths() -> tuple[list[Path], int, int]:
    excluded_output_root = _output_dir_to_exclude_from_inputs()
    all_paths = sorted(
        [
            p
            for p in DATA.rglob("*")
            if p.suffix.lower() in EXTS
            and p.is_file()
            and not p.name.startswith("._")
            and not (
                excluded_output_root is not None
                and _path_is_relative_to(p, excluded_output_root)
            )
        ],
        key=natural_key,
    )
    if not all_paths:
        raise SystemExit(f"No image files found below {DATA}")
    frame_stride = env_int("FRAME_STRIDE", 1, minimum=1)
    paths = all_paths[::frame_stride]
    max_frames = env_int("MAX_FRAMES", 0, minimum=0)
    if max_frames > 0:
        paths = paths[:max_frames]
    return paths, len(all_paths), frame_stride


def read_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.float32)


def crop(frame: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
    top, left, height, width = region
    return frame[top: top + height, left: left + width]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def display_values(array: Any) -> np.ndarray:
    """Return the numeric image values used for diagnostic preview rendering."""

    arr = np.asarray(array.normalized if isinstance(array, ResidualImage) else array, dtype=np.float64)
    return arr


def save_scaled_png(array: Any, path: Path, *, scale: tuple[float, float]) -> None:
    arr = display_values(array)
    low, high = scale
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = 0.0, 1.0
    Image.fromarray(np.clip((arr - low) / (high - low) * 255, 0, 255).astype(np.uint8)).save(path)


def save_png(array: Any, path: Path) -> None:
    arr = display_values(array)
    finite = np.isfinite(arr)
    low, high = np.percentile(arr[finite], [1, 99]) if finite.any() else (0, 1)
    save_scaled_png(array, path, scale=(float(low), float(high)))
