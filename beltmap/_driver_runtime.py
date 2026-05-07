"""Runtime helpers for the packaged BeltMap image driver."""

from __future__ import annotations

import csv
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .residual import ResidualImage

DATA = Path(os.getenv("BELTMAP_IMAGE_DIR", "data/images"))
OUT = Path(os.getenv("BELTMAP_OUTPUT_DIR", "outputs"))
EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
START_TIME = time.perf_counter()


def refresh_runtime_paths() -> None:
    global DATA, OUT, START_TIME
    DATA = Path(os.getenv("BELTMAP_IMAGE_DIR", "data/images"))
    OUT = Path(os.getenv("BELTMAP_OUTPUT_DIR", "outputs"))
    START_TIME = time.perf_counter()


def env_int(name: str, default: int, minimum: int | None = None) -> int:
    value = os.getenv(name, "").strip()
    parsed = default if value == "" else int(value)
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name}={parsed} is below minimum {minimum}")
    return parsed


def env_float(name: str, default: float, minimum: float | None = None) -> float:
    value = os.getenv(name, "").strip()
    parsed = default if value == "" else float(value)
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
    payload.update({k: jsonable(v) for k, v in data.items()})
    compact = {k: v for k, v in payload.items() if k not in {"timestamp", "stage", "message"}}
    print(f"[{payload['elapsed_s']:9.1f}s] {stage}: {message} {json.dumps(compact, sort_keys=True)}", flush=True)
    with (OUT / "progress.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    (OUT / "progress_latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def natural_key(path: Path) -> list[int | str]:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(path))]


def image_paths() -> tuple[list[Path], int, int]:
    all_paths = sorted(
        [p for p in DATA.rglob("*") if p.suffix.lower() in EXTS and not p.name.startswith("._")],
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


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_png(array: Any, path: Path) -> None:
    arr = np.asarray(array.normalized if isinstance(array, ResidualImage) else array, dtype=np.float64)
    finite = np.isfinite(arr)
    low, high = np.percentile(arr[finite], [1, 99]) if finite.any() else (0, 1)
    if high <= low:
        high = low + 1
    Image.fromarray(np.clip((arr - low) / (high - low) * 255, 0, 255).astype(np.uint8)).save(path)
