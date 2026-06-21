from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from PIL import Image

IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
DEFAULT_FRAME_INDEX_PATTERN = r"(\d+)"
DETECTION_FIELDS = [
    "frame_index",
    "label",
    "y",
    "x",
    "area_px",
    "bbox_top",
    "bbox_left",
    "bbox_bottom",
    "bbox_right",
    "score",
    "confidence",
    "class_id",
    "source",
]
DETECTIONS_PER_FRAME_FIELDS = ["frame_index", "n_detections"]


@dataclass(frozen=True)
class YoloPrediction:
    """One normalized YOLO-format bounding box."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float


@dataclass(frozen=True)
class YoloExportSummary:
    """Summary of a YOLO-to-BeltMap detection export."""

    output_dir: Path
    images_dir: Path
    labels_dir: Path
    n_images: int
    n_label_files: int
    n_detections: int
    n_frames_with_detections: int
    frame_index_min: int | None
    frame_index_max: int | None


@dataclass(frozen=True)
class ImageRecord:
    """One source image available for YOLO prediction export."""

    path: Path
    stem: str
    frame_index: int
    width: int
    height: int


def natural_key(path: Path) -> list[int | str]:
    """Return a deterministic key that sorts embedded numbers numerically."""

    parts = re.split(r"(\d+)", str(path))
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def infer_frame_index(
    name: str,
    *,
    pattern: str = DEFAULT_FRAME_INDEX_PATTERN,
) -> int:
    """Infer a frame index from an image or label stem.

    The last regex match is used so names such as ``frame_000123_combined`` still
    map to frame 123.  The regex may contain one capture group; when no capture
    group is present, the full match is used.
    """

    regex = re.compile(pattern)
    matches = list(regex.finditer(name))
    if not matches:
        raise ValueError(f"could not infer a frame index from {name!r} using {pattern!r}")
    match = matches[-1]
    value = match.group(1) if match.groups() else match.group(0)
    return int(value)


def find_images(
    images_dir: Path,
    *,
    frame_index_pattern: str = DEFAULT_FRAME_INDEX_PATTERN,
) -> dict[str, ImageRecord]:
    """Return image records keyed by stem for all supported images below a root."""

    if not images_dir.is_dir():
        raise FileNotFoundError(images_dir)

    records: dict[str, ImageRecord] = {}
    for path in sorted(images_dir.rglob("*"), key=natural_key):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        frame_index = infer_frame_index(path.stem, pattern=frame_index_pattern)
        with Image.open(path) as image:
            width, height = image.size
        records[path.stem] = ImageRecord(
            path=path,
            stem=path.stem,
            frame_index=frame_index,
            width=int(width),
            height=int(height),
        )
    if not records:
        raise ValueError(f"no supported images found below {images_dir}")
    return records


def parse_yolo_label_line(
    line: str,
    *,
    default_confidence: float = 1.0,
) -> YoloPrediction | None:
    """Parse one YOLO label line.

    The expected format is ``class x_center y_center width height [confidence]``
    with normalized center-size coordinates.  Blank lines return ``None``.
    """

    stripped = line.strip()
    if not stripped:
        return None
    parts = stripped.split()
    if len(parts) not in (5, 6):
        raise ValueError(
            "YOLO prediction lines must contain 5 or 6 values: "
            "class x_center y_center width height [confidence]"
        )
    class_id_float = float(parts[0])
    if not class_id_float.is_integer():
        raise ValueError(f"YOLO class id must be an integer, got {parts[0]!r}")
    x_center, y_center, width, height = (float(value) for value in parts[1:5])
    confidence = float(parts[5]) if len(parts) == 6 else float(default_confidence)
    values = (x_center, y_center, width, height, confidence)
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"non-finite YOLO prediction value in line {line!r}")
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"YOLO width/height must be positive in line {line!r}")
    return YoloPrediction(
        class_id=int(class_id_float),
        x_center=x_center,
        y_center=y_center,
        width=width,
        height=height,
        confidence=confidence,
    )


def load_yolo_label_file(
    path: Path,
    *,
    default_confidence: float = 1.0,
) -> list[YoloPrediction]:
    """Load all YOLO predictions from one text file."""

    predictions: list[YoloPrediction] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        prediction = parse_yolo_label_line(line, default_confidence=default_confidence)
        if prediction is not None:
            predictions.append(prediction)
    return predictions


def yolo_prediction_to_detection_row(
    prediction: YoloPrediction,
    *,
    image: ImageRecord,
    label_index: int,
    source: str,
) -> dict[str, str]:
    """Convert one normalized YOLO prediction to a BeltMap-style detection row."""

    left = (prediction.x_center - 0.5 * prediction.width) * image.width
    right = (prediction.x_center + 0.5 * prediction.width) * image.width
    top = (prediction.y_center - 0.5 * prediction.height) * image.height
    bottom = (prediction.y_center + 0.5 * prediction.height) * image.height
    left = min(max(left, 0.0), float(image.width))
    right = min(max(right, 0.0), float(image.width))
    top = min(max(top, 0.0), float(image.height))
    bottom = min(max(bottom, 0.0), float(image.height))
    if right <= left or bottom <= top:
        raise ValueError(f"YOLO prediction clips to an empty box on frame {image.frame_index}")

    x = 0.5 * (left + right)
    y = 0.5 * (top + bottom)
    area = (right - left) * (bottom - top)
    confidence = float(prediction.confidence)
    return {
        "frame_index": str(image.frame_index),
        "label": str(label_index),
        "y": f"{y:.6f}",
        "x": f"{x:.6f}",
        "area_px": str(int(round(area))),
        "bbox_top": str(int(math.floor(top))),
        "bbox_left": str(int(math.floor(left))),
        "bbox_bottom": str(int(math.ceil(bottom))),
        "bbox_right": str(int(math.ceil(right))),
        "score": f"{confidence:.8f}",
        "confidence": f"{confidence:.8f}",
        "class_id": str(prediction.class_id),
        "source": source,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _label_files(labels_dir: Path) -> dict[str, Path]:
    if not labels_dir.is_dir():
        raise FileNotFoundError(labels_dir)
    files: dict[str, Path] = {}
    for path in sorted(labels_dir.rglob("*.txt"), key=natural_key):
        files[path.stem] = path
    return files


def export_yolo_predictions_to_beltmap_run(
    *,
    labels_dir: Path,
    images_dir: Path,
    output_dir: Path,
    frame_index_pattern: str = DEFAULT_FRAME_INDEX_PATTERN,
    default_confidence: float = 1.0,
    allow_label_without_image: bool = False,
    source: str = "yolo",
) -> YoloExportSummary:
    """Write BeltMap-compatible CSV outputs from YOLO prediction text files."""

    images = find_images(images_dir, frame_index_pattern=frame_index_pattern)
    labels = _label_files(labels_dir)
    missing_images = sorted(set(labels) - set(images))
    if missing_images and not allow_label_without_image:
        preview = ", ".join(missing_images[:5])
        raise ValueError(
            f"{len(missing_images)} YOLO label file(s) have no matching image stem; "
            f"first missing: {preview}"
        )

    detections: list[dict[str, str]] = []
    per_frame: list[dict[str, object]] = []
    frames_with_detections = 0
    for image in sorted(images.values(), key=lambda item: item.frame_index):
        label_path = labels.get(image.stem)
        predictions = (
            []
            if label_path is None
            else load_yolo_label_file(label_path, default_confidence=default_confidence)
        )
        if predictions:
            frames_with_detections += 1
        for label_index, prediction in enumerate(predictions, start=1):
            detections.append(
                yolo_prediction_to_detection_row(
                    prediction,
                    image=image,
                    label_index=label_index,
                    source=source,
                )
            )
        per_frame.append(
            {
                "frame_index": image.frame_index,
                "n_detections": len(predictions),
            }
        )

    detections.sort(key=lambda row: (int(row["frame_index"]), int(row["label"])))
    _write_csv(output_dir / "detections.csv", detections, DETECTION_FIELDS)
    _write_csv(output_dir / "detections_per_frame.csv", per_frame, DETECTIONS_PER_FRAME_FIELDS)

    frame_indices = [image.frame_index for image in images.values()]
    summary = YoloExportSummary(
        output_dir=output_dir,
        images_dir=images_dir,
        labels_dir=labels_dir,
        n_images=len(images),
        n_label_files=len(labels),
        n_detections=len(detections),
        n_frames_with_detections=frames_with_detections,
        frame_index_min=min(frame_indices) if frame_indices else None,
        frame_index_max=max(frame_indices) if frame_indices else None,
    )
    metadata = {
        "mode": "yolo_export",
        "source": source,
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "n_images": summary.n_images,
        "n_label_files": summary.n_label_files,
        "n_detections": summary.n_detections,
        "n_frames_with_detections": summary.n_frames_with_detections,
        "frame_index_min": summary.frame_index_min,
        "frame_index_max": summary.frame_index_max,
        "frame_index_pattern": frame_index_pattern,
        "default_confidence": default_confidence,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output_dir / "config_resolved.json").write_text(
        json.dumps({"mode": "yolo_export", "detection": {"score_field": "confidence"}}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return summary
