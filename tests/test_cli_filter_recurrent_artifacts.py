from __future__ import annotations

import csv
import json
from pathlib import Path

from beltmap.cli.filter_recurrent_artifacts import main


DETECTION_FIELDS = [
    "frame_index",
    "image",
    "label",
    "y",
    "x",
    "area_px",
    "bbox_top",
    "bbox_left",
    "bbox_bottom",
    "bbox_right",
    "mean_signal",
    "peak_signal",
    "recurrent_artifact_overlap_fraction",
    "recurrent_artifact_probability",
    "recurrent_artifact_required_peak_signal",
]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_postrun_recurrent_artifact_filter_rejects_cross_revolution_hits(tmp_path):
    input_dir = tmp_path / "source"
    output_dir = tmp_path / "filtered"
    input_dir.mkdir()
    (input_dir / "metadata.json").write_text(
        json.dumps(
            {
                "n_images": 3,
                "belt_region": {"top": 0, "left": 0, "height": 5, "width": 5},
                "belt_velocity_px_per_frame": 10.0,
                "belt_period_px_input": 10,
                "belt_map_height_px": 10,
                "reference_phase_px": 0.0,
                "detection_threshold": 5.0,
            }
        ),
        encoding="utf-8",
    )
    write_csv(
        input_dir / "phase_estimates.csv",
        [
            {
                "frame_index": index,
                "image": f"frame_{index}.png",
                "phase_px": 0.0,
                "phase_fraction": 0.0,
                "phase_rad": 0.0,
                "predicted_phase_px": 0.0,
                "correction_px": 0.0,
                "phase_drift_px": 0.0,
                "loss": "",
                "score": "",
                "method": "test",
            }
            for index in range(3)
        ],
        [
            "frame_index",
            "image",
            "phase_px",
            "phase_fraction",
            "phase_rad",
            "predicted_phase_px",
            "correction_px",
            "phase_drift_px",
            "loss",
            "score",
            "method",
        ],
    )
    write_csv(
        input_dir / "detections.csv",
        [
            {
                "frame_index": 0,
                "image": "frame_0.png",
                "label": 1,
                "y": 1.5,
                "x": 1.5,
                "area_px": 1,
                "bbox_top": 1,
                "bbox_left": 1,
                "bbox_bottom": 2,
                "bbox_right": 2,
                "mean_signal": 6.0,
                "peak_signal": 6.0,
                "recurrent_artifact_overlap_fraction": "",
                "recurrent_artifact_probability": "",
                "recurrent_artifact_required_peak_signal": "",
            },
            {
                "frame_index": 1,
                "image": "frame_1.png",
                "label": 1,
                "y": 1.5,
                "x": 1.5,
                "area_px": 1,
                "bbox_top": 1,
                "bbox_left": 1,
                "bbox_bottom": 2,
                "bbox_right": 2,
                "mean_signal": 6.0,
                "peak_signal": 6.0,
                "recurrent_artifact_overlap_fraction": "",
                "recurrent_artifact_probability": "",
                "recurrent_artifact_required_peak_signal": "",
            },
            {
                "frame_index": 2,
                "image": "frame_2.png",
                "label": 1,
                "y": 3.5,
                "x": 3.5,
                "area_px": 1,
                "bbox_top": 3,
                "bbox_left": 3,
                "bbox_bottom": 4,
                "bbox_right": 4,
                "mean_signal": 6.0,
                "peak_signal": 6.0,
                "recurrent_artifact_overlap_fraction": "",
                "recurrent_artifact_probability": "",
                "recurrent_artifact_required_peak_signal": "",
            },
        ],
        DETECTION_FIELDS,
    )

    assert main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--min-revolutions",
            "2",
            "--margin-px",
            "0",
            "--reject-max-area-px",
            "1",
            "--reject-max-peak-signal",
            "6",
            "--quiet",
        ]
    ) == 0

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["n_source_detections"] == 3
    assert metadata["n_recurrent_artifact_rejected"] == 2
    assert metadata["n_detections"] == 1
    assert metadata["recurrent_artifact_reject_max_area_px"] == 1
    assert metadata["recurrent_artifact_reject_max_peak_signal"] == 6.0

    with (output_dir / "recurrent_artifact_detections.csv").open(newline="", encoding="utf-8") as handle:
        recurrent_rows = list(csv.DictReader(handle))
    assert [row["recurrent_artifact_rejected"] for row in recurrent_rows] == [
        "True",
        "True",
        "False",
    ]

    with (output_dir / "detections.csv").open(newline="", encoding="utf-8") as handle:
        kept_rows = list(csv.DictReader(handle))
    assert [row["frame_index"] for row in kept_rows] == ["2"]
