from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from beltmap.cli import filter_recurrent_artifacts as fra
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
TRACK_DETECTION_FIELDS = ["track_id", "track_detection_index", *DETECTION_FIELDS]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def detection_row(**updates):
    row = {
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
    }
    row.update(updates)
    return row


def test_group_detections_rejects_fractional_frame_indices():
    with pytest.raises(ValueError, match="frame_index must be an integer-valued field"):
        fra.group_detections([detection_row(frame_index="0.5")], frame_count=1)


def test_load_phase_px_by_frame_does_not_truncate_fractional_frames(tmp_path):
    phase_path = tmp_path / "phase_estimates.csv"
    write_csv(
        phase_path,
        [
            {"frame_index": "0.5", "phase_px": 2.0},
        ],
        ["frame_index", "phase_px"],
    )

    with pytest.raises(ValueError, match="missing 1 frames"):
        fra.load_phase_px_by_frame(phase_path, frame_count=1)


def test_infer_region_rejects_fractional_metadata_dimensions():
    with pytest.raises(ValueError, match="belt_region.height must be an integer"):
        fra.infer_region(
            {"belt_region": {"top": 0, "left": 0, "height": "5.5", "width": 10}},
            {},
        )


def test_int_option_rejects_fractional_config_values():
    with pytest.raises(ValueError, match="min_track_length must be an integer"):
        fra.int_option(
            {"options": {"min_track_length": {"value": "2.5"}}},
            "min_track_length",
            2,
        )


def test_cli_preserves_explicit_zero_track_filter_min_length(tmp_path, monkeypatch):
    captured = {}

    def fake_filter_recurrent_artifacts(**kwargs):
        captured["track_filter_config"] = kwargs["track_filter_config"]
        return {}

    monkeypatch.setattr(fra, "filter_recurrent_artifacts", fake_filter_recurrent_artifacts)

    assert main(
        [
            "--input-dir",
            str(tmp_path / "source"),
            "--output-dir",
            str(tmp_path / "filtered"),
            "--track-filter-min-length",
            "0",
            "--quiet",
        ]
    ) == 0

    assert captured["track_filter_config"].min_track_length == 0


def write_minimal_filter_input(path: Path, metadata_updates: dict[str, object]) -> None:
    path.mkdir()
    metadata = {
        "n_images": 0,
        "belt_region": {"top": 0, "left": 0, "height": 5, "width": 5},
        "belt_velocity_px_per_frame": 10.0,
        "belt_period_px_input": 10,
        "belt_map_height_px": 10,
        "reference_phase_px": 0.0,
        "detection_threshold": 5.0,
    }
    metadata.update(metadata_updates)
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    write_csv(path / "detections.csv", [], DETECTION_FIELDS)


@pytest.mark.parametrize(
    ("metadata_updates", "message"),
    [
        ({"belt_map_height_px": 0}, "belt_map_height_px must be positive"),
        ({"belt_period_px_input": 0}, "belt_period_px_input must be positive"),
    ],
)
def test_filter_recurrent_artifacts_rejects_explicit_zero_geometry_metadata(
    tmp_path,
    metadata_updates,
    message,
):
    input_dir = tmp_path / "source"
    write_minimal_filter_input(input_dir, metadata_updates)

    with pytest.raises(ValueError, match=message):
        fra.filter_recurrent_artifacts(
            input_dir=input_dir,
            output_dir=tmp_path / "filtered",
            recurrent_config=fra.RecurrentArtifactConfig(),
            min_track_length=None,
            velocity_fit_method=None,
            track_filter_config=None,
        )


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
    write_csv(
        input_dir / "filtered_tracks.csv",
        [
            {
                "track_id": 1,
                "track_detection_index": 0,
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
            }
        ],
        TRACK_DETECTION_FIELDS,
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

    protected_output_dir = tmp_path / "filtered_protected"
    assert main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(protected_output_dir),
            "--min-revolutions",
            "2",
            "--margin-px",
            "0",
            "--protect-source-filtered-tracks",
            "--quiet",
        ]
    ) == 0

    protected_metadata = json.loads(
        (protected_output_dir / "metadata.json").read_text(encoding="utf-8")
    )
    assert protected_metadata["n_recurrent_artifact_rejected"] == 1
    assert protected_metadata["n_recurrent_artifact_protected_by_source_track"] == 1
    assert protected_metadata["n_detections"] == 2

    with (protected_output_dir / "recurrent_artifact_detections.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        protected_recurrent_rows = list(csv.DictReader(handle))
    assert [
        row["recurrent_artifact_protected_by_source_track"]
        for row in protected_recurrent_rows
    ] == ["True", "False", "False"]
    assert [row["recurrent_artifact_rejected"] for row in protected_recurrent_rows] == [
        "False",
        "True",
        "False",
    ]
