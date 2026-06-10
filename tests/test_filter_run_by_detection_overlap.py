import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts import filter_run_by_detection_overlap as overlap_filter


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_run(output_dir: Path, detections: list[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    n_images = max(int(row["frame_index"]) for row in detections) + 1 if detections else 0
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "n_images": n_images,
                "belt_velocity_px_per_frame": 5.0,
                "n_detections": len(detections),
            }
        ),
        encoding="utf-8",
    )
    fields = [
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
    write_csv(output_dir / "detections.csv", detections, fields)
    counts = {
        index: sum(1 for row in detections if int(row["frame_index"]) == index)
        for index in range(n_images)
    }
    write_csv(
        output_dir / "detections_per_frame.csv",
        [{"frame_index": index, "n_detections": counts[index]} for index in range(n_images)],
    )
    Image.new("L", (16, 16), 96).save(output_dir / "residual_frame_000000.png")


def detection(
    label: int,
    *,
    x: float,
    y: float,
    frame_index: int = 0,
    area: int = 9,
    peak: float = 9.0,
) -> dict[str, object]:
    return {
        "frame_index": frame_index,
        "image": f"frame{frame_index}.bmp",
        "label": label,
        "y": y,
        "x": x,
        "area_px": area,
        "bbox_top": y - 1,
        "bbox_left": x - 1,
        "bbox_bottom": y + 2,
        "bbox_right": x + 2,
        "mean_signal": 6.0,
        "peak_signal": peak,
        "recurrent_artifact_overlap_fraction": "",
        "recurrent_artifact_probability": "",
        "recurrent_artifact_required_peak_signal": "",
    }


def test_filter_run_keeps_only_detections_confirmed_by_overlap(tmp_path):
    candidate = tmp_path / "candidate"
    confirming = tmp_path / "confirming"
    output = tmp_path / "confirmed"
    make_run(candidate, [detection(1, x=10, y=10), detection(2, x=40, y=40)])
    make_run(confirming, [detection(1, x=10, y=10)])

    exit_code = overlap_filter.main(
        [
            "--candidate-run",
            str(candidate),
            "--confirming-run",
            str(confirming),
            "--output-dir",
            str(output),
            "--min-iou",
            "0.1",
            "--max-center-distance-px",
            "0",
            "--quiet",
        ]
    )

    assert exit_code == 0
    kept = list(csv.DictReader((output / "detections.csv").open(newline="", encoding="utf-8")))
    rejected = list(csv.DictReader((output / "rejected_detections.csv").open(newline="", encoding="utf-8")))
    assert [row["label"] for row in kept] == ["1"]
    assert [row["label"] for row in rejected] == ["2"]
    assert kept[0]["confirmation_label"] == "1"
    assert (output / "residual_frame_000000.png").is_file()
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["n_detections"] == 1
    assert metadata["confirmation"]["rejected_detections"] == 1


def test_filter_run_rejects_fractional_candidate_frame_indices(tmp_path, capsys):
    candidate = tmp_path / "candidate"
    confirming = tmp_path / "confirming"
    output = tmp_path / "confirmed"
    make_run(candidate, [detection(1, x=10, y=10)])
    make_run(confirming, [detection(1, x=10, y=10)])
    bad_row = detection(1, x=10, y=10)
    bad_row["frame_index"] = "0.5"
    write_csv(candidate / "detections.csv", [bad_row], overlap_filter.DETECTION_FIELDS)

    with pytest.raises(SystemExit) as exc_info:
        overlap_filter.main(
            [
                "--candidate-run",
                str(candidate),
                "--confirming-run",
                str(confirming),
                "--output-dir",
                str(output),
                "--quiet",
            ]
        )

    assert exc_info.value.code == 2
    assert "frame_index must be an integer-like value" in capsys.readouterr().err


def test_filter_run_can_reject_distant_iou_confirmation(tmp_path):
    candidate = tmp_path / "candidate"
    confirming = tmp_path / "confirming"
    output = tmp_path / "confirmed"
    candidate_row = detection(1, x=10, y=10)
    confirming_row = detection(1, x=80, y=80)
    for row in (candidate_row, confirming_row):
        row["bbox_top"] = 0
        row["bbox_left"] = 0
        row["bbox_bottom"] = 100
        row["bbox_right"] = 100
    make_run(candidate, [candidate_row])
    make_run(confirming, [confirming_row])

    exit_code = overlap_filter.main(
        [
            "--candidate-run",
            str(candidate),
            "--confirming-run",
            str(confirming),
            "--output-dir",
            str(output),
            "--min-iou",
            "0.5",
            "--max-center-distance-px",
            "0",
            "--max-match-center-distance-px",
            "50",
            "--quiet",
        ]
    )

    assert exit_code == 0
    kept = list(csv.DictReader((output / "detections.csv").open(newline="", encoding="utf-8")))
    rejected = list(csv.DictReader((output / "rejected_detections.csv").open(newline="", encoding="utf-8")))
    assert kept == []
    assert [row["label"] for row in rejected] == ["1"]
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["confirmation"]["max_match_center_distance_px"] == 50.0


def test_filter_run_can_rescue_unmatched_detections_from_confirmed_tracks(tmp_path):
    candidate = tmp_path / "candidate"
    confirming = tmp_path / "confirming"
    output = tmp_path / "confirmed"
    candidate_detections = [
        detection(1, x=10, y=10, frame_index=0),
        detection(1, x=15, y=10, frame_index=1),
        detection(1, x=20, y=10, frame_index=2),
    ]
    make_run(candidate, candidate_detections)
    make_run(
        confirming,
        [
            detection(1, x=10, y=10, frame_index=0),
            detection(1, x=15, y=10, frame_index=1),
        ],
    )
    write_csv(
        candidate / "tracks.csv",
        [
            {"track_id": 7, "track_detection_index": index, **row}
            for index, row in enumerate(candidate_detections)
        ],
        ["track_id", "track_detection_index", *overlap_filter.DETECTION_FIELDS],
    )

    exit_code = overlap_filter.main(
        [
            "--candidate-run",
            str(candidate),
            "--confirming-run",
            str(confirming),
            "--output-dir",
            str(output),
            "--min-iou",
            "0.1",
            "--max-center-distance-px",
            "0",
            "--track-rescue-min-detections",
            "3",
            "--track-rescue-min-confirmed",
            "2",
            "--track-rescue-min-confirmed-fraction",
            "0.5",
            "--quiet",
        ]
    )

    assert exit_code == 0
    kept = list(csv.DictReader((output / "detections.csv").open(newline="", encoding="utf-8")))
    assert len(kept) == 3
    assert kept[-1]["confirmation_rescued_by_track"] == "True"
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["confirmation"]["track_rescued_detections"] == 1


def test_filter_run_can_confirm_against_filtered_tracks(tmp_path):
    candidate = tmp_path / "candidate"
    confirming = tmp_path / "confirming"
    output = tmp_path / "confirmed"
    make_run(candidate, [detection(1, x=10, y=10), detection(2, x=40, y=40)])
    confirming_detections = [detection(1, x=10, y=10), detection(2, x=40, y=40)]
    make_run(confirming, confirming_detections)
    write_csv(
        confirming / "filtered_tracks.csv",
        [{"track_id": 1, "track_detection_index": 0, **confirming_detections[0]}],
        ["track_id", "track_detection_index", *overlap_filter.DETECTION_FIELDS],
    )

    exit_code = overlap_filter.main(
        [
            "--candidate-run",
            str(candidate),
            "--confirming-run",
            str(confirming),
            "--confirming-detections-source",
            "filtered_tracks",
            "--output-dir",
            str(output),
            "--min-iou",
            "0.1",
            "--max-center-distance-px",
            "0",
            "--quiet",
        ]
    )

    assert exit_code == 0
    kept = list(csv.DictReader((output / "detections.csv").open(newline="", encoding="utf-8")))
    assert [row["label"] for row in kept] == ["1"]
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["confirmation"]["confirming_detections_source"] == "filtered_tracks"
    assert metadata["confirmation"]["n_confirming_detections"] == 1


def test_filter_run_can_limit_track_rescue_to_nearby_confirmed_detections(tmp_path):
    candidate = tmp_path / "candidate"
    confirming = tmp_path / "confirming"
    output = tmp_path / "confirmed"
    candidate_detections = [
        detection(1, x=10, y=10, frame_index=0),
        detection(1, x=15, y=10, frame_index=1),
        detection(1, x=20, y=10, frame_index=2),
        detection(1, x=25, y=10, frame_index=3),
    ]
    make_run(candidate, candidate_detections)
    make_run(confirming, [detection(1, x=10, y=10, frame_index=0)])
    write_csv(
        candidate / "tracks.csv",
        [
            {"track_id": 7, "track_detection_index": index, **row}
            for index, row in enumerate(candidate_detections)
        ],
        ["track_id", "track_detection_index", *overlap_filter.DETECTION_FIELDS],
    )

    exit_code = overlap_filter.main(
        [
            "--candidate-run",
            str(candidate),
            "--confirming-run",
            str(confirming),
            "--output-dir",
            str(output),
            "--min-iou",
            "0.1",
            "--max-center-distance-px",
            "0",
            "--track-rescue-min-detections",
            "3",
            "--track-rescue-min-confirmed",
            "1",
            "--track-rescue-min-confirmed-fraction",
            "0.1",
            "--track-rescue-max-frame-distance",
            "1",
            "--quiet",
        ]
    )

    assert exit_code == 0
    kept = list(csv.DictReader((output / "detections.csv").open(newline="", encoding="utf-8")))
    rejected = list(csv.DictReader((output / "rejected_detections.csv").open(newline="", encoding="utf-8")))
    assert [row["frame_index"] for row in kept] == ["0", "1"]
    assert [row["frame_index"] for row in rejected] == ["2", "3"]
    assert kept[-1]["confirmation_rescued_by_track"] == "True"
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["confirmation"]["track_rescue_max_frame_distance"] == 1.0
    assert metadata["confirmation"]["track_rescued_detections"] == 1


def test_filter_run_can_gate_track_rescued_detections_by_quality(tmp_path):
    candidate = tmp_path / "candidate"
    confirming = tmp_path / "confirming"
    output = tmp_path / "confirmed"
    candidate_detections = [
        detection(1, x=10, y=10, frame_index=0, area=16, peak=8.0),
        detection(1, x=15, y=10, frame_index=1, area=4, peak=5.2),
        detection(1, x=20, y=10, frame_index=2, area=9, peak=6.0),
    ]
    make_run(candidate, candidate_detections)
    make_run(confirming, [detection(1, x=10, y=10, frame_index=0, area=16, peak=8.0)])
    write_csv(
        candidate / "tracks.csv",
        [
            {"track_id": 7, "track_detection_index": index, **row}
            for index, row in enumerate(candidate_detections)
        ],
        ["track_id", "track_detection_index", *overlap_filter.DETECTION_FIELDS],
    )

    exit_code = overlap_filter.main(
        [
            "--candidate-run",
            str(candidate),
            "--confirming-run",
            str(confirming),
            "--output-dir",
            str(output),
            "--min-iou",
            "0.1",
            "--max-center-distance-px",
            "0",
            "--track-rescue-min-detections",
            "3",
            "--track-rescue-min-confirmed",
            "1",
            "--track-rescue-min-confirmed-fraction",
            "0.1",
            "--rescued-min-area-px",
            "6",
            "--quiet",
        ]
    )

    assert exit_code == 0
    kept = list(csv.DictReader((output / "detections.csv").open(newline="", encoding="utf-8")))
    rejected = list(csv.DictReader((output / "rejected_detections.csv").open(newline="", encoding="utf-8")))
    assert [row["frame_index"] for row in kept] == ["0", "2"]
    assert [row["frame_index"] for row in rejected] == ["1"]
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["confirmation"]["rescued_min_area_px"] == 6.0
    assert metadata["confirmation"]["rescued_quality_rejected_detections"] == 1
    assert metadata["confirmation"]["track_rescued_detections"] == 1


def test_filter_run_can_require_direct_confirmation_in_final_tracks(tmp_path):
    candidate = tmp_path / "candidate"
    confirming = tmp_path / "confirming"
    output = tmp_path / "confirmed"
    candidate_detections = [
        detection(1, x=10, y=10, frame_index=0),
        detection(1, x=10, y=15, frame_index=1),
        detection(1, x=10, y=20, frame_index=2),
    ]
    make_run(candidate, candidate_detections)
    make_run(confirming, [detection(1, x=10, y=10, frame_index=0)])
    write_csv(
        candidate / "tracks.csv",
        [
            {"track_id": 7, "track_detection_index": index, **row}
            for index, row in enumerate(candidate_detections)
        ],
        ["track_id", "track_detection_index", *overlap_filter.DETECTION_FIELDS],
    )

    exit_code = overlap_filter.main(
        [
            "--candidate-run",
            str(candidate),
            "--confirming-run",
            str(confirming),
            "--output-dir",
            str(output),
            "--min-iou",
            "0.1",
            "--max-center-distance-px",
            "0",
            "--track-rescue-min-detections",
            "3",
            "--track-rescue-min-confirmed",
            "1",
            "--track-rescue-min-confirmed-fraction",
            "0.1",
            "--min-track-length",
            "2",
            "--track-filter-min-length",
            "2",
            "--track-filter-min-confirmed-detections",
            "2",
            "--quiet",
        ]
    )

    assert exit_code == 0
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["n_velocity_accepted_before_confirmation_filter"] == 1
    assert metadata["n_confirmation_filtered_velocity_estimates"] == 1
    assert metadata["n_filtered_velocity_estimates"] == 0
    score_rows = list(csv.DictReader((output / "track_scores.csv").open(newline="", encoding="utf-8")))
    assert score_rows[0]["direct_confirmed_detections"] == "1"
    assert score_rows[0]["passes_confirmation"] == "False"
    assert score_rows[0]["accepted"] == "False"


def test_filter_run_can_use_one_to_one_confirmation(tmp_path):
    candidate = tmp_path / "candidate"
    confirming = tmp_path / "confirming"
    output = tmp_path / "confirmed"
    make_run(candidate, [detection(1, x=10, y=10), detection(2, x=12, y=10)])
    make_run(confirming, [detection(1, x=10, y=10)])

    exit_code = overlap_filter.main(
        [
            "--candidate-run",
            str(candidate),
            "--confirming-run",
            str(confirming),
            "--output-dir",
            str(output),
            "--min-iou",
            "0.01",
            "--max-center-distance-px",
            "10",
            "--one-to-one-confirmation",
            "--disable-track-rescue",
            "--quiet",
        ]
    )

    assert exit_code == 0
    kept = list(csv.DictReader((output / "detections.csv").open(newline="", encoding="utf-8")))
    rejected = list(csv.DictReader((output / "rejected_detections.csv").open(newline="", encoding="utf-8")))
    assert [row["label"] for row in kept] == ["1"]
    assert [row["label"] for row in rejected] == ["2"]


def test_filter_run_allows_many_to_one_confirmation_by_default(tmp_path):
    candidate = tmp_path / "candidate"
    confirming = tmp_path / "confirming"
    output = tmp_path / "confirmed"
    make_run(candidate, [detection(1, x=10, y=10), detection(2, x=12, y=10)])
    make_run(confirming, [detection(1, x=10, y=10)])

    exit_code = overlap_filter.main(
        [
            "--candidate-run",
            str(candidate),
            "--confirming-run",
            str(confirming),
            "--output-dir",
            str(output),
            "--min-iou",
            "0.01",
            "--max-center-distance-px",
            "10",
            "--disable-track-rescue",
            "--quiet",
        ]
    )

    assert exit_code == 0
    kept = list(csv.DictReader((output / "detections.csv").open(newline="", encoding="utf-8")))
    assert [row["label"] for row in kept] == ["1", "2"]


def test_filter_run_can_suppress_duplicate_confirmed_detections(tmp_path):
    candidate = tmp_path / "candidate"
    confirming = tmp_path / "confirming"
    output = tmp_path / "confirmed"
    smaller = detection(1, x=10, y=10)
    larger = detection(2, x=10.5, y=10.5)
    larger["area_px"] = 16
    larger["bbox_top"] = 8.5
    larger["bbox_left"] = 8.5
    larger["bbox_bottom"] = 13.5
    larger["bbox_right"] = 13.5
    larger["peak_signal"] = 12.0
    make_run(candidate, [smaller, larger])
    make_run(confirming, [detection(1, x=10, y=10)])

    exit_code = overlap_filter.main(
        [
            "--candidate-run",
            str(candidate),
            "--confirming-run",
            str(confirming),
            "--output-dir",
            str(output),
            "--min-iou",
            "0.01",
            "--max-center-distance-px",
            "10",
            "--dedupe-iou-threshold",
            "0.1",
            "--dedupe-margin-px",
            "1",
            "--disable-track-rescue",
            "--quiet",
        ]
    )

    assert exit_code == 0
    kept = list(csv.DictReader((output / "detections.csv").open(newline="", encoding="utf-8")))
    rejected = list(csv.DictReader((output / "rejected_detections.csv").open(newline="", encoding="utf-8")))
    assert [row["label"] for row in kept] == ["2"]
    assert [row["label"] for row in rejected] == ["1"]
    assert rejected[0]["confirmation_duplicate_suppressed"] == "True"
    assert rejected[0]["confirmation_duplicate_kept_label"] == "2"
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["confirmation"]["duplicate_suppressed_detections"] == 1
    assert metadata["confirmation"]["rejected_detections"] == 1
