from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from beltmap.cli.yolo_recurrence_filter import main as yolo_recurrence_main
from beltmap.rendering import BeltRegion
from beltmap.yolo_recurrence import (
    YoloRecurrenceConfig,
    find_revisit,
    load_phase_px_by_frame,
    run_yolo_recurrence_filter,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source_dir = tmp_path / "images"
    yolo_run = tmp_path / "yolo_run"
    reference = tmp_path / "beltmap_reference"
    truth_path = tmp_path / "truth.json"
    source_dir.mkdir()
    yolo_run.mkdir()
    reference.mkdir()

    belt_map = np.full((20, 12), 50.0, dtype=np.float32)
    np.save(reference / "belt_map.npy", belt_map)
    phase_rows = []
    for frame, phase in enumerate([0, 10, 0, 10, 0, 10]):
        phase_rows.append(
            {
                "frame_index": frame,
                "image": f"frame_{frame:06d}.png",
                "phase_px": phase,
                "predicted_phase_px": phase,
                "correction_px": 0,
                "score": 1,
                "method": "test",
            }
        )
    write_csv(
        reference / "phase_estimates.csv",
        phase_rows,
        ["frame_index", "image", "phase_px", "predicted_phase_px", "correction_px", "score", "method"],
    )
    (reference / "metadata.json").write_text(
        json.dumps(
            {
                "n_images": 6,
                "belt_velocity_px_per_frame": 10.0,
                "belt_period_px_input": 20.0,
                "belt_map_height_px": 20,
                "reference_phase_px": 0.0,
                "belt_region": {"top": 0, "left": 0, "height": 10, "width": 12},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_csv(
        reference / "detections.csv",
        [],
        [
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
        ],
    )
    write_csv(
        reference / "detections_per_frame.csv",
        [{"frame_index": frame, "n_detections": 0} for frame in range(6)],
        ["frame_index", "n_detections"],
    )

    for frame in range(6):
        image = np.full((10, 12), 50, dtype=np.uint8)
        if frame in {0, 2, 4}:
            image[4:7, 5:8] = np.asarray(
                [[120, 180, 120], [180, 220, 180], [120, 180, 120]],
                dtype=np.uint8,
            )  # belt-fixed false candidate
        if frame == 2:
            image[4:7, 1:4] = np.asarray(
                [[120, 180, 120], [180, 220, 180], [120, 180, 120]],
                dtype=np.uint8,
            )  # transient true particle
        Image.fromarray(image).save(source_dir / f"frame_{frame:06d}.png")

    detection_rows = [
        {
            "frame_index": 2,
            "label": 1,
            "y": 5,
            "x": 2,
            "area_px": 9,
            "bbox_top": 4,
            "bbox_left": 1,
            "bbox_bottom": 7,
            "bbox_right": 4,
            "score": "0.90000000",
            "confidence": "0.90000000",
            "class_id": 0,
            "source": "yolo11_raw",
        },
        {
            "frame_index": 2,
            "label": 2,
            "y": 5,
            "x": 6,
            "area_px": 9,
            "bbox_top": 4,
            "bbox_left": 5,
            "bbox_bottom": 7,
            "bbox_right": 8,
            "score": "0.80000000",
            "confidence": "0.80000000",
            "class_id": 0,
            "source": "yolo11_raw",
        },
    ]
    fields = [
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
    write_csv(yolo_run / "detections.csv", detection_rows, fields)
    write_csv(yolo_run / "detections_per_frame.csv", [{"frame_index": 2, "n_detections": 2}], ["frame_index", "n_detections"])
    (yolo_run / "metadata.json").write_text('{"mode":"yolo_export","n_images":1}\n', encoding="utf-8")
    truth_path.write_text(
        json.dumps(
            {
                "status": "reviewed_ground_truth",
                "requires_manual_review": False,
                "scored_frames": [2],
                "particles": [
                    {
                        "frame_index": 2,
                        "top": 4,
                        "left": 1,
                        "bottom": 7,
                        "right": 4,
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return source_dir, yolo_run, reference, truth_path


def test_find_revisit_selects_adjacent_revolutions(tmp_path: Path) -> None:
    source_dir, _yolo_run, reference, _truth_path = make_fixture(tmp_path)
    phases = load_phase_px_by_frame(reference / "phase_estimates.csv", frame_count=6)
    source_images = {frame: source_dir / f"frame_{frame:06d}.png" for frame in range(6)}
    revolution_by_frame = np.asarray([0, 0, 1, 1, 2, 2])

    prev = find_revisit(
        frame_index=2,
        belt_y=5,
        x=6,
        patch_shape=(3, 3),
        revolution_offset=-1,
        phase_by_frame=phases,
        revolution_by_frame=revolution_by_frame,
        source_images=source_images,
        image_shape=(10, 12),
        map_height=20,
    )
    next_revisit = find_revisit(
        frame_index=2,
        belt_y=5,
        x=6,
        patch_shape=(3, 3),
        revolution_offset=1,
        phase_by_frame=phases,
        revolution_by_frame=revolution_by_frame,
        source_images=source_images,
        image_shape=(10, 12),
        map_height=20,
    )

    assert prev is not None
    assert next_revisit is not None
    assert prev[0] == 0
    assert next_revisit[0] == 4


def test_yolo_recurrence_filter_removes_belt_fixed_detection(tmp_path: Path) -> None:
    source_dir, yolo_run, reference, truth_path = make_fixture(tmp_path)
    out = tmp_path / "outputs" / "yolo_recurrence"

    summary = run_yolo_recurrence_filter(
        yolo_run_dir=yolo_run,
        beltmap_reference_dir=reference,
        source_image_dir=source_dir,
        truth_path=truth_path,
        output_dir=out,
        config=YoloRecurrenceConfig(
            frame_count=6,
            belt_region=BeltRegion(0, 0, 10, 12),
            patch_margin_px=0,
            min_patch_size_px=3,
            hard_ratio_threshold=0.4,
            hard_min_revisits=2,
        ),
    )

    features = read_csv(out / "yolo_recurrence_features.csv")
    by_label = {row["label"]: row for row in features}
    assert by_label["1"]["raw_match_role"] == "TP"
    assert by_label["1"]["hard_reject"] == "False"
    assert by_label["2"]["raw_match_role"] == "FP"
    assert by_label["2"]["hard_reject"] == "True"
    assert summary.n_raw_false_positives_removed == 1
    assert summary.n_raw_true_positives_removed == 0

    hard_rows = read_csv(summary.hard_run_dir / "detections.csv")
    rerank_rows = read_csv(summary.rerank_run_dir / "detections.csv")
    assert [row["label"] for row in hard_rows] == ["1"]
    assert len(rerank_rows) == 2
    assert float(rerank_rows[1]["score"]) < float(rerank_rows[1]["yolo_confidence_original"])
    assert (out / "yolo_fp_fn_recurrence_contact_sheet.png").is_file()
    assert (out / "yolo_recurrence_report.md").is_file()


def test_yolo_recurrence_cli_smoke(tmp_path: Path) -> None:
    source_dir, yolo_run, reference, truth_path = make_fixture(tmp_path)
    out = tmp_path / "outputs" / "yolo_recurrence"

    code = yolo_recurrence_main(
        [
            "--yolo-run-dir",
            str(yolo_run),
            "--beltmap-reference-dir",
            str(reference),
            "--source-image-dir",
            str(source_dir),
            "--truth-path",
            str(truth_path),
            "--output-dir",
            str(out),
            "--frame-count",
            "6",
            "--belt-region",
            "0,0,10,12",
            "--patch-margin-px",
            "0",
            "--min-patch-size-px",
            "3",
            "--quiet",
        ]
    )

    assert code == 0
    assert (out / "yolo_recurrence_features.csv").is_file()
    assert (out.parent / "beltmap_runs" / "yolo11_raw_recurrence_hard_test" / "metadata.json").is_file()
    assert (out / "compare" / "summary.csv").is_file()


def test_yolo_recurrence_filter_cli_rejects_nonpositive_excess_floor(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_dir, yolo_run, reference, truth_path = make_fixture(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        yolo_recurrence_main(
            [
                "--yolo-run-dir",
                str(yolo_run),
                "--beltmap-reference-dir",
                str(reference),
                "--source-image-dir",
                str(source_dir),
                "--truth-path",
                str(truth_path),
                "--output-dir",
                str(tmp_path / "bad_out"),
                "--frame-count",
                "6",
                "--belt-region",
                "0,0,10,12",
                "--excess-floor",
                "0",
            ]
        )
    assert excinfo.value.code == 2
    assert "--excess-floor must be finite and positive" in capsys.readouterr().err
