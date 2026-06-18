from __future__ import annotations

import csv
import json
from pathlib import Path

from beltmap.evaluation import RunSpec, finite_float, fraction, summarize_output_dir, write_evaluation
from beltmap.cli.evaluate import main as evaluate_main


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "n_images": 3,
                "n_phase_estimates": 3,
                "n_detections": 4,
                "n_tracks": 2,
                "n_velocity_estimates": 2,
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "config_resolved.json").write_text("{}", encoding="utf-8")
    (output_dir / "progress.jsonl").write_text(
        json.dumps(
            {
                "stage": "belt_map",
                "observed_pixels": 80,
                "total_pixels": 100,
                "masked_pixels": 5,
                "contributed_pixels": 75,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "belt_map.png").write_bytes(b"not-a-real-png-for-this-test")
    write_csv(
        output_dir / "phase_estimates.csv",
        [
            {"frame_index": 0, "correction_px": -1.0, "score": 4.0},
            {"frame_index": 1, "correction_px": 0.0, "score": 2.0},
            {"frame_index": 2, "correction_px": 2.0, "score": 3.0},
        ],
    )
    write_csv(
        output_dir / "detections_per_frame.csv",
        [
            {"frame_index": 0, "n_detections": 1},
            {"frame_index": 1, "n_detections": 3},
            {"frame_index": 2, "n_detections": 0},
        ],
    )
    write_csv(
        output_dir / "detections.csv",
        [
            {"frame_index": 0, "centroid_y": 1.0, "centroid_x": 2.0},
            {"frame_index": 1, "centroid_y": 2.0, "centroid_x": 2.0},
            {"frame_index": 1, "centroid_y": 3.0, "centroid_x": 2.0},
            {"frame_index": 1, "centroid_y": 4.0, "centroid_x": 2.0},
        ],
    )
    write_csv(
        output_dir / "velocities.csv",
        [
            {"track_id": 0, "velocity_ratio_y": 0.5},
            {"track_id": 1, "velocity_ratio_y": 1.5},
        ],
    )


def test_summarize_output_dir_reports_ablation_metrics(tmp_path: Path) -> None:
    output_dir = tmp_path / "baseline"
    write_run(output_dir)

    summary = summarize_output_dir(RunSpec(name="baseline", output_dir=output_dir))

    assert summary["run"] == "baseline"
    assert summary["n_images"] == 3
    assert summary["n_detections"] == 4
    assert summary["phase_correction_abs_median_px"] == 1.0
    assert summary["registration_score_median"] == 3.0
    assert summary["belt_map_observed_fraction"] == 0.8
    assert summary["velocity_ratio_y_outlier_fraction"] == 0.5
    assert summary["missing_files"] == ""


def test_finite_numeric_helpers_reject_boolean_values() -> None:
    assert finite_float(True) is None
    assert finite_float(False) is None
    assert fraction(True, 100) is None
    assert fraction(10, False) is None


def test_summarize_output_dir_reports_zero_for_present_empty_csv_fallbacks(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "empty"
    output_dir.mkdir()
    for filename in [
        "phase_estimates.csv",
        "detections_per_frame.csv",
        "detections.csv",
        "velocities.csv",
    ]:
        (output_dir / filename).write_text("", encoding="utf-8")

    summary = summarize_output_dir(RunSpec(name="empty", output_dir=output_dir))

    assert summary["n_images"] == 0
    assert summary["n_phase_estimates"] == 0
    assert summary["n_detections"] == 0
    assert summary["n_velocity_estimates"] == 0
    assert summary["n_tracks"] is None
    assert "metadata.json" in summary["missing_files"]


def test_summarize_output_dir_ignores_fractional_metadata_counts(tmp_path: Path) -> None:
    output_dir = tmp_path / "fractional"
    write_run(output_dir)
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata["n_detections"] = 4.5
    metadata["n_tracks"] = 2.5
    (output_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    summary = summarize_output_dir(RunSpec(name="fractional", output_dir=output_dir))

    assert summary["n_detections"] == 4
    assert summary["n_tracks"] is None


def test_summarize_output_dir_rejects_boolean_metadata_and_progress(tmp_path: Path) -> None:
    output_dir = tmp_path / "boolean_metadata"
    write_run(output_dir)
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata["n_images"] = True
    metadata["n_detections"] = True
    metadata["n_tracks"] = False
    (output_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (output_dir / "progress.jsonl").write_text(
        json.dumps(
            {
                "stage": "belt_map",
                "observed_pixels": True,
                "total_pixels": 100,
                "masked_pixels": False,
                "contributed_pixels": 75,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_output_dir(RunSpec(name="boolean_metadata", output_dir=output_dir))

    assert summary["n_images"] == 3
    assert summary["n_detections"] == 4
    assert summary["n_tracks"] is None
    assert summary["belt_map_observed_fraction"] is None
    assert summary["belt_map_masked_fraction"] is None
    assert summary["belt_map_contributed_fraction"] == 0.75


def test_write_evaluation_writes_json_csv_and_markdown(tmp_path: Path) -> None:
    output_dir = tmp_path / "baseline"
    eval_dir = tmp_path / "eval"
    write_run(output_dir)

    artifacts = write_evaluation(
        [RunSpec(name="baseline", output_dir=output_dir)],
        output_dir=eval_dir,
    )

    assert artifacts.json_path.is_file()
    assert artifacts.csv_path.is_file()
    assert artifacts.markdown_path.is_file()
    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))
    assert payload["runs"][0]["run"] == "baseline"
    assert "baseline" in artifacts.markdown_path.read_text(encoding="utf-8")


def test_evaluation_markdown_treats_higher_registration_scores_as_better(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "baseline"
    eval_dir = tmp_path / "eval"
    write_run(output_dir)

    artifacts = write_evaluation(
        [RunSpec(name="baseline", output_dir=output_dir)],
        output_dir=eval_dir,
    )

    report = artifacts.markdown_path.read_text(encoding="utf-8")
    assert "higher registration scores" in report


def test_cli_evaluate_accepts_named_runs(tmp_path: Path) -> None:
    output_dir = tmp_path / "candidate"
    eval_dir = tmp_path / "eval"
    write_run(output_dir)

    status = evaluate_main(
        [
            "--run",
            f"candidate={output_dir}",
            "--output-dir",
            str(eval_dir),
            "--quiet",
        ]
    )

    assert status == 0
    assert (eval_dir / "evaluation_summary.json").is_file()
