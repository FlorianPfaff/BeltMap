import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from beltmap.cli import sweep as cli_sweep


def write_base_config(path: Path, image_dir: Path) -> None:
    path.write_text(
        f"""[paths]
image_dir = {json.dumps(str(image_dir))}
output_dir = "unused"

[detection]
threshold = 3.0
low_threshold = 0.0
""",
        encoding="utf-8",
    )


@pytest.mark.parametrize("value", ["", " ", "nan", "inf"])
def test_sweep_parse_scalar_rejects_empty_or_nonfinite_values(value):
    with pytest.raises(ValueError):
        cli_sweep.parse_scalar(value)


@pytest.mark.parametrize(
    "value",
    ["=1", "detection.=1", "detection.threshold=", "detection.threshold=1,"],
)
def test_sweep_parse_param_rejects_malformed_keys_or_values(value):
    with pytest.raises(ValueError):
        cli_sweep.parse_param(value)


def test_sweep_set_and_get_dotted_reject_empty_components():
    with pytest.raises(ValueError, match="dotted keys"):
        cli_sweep.set_dotted({}, "detection..threshold", 3.0)
    with pytest.raises(ValueError, match="dotted keys"):
        cli_sweep.get_dotted({}, "detection..threshold")


def test_sweep_toml_value_rejects_none_and_nonfinite_numbers():
    with pytest.raises(ValueError, match="None"):
        cli_sweep.toml_value(None)
    with pytest.raises(ValueError, match="finite"):
        cli_sweep.toml_value(float("nan"))


def test_sweep_false_positives_per_frame_ignores_invalid_counts():
    assert (
        cli_sweep.false_positives_per_frame(
            {"detections": {"false_positives": -1}, "case": {"frames": 10}}
        )
        is None
    )
    assert (
        cli_sweep.false_positives_per_frame(
            {"detections": {"false_positives": 1}, "case": {"frames": -10}}
        )
        is None
    )


def test_sweep_froc_points_filters_invalid_operating_points():
    points = cli_sweep.sweep_froc_points(
        [
            {"false_positives_per_frame": -0.1, "detection_recall": 0.8},
            {"false_positives_per_frame": 0.1, "detection_recall": 1.2},
            {
                "false_positives_per_frame": 0.2,
                "detection_recall": 0.7,
                "detection_threshold": 3.0,
            },
        ]
    )

    assert points == [(0.2, 0.7, "3.0")]


def test_sweep_benchmark_summary_sanitizes_invalid_metrics(tmp_path):
    row = cli_sweep.benchmark_summary_row(
        run_index=0,
        output_dir=tmp_path / "run",
        config={"detection": {"threshold": 3.0}},
        overrides={},
        metrics={
            "case": {"frames": 10},
            "detections": {
                "precision": 1.2,
                "recall": 0.5,
                "f1": float("nan"),
                "false_positives": -1,
            },
            "tracks": {"mean_track_length": -2, "single_frame_track_fraction": 1.5},
            "velocity": {
                "velocity_y_error_std_px_per_frame": -0.1,
                "velocity_y_bias_px_per_frame": float("inf"),
            },
            "phase": {"rmse_px": -0.5},
        },
    )

    assert row["detection_precision"] is None
    assert row["detection_recall"] == 0.5
    assert row["detection_f1"] is None
    assert row["false_positives_per_frame"] is None
    assert row["mean_track_length"] is None
    assert row["single_frame_track_fraction"] is None
    assert row["velocity_y_error_std_px_per_frame"] is None
    assert row["velocity_y_bias_px_per_frame"] is None
    assert row["phase_rmse_px"] is None


def test_sweep_markdown_value_hides_nonfinite_values():
    assert cli_sweep.markdown_value(float("nan")) == "n/a"


def test_sweep_cli_rejects_invalid_benchmark_iou_threshold(tmp_path, capsys):
    base_config = tmp_path / "beltmap.toml"
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    write_base_config(base_config, image_dir)

    with pytest.raises(SystemExit) as exc_info:
        cli_sweep.main(
            [
                "--base-config",
                str(base_config),
                "--benchmark-iou-threshold",
                "nan",
            ]
        )

    assert exc_info.value.code == 2
    assert "IoU threshold" in capsys.readouterr().err


def test_sweep_cli_rejects_none_sweep_values_that_cannot_be_written(tmp_path, capsys):
    base_config = tmp_path / "beltmap.toml"
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    write_base_config(base_config, image_dir)

    with pytest.raises(SystemExit) as exc_info:
        cli_sweep.main(
            [
                "--base-config",
                str(base_config),
                "--param",
                "detection.threshold=none",
                "--output-root",
                str(tmp_path / "sweep"),
            ]
        )

    assert exc_info.value.code == 2
    assert "None" in capsys.readouterr().err


def test_sweep_writes_benchmark_curve_summaries(tmp_path, monkeypatch):
    base_config = tmp_path / "beltmap.toml"
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    truth_path = tmp_path / "synthetic_metadata.json"
    truth_path.write_text("{}", encoding="utf-8")
    write_base_config(base_config, image_dir)

    def fake_benchmark_report(*, output_dir, metrics_path=None, **_kwargs):
        run_index = int(Path(output_dir).name.split("_")[-1])
        metrics_file = metrics_path or Path(output_dir) / "benchmark_metrics.json"
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        metrics = {
            "case": {"frames": 10},
            "run": {"n_images": 10},
            "runtime": {"frames": 10},
            "detections": {
                "precision": 0.8 + 0.1 * run_index,
                "recall": 0.7,
                "f1": 0.746 + 0.044 * run_index,
                "false_positives": 2 - run_index,
            },
            "events": {
                "precision": 0.75,
                "recall": 0.5,
                "f1": 0.6,
                "track_fragmentation": 0.25,
                "fragmented_truth_events": 1,
                "mean_fragments_per_truth_event": 1.25,
                "birth_false_positive_rate": 0.25,
                "missed_event_rate": 0.5,
            },
            "filtered_events": {
                "precision": 1.0,
                "recall": 0.5,
                "f1": 2 / 3,
                "track_fragmentation": 0.0,
            },
            "tracks": {
                "mean_track_length": 2.5,
                "median_track_length": 2.0,
                "single_frame_tracks": 3,
                "single_frame_track_fraction": 0.3,
            },
            "filtered_tracks": {
                "mean_track_length": 4.0,
                "median_track_length": 4.0,
                "single_frame_tracks": 0,
                "single_frame_track_fraction": 0.0,
            },
            "velocity": {
                "velocity_y_error_px_per_frame": -0.2,
                "velocity_y_mean_abs_error_px_per_frame": 0.2,
                "velocity_y_bias_px_per_frame": -0.15,
                "velocity_y_error_std_px_per_frame": 0.05,
                "truth_matched_velocity_y_error_px_per_frame": -0.1,
            },
            "filtered_velocity": {
                "velocity_y_error_px_per_frame": -0.05,
                "velocity_y_mean_abs_error_px_per_frame": 0.05,
                "velocity_y_bias_px_per_frame": -0.04,
                "velocity_y_error_std_px_per_frame": 0.01,
                "truth_matched_velocity_y_error_px_per_frame": -0.02,
            },
            "phase": {"rmse_px": 0.125},
            "belt_map": {"rmse_gray": 1.5},
        }
        metrics_file.write_text(json.dumps(metrics), encoding="utf-8")
        return SimpleNamespace(
            metrics=metrics_file, report=Path(output_dir) / "benchmark_report.md"
        )

    monkeypatch.setattr(cli_sweep, "generate_benchmark_report", fake_benchmark_report)

    exit_code = cli_sweep.main(
        [
            "--base-config",
            str(base_config),
            "--param",
            "detection.threshold=2.0,3.0",
            "--output-root",
            str(tmp_path / "sweep"),
            "--benchmark-truth-path",
            str(truth_path),
        ]
    )

    assert exit_code == 0
    rows = list(
        csv.DictReader(
            (tmp_path / "sweep" / "sweep_metrics.csv").open(
                newline="", encoding="utf-8"
            )
        )
    )
    assert [row["detection_threshold"] for row in rows] == ["2.0", "3.0"]
    assert rows[0]["false_positives_per_frame"] == "0.2"
    assert rows[0]["track_fragmentation"] == "0.25"
    assert rows[0]["single_frame_tracks"] == "3"
    assert rows[0]["median_track_length"] == "2.0"
    assert rows[0]["velocity_y_bias_px_per_frame"] == "-0.15"
    assert rows[0]["birth_false_positive_rate"] == "0.25"
    assert rows[0]["truth_matched_velocity_y_error_px_per_frame"] == "-0.1"
    assert (
        json.loads(
            (tmp_path / "sweep" / "sweep_metrics.json").read_text(encoding="utf-8")
        )[1]["detection_precision"]
        == 0.9
    )
    report = (tmp_path / "sweep" / "sweep_report.md").read_text(encoding="utf-8")
    assert (tmp_path / "sweep" / "sweep_froc_curve.svg").is_file()
    assert "Detection FROC" in report
    assert "sweep_froc_curve.svg" in report
    assert "FP/frame" in report
    assert "Track fragmentation" in report
    assert "Single-frame tracks" in report
