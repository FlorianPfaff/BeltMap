import numpy as np

import beltmap.driver as driver
from beltmap import _driver_runtime as rt
from beltmap.phase import PhaseEstimate
from beltmap.rendering import BeltRegion, CleanBeltRender
from beltmap.residual import ResidualImage
from beltmap.tracking import ParticleDetection, ParticleTrack


def test_clear_generated_outputs_removes_optional_driver_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "OUT", tmp_path)
    stale_names = [
        "photometric_fits.csv",
        "recurrent_artifact_exposure_counts.npy",
        "recurrent_artifact_probability.npy",
        "recurrent_artifact_probability.png",
    ]
    for name in stale_names:
        (tmp_path / name).write_text("stale", encoding="utf-8")
    keep_path = tmp_path / "manual_notes.txt"
    keep_path.write_text("keep", encoding="utf-8")

    rt.clear_generated_outputs()

    assert keep_path.is_file()
    for name in stale_names:
        assert not (tmp_path / name).exists()


def _residual_with_phase_estimate() -> ResidualImage:
    image = np.zeros((1, 1), dtype=np.float64)
    mask = np.ones((1, 1), dtype=bool)
    estimate = PhaseEstimate(
        phase_px=1.0,
        frame_index=0.0,
        predicted_phase_px=1.0,
    )
    clean_render = CleanBeltRender(
        image=image,
        mask=mask,
        phase_estimate=estimate,
        belt_region=BeltRegion(top=0, left=0, height=1, width=1),
    )
    return ResidualImage(
        raw=image,
        local_noise=np.ones((1, 1), dtype=np.float64),
        normalized=image,
        mask=mask,
        expected_background=image,
        clean_render=clean_render,
    )


def _detection() -> ParticleDetection:
    return ParticleDetection(
        frame_index=0.0,
        label=1,
        y=0.5,
        x=0.5,
        area_px=1,
        bbox_top=0,
        bbox_left=0,
        bbox_bottom=1,
        bbox_right=1,
    )


def test_driver_output_rows_tolerate_paths_outside_data_root(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    image_path = tmp_path / "external" / "frame_000.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"not-an-image-needed-for-row-tests")
    monkeypatch.setattr(rt, "DATA", data_root)

    residual = _residual_with_phase_estimate()
    detection = _detection()

    phase_row = driver.phase_estimate_row(
        0,
        image_path,
        residual,
        period_px=2.0,
    )
    detection_row = driver.detection_rows_for_frame(
        [detection],
        image_path,
        frame_index=0,
    )[0]
    track_row = driver.track_detection_rows(
        [ParticleTrack(track_id=0, detections=(detection,))],
        [image_path],
    )[0]

    assert phase_row["image"] == str(image_path)
    assert detection_row["image"] == str(image_path)
    assert track_row["image"] == str(image_path)
