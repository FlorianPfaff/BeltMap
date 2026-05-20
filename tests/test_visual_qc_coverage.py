from beltmap.visual_qc import draw_coverage_image, estimate_belt_map_coverage


def test_estimated_belt_map_coverage_keeps_compact_row_counts(tmp_path):
    metadata = {
        "belt_map_height_px": 16,
        "belt_region": {"top": 0, "left": 0, "height": 4, "width": 1000},
    }
    phase_rows = [
        {"frame_index": 0, "phase_px": 0.0},
        {"frame_index": 1, "phase_px": 2.0},
    ]

    coverage = estimate_belt_map_coverage(phase_rows, metadata)

    assert coverage is not None
    assert coverage.shape == (16,)
    assert coverage.sum() == 8.0

    output_path = tmp_path / "coverage.png"
    draw_coverage_image(output_path, coverage)

    assert output_path.is_file()
    assert output_path.stat().st_size > 0
