from beltmap.cli.validate import final_belt_map_progress


def test_final_belt_map_progress_skips_save_event_without_coverage_fields():
    progress_rows = [
        {
            "stage": "belt_map",
            "message": "interpolating unobserved belt-map pixels",
            "observed_pixels": 12,
            "total_pixels": 20,
            "masked_pixels": 3,
        },
        {
            "stage": "belt_map",
            "message": "saved belt-map outputs",
            "belt_map_npy": "outputs/belt_map.npy",
        },
    ]

    progress = final_belt_map_progress(progress_rows)

    assert progress["observed_pixels"] == 12
    assert progress["total_pixels"] == 20
    assert progress["masked_pixels"] == 3
