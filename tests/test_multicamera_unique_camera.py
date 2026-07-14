from beltmap.operational_improvements import stitch_multicamera_events


def test_stitch_multicamera_events_keeps_same_camera_rows_separate() -> None:
    events = stitch_multicamera_events(
        {
            "camera-a": [
                {"row_id": "a1", "time_s": 1.0, "phase_px": 10.0},
                {"row_id": "a2", "time_s": 1.01, "phase_px": 11.0},
            ]
        }
    )

    assert len(events) == 2
    assert all(len(event.camera_rows) == 1 for event in events)
    assert {event.camera_rows[0]["row_id"] for event in events} == {"a1", "a2"}


def test_stitch_multicamera_events_uses_each_camera_at_most_once() -> None:
    events = stitch_multicamera_events(
        {
            "camera-a": [
                {"row_id": "a1", "time_s": 1.0, "phase_px": 10.0},
            ],
            "camera-b": [
                {"row_id": "b1", "time_s": 1.01, "phase_px": 10.5},
                {"row_id": "b2", "time_s": 1.02, "phase_px": 11.0},
            ],
            "camera-c": [
                {"row_id": "c1", "time_s": 1.01, "phase_px": 9.5},
            ],
        }
    )

    assert sorted(len(event.camera_rows) for event in events) == [1, 3]
    for event in events:
        cameras = [row["camera"] for row in event.camera_rows]
        assert len(cameras) == len(set(cameras))
