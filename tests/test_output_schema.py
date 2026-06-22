from __future__ import annotations

from beltmap import output_schema
from beltmap.yolo_export import DETECTION_FIELDS as YOLO_EXPORT_DETECTION_FIELDS


def test_yolo_export_uses_shared_schema() -> None:
    assert YOLO_EXPORT_DETECTION_FIELDS == output_schema.YOLO_DETECTION_FIELDS
    assert output_schema.DETECTIONS_PER_FRAME_FIELDS == ["frame_index", "n_detections"]


def test_track_detection_schema_extends_driver_detection_schema() -> None:
    assert output_schema.TRACK_DETECTION_FIELDS[:2] == [
        "track_id",
        "track_detection_index",
    ]
    assert output_schema.TRACK_DETECTION_FIELDS[2:] == output_schema.DRIVER_DETECTION_FIELDS


def test_specialized_detection_schemas_keep_driver_prefix() -> None:
    specialized = [
        output_schema.RECURRENT_ARTIFACT_DETECTION_FIELDS,
        output_schema.MAP_RISK_DETECTION_FIELDS,
        output_schema.CROSS_MAP_AGREEMENT_FIELDS,
        output_schema.REVOLUTION_SPLIT_GHOST_DETECTION_FIELDS,
    ]
    for fields in specialized:
        assert fields[: len(output_schema.DRIVER_DETECTION_FIELDS)] == output_schema.DRIVER_DETECTION_FIELDS
