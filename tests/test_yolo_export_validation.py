from __future__ import annotations

import importlib

import pytest

import beltmap.yolo_export as yolo_export
import beltmap.yolo_export_image_patch as yolo_export_patch


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("-1 0.5 0.5 0.2 0.2 0.7", "class id must be non-negative"),
        ("0 0.5 0.5 0.2 0.2 -0.01", "confidence must be in"),
        ("0 0.5 0.5 0.2 0.2 1.01", "confidence must be in"),
    ],
)
def test_parse_yolo_label_line_rejects_invalid_class_and_confidence(
    line: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        yolo_export.parse_yolo_label_line(line)


@pytest.mark.parametrize("default_confidence", [-0.01, 1.01])
def test_parse_yolo_label_line_rejects_invalid_default_confidence(
    default_confidence: float,
) -> None:
    with pytest.raises(ValueError, match="confidence must be in"):
        yolo_export.parse_yolo_label_line(
            "0 0.5 0.5 0.2 0.2",
            default_confidence=default_confidence,
        )


def test_yolo_label_validation_patch_is_reload_safe() -> None:
    importlib.reload(yolo_export_patch)
    importlib.reload(yolo_export_patch)

    prediction = yolo_export.parse_yolo_label_line("0 0.5 0.5 0.2 0.2 1.0")

    assert prediction is not None
    assert prediction.class_id == 0
    assert prediction.confidence == 1.0
