from __future__ import annotations

import importlib

import pytest

import beltmap.cli  # noqa: F401
import beltmap.yolo_recurrence as yolo_recurrence
import beltmap.yolo_recurrence_region_patch as region_patch


def test_cli_autoloads_strict_belt_region_parser() -> None:
    assert getattr(
        yolo_recurrence.parse_belt_region,
        "_beltmap_yolo_recurrence_region_validated",
        False,
    )


def test_parse_belt_region_preserves_integer_valued_numeric_forms() -> None:
    region = yolo_recurrence.parse_belt_region("1.0,2e0,3,4")

    assert region.top == 1
    assert region.left == 2
    assert region.height == 3
    assert region.width == 4


@pytest.mark.parametrize(
    ("value", "component"),
    [
        ("0.5,2,3,4", "top"),
        ("1,2.5,3,4", "left"),
        ("1,2,3.5,4", "height"),
        ("1,2,3,4.5", "width"),
    ],
)
def test_parse_belt_region_rejects_fractional_components(
    value: str,
    component: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"belt region {component} must be a finite integer",
    ):
        yolo_recurrence.parse_belt_region(value)


@pytest.mark.parametrize("value", ["nan,2,3,4", "1,inf,3,4", "1,2,-inf,4"])
def test_parse_belt_region_rejects_non_finite_components(value: str) -> None:
    with pytest.raises(ValueError, match="must be a finite integer"):
        yolo_recurrence.parse_belt_region(value)


def test_yolo_recurrence_region_patch_reload_is_idempotent() -> None:
    before = yolo_recurrence.parse_belt_region
    before_original = getattr(
        before,
        "_beltmap_yolo_recurrence_original_parse_belt_region",
        before,
    )

    importlib.reload(region_patch)

    after = yolo_recurrence.parse_belt_region
    after_original = getattr(
        after,
        "_beltmap_yolo_recurrence_original_parse_belt_region",
        after,
    )
    assert getattr(after, "_beltmap_yolo_recurrence_region_validated", False)
    assert after_original is before_original
    assert after_original is not after
