from __future__ import annotations

import csv

from PIL import Image

from beltmap.cli.annotation_audit_review import (
    build_payload,
    find_source_image,
    merge_frame_review,
)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_png(path, size=(20, 10)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, 128).save(path)


def test_annotation_audit_review_builds_context_state(tmp_path):
    audit_dir = tmp_path / "audit"
    crop_dir = tmp_path / "crops"
    write_csv(
        audit_dir / "audit_frame_selection.csv",
        [
            {
                "audit_ordinal": 1,
                "frame_index": 10,
                "primary_bucket": "artifact_heavy",
                "raw_crop_path": "raw_crops/frame_000010.png",
                "context_frames_json": "[9,10,11]",
                "context_raw_sheet_path": "",
                "context_target_label_sheet_path": "",
                "selection_method": "test",
            }
        ],
        [
            "audit_ordinal",
            "frame_index",
            "primary_bucket",
            "raw_crop_path",
            "context_frames_json",
            "context_raw_sheet_path",
            "context_target_label_sheet_path",
            "selection_method",
        ],
    )
    (audit_dir / "audit_original_reference_labels.json").write_text(
        """
{
  "coordinate_system": {
    "crop_region_in_source_image": {"top": 0, "left": 0, "height": 10, "width": 20}
  },
  "frames": [
    {
      "frame_index": 10,
      "primary_bucket": "artifact_heavy",
      "original_particles": [
        {"frame_index": 10, "top": 1, "left": 2, "bottom": 5, "right": 8, "event_id": "p1"}
      ]
    }
  ]
}
""",
        encoding="utf-8",
    )
    write_png(audit_dir / "raw_crops" / "frame_000010.png")
    write_png(crop_dir / "frame_000009.png")
    write_png(crop_dir / "frame_000011.png")

    state = build_payload(
        audit_dir=audit_dir,
        crop_dir=crop_dir,
        source_image_dir=None,
        review_path=audit_dir / "audit_click_review.json",
        context_radius=1,
    )

    assert state.payload["crop_size"] == {"width": 20, "height": 10}
    assert len(state.payload["frames"]) == 1
    frame = state.payload["frames"][0]
    assert frame["frame_index"] == 10
    assert frame["primary_bucket"] == "artifact_heavy"
    assert len(frame["original_particles"]) == 1
    assert [item["missing"] for item in frame["context"]] == [False, False, False]
    assert len(state.image_paths) == 3


def test_annotation_audit_review_finds_generic_supported_source_images(tmp_path):
    source_dir = tmp_path / "source"
    expected = source_dir / "nested" / "camera_frame_000011.jpg"
    write_png(expected)

    assert find_source_image(source_dir, 11) == expected


def test_annotation_audit_review_merge_updates_status():
    review = {
        "frames": [
            {
                "frame_index": 10,
                "primary_bucket": "artifact_heavy",
                "review_status": "unreviewed",
                "mistake_points": [],
            }
        ]
    }

    merged = merge_frame_review(
        review,
        {
            "frame_index": 10,
            "review_status": "needs_correction",
            "accept_existing_boxes": False,
            "mistake_points": [{"x": 4.0, "y": 3.0, "kind": "missed_particle"}],
            "notes": "possible missed particle",
        },
    )

    assert merged["review_status"] == "needs_correction"
    assert review["status"] == "reviewed_with_click_flags"
    assert review["frames"][0]["mistake_points"][0]["kind"] == "missed_particle"
