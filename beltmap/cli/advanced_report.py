from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from beltmap.advanced_quality import (
    estimate_integer_xy_shift,
    quality_flags,
    write_provenance,
)
from beltmap.compare_runs import finite_int
from beltmap.phase import render_belt_view


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beltmap-advanced-report",
        description="Write additional BeltMap failure-mode diagnostics and provenance.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="BeltMap output directory to inspect.")
    parser.add_argument("--image-dir", type=Path, default=None, help="Optional image directory used to build a lightweight dataset manifest hash.")
    parser.add_argument("--xy-shift-samples", type=int, default=0, help="Optional number of frames used for diagnostic 2-D crop-shift checks. 0 disables.")
    parser.add_argument("--xy-shift-max-px", type=int, default=4, help="Maximum x/y integer shift searched by the diagnostic 2-D alignment check.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = quality_flags(args.output_dir)
    xy_shift = estimate_xy_shift_diagnostics(
        output_dir=args.output_dir,
        image_dir=args.image_dir,
        sample_count=args.xy_shift_samples,
        max_shift_px=args.xy_shift_max_px,
    )
    if xy_shift:
        diagnostics["xy_shift"] = xy_shift
    provenance = write_provenance(args.output_dir / "provenance.json", image_dir=args.image_dir)
    (args.output_dir / "failure_modes.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    lines = ["# Advanced BeltMap diagnostics", ""]
    lines.append(f"Output directory: `{args.output_dir}`")
    lines.append("")
    lines.append("## Failure-mode flags")
    lines.append("")
    flags = diagnostics.get("flags", [])
    if not flags:
        lines.append("No high-level failure-mode flags were triggered.")
    else:
        for flag in flags:
            lines.append(f"- **{flag['code']}** ({flag['severity']}): {flag['message']}")
    if xy_shift:
        lines.append("")
        lines.append("## Diagnostic 2-D alignment shift")
        lines.append("")
        lines.append(f"- Samples: `{xy_shift.get('samples', 0)}`")
        lines.append(f"- Median y-shift: `{xy_shift.get('median_shift_y_px')}` px")
        lines.append(f"- Median x-shift: `{xy_shift.get('median_shift_x_px')}` px")
        lines.append(f"- Max absolute y-shift: `{xy_shift.get('max_abs_shift_y_px')}` px")
        lines.append(f"- Max absolute x-shift: `{xy_shift.get('max_abs_shift_x_px')}` px")
        lines.append(f"- Nonzero x-shift share: `{xy_shift.get('nonzero_x_shift_share')}`")
        lines.append("")
        lines.append("Repeated nonzero x-shifts indicate crop drift, camera motion, or perspective misalignment; tune geometry before lowering detection thresholds.")
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- Git commit: `{provenance.git_commit or 'unknown'}`")
    lines.append(f"- Git dirty: `{provenance.git_dirty}`")
    lines.append(f"- Python: `{provenance.python_version.split()[0]}`")
    lines.append(f"- Dataset manifest SHA-256: `{provenance.input_manifest_sha256 or 'not recorded'}`")
    (args.output_dir / "advanced_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_dir / "advanced_diagnostics.md")
    return 0


def estimate_xy_shift_diagnostics(
    *,
    output_dir: Path,
    image_dir: Path | None,
    sample_count: int,
    max_shift_px: int,
) -> dict[str, Any]:
    """Write optional diagnostic 2-D registration shifts for sampled frames."""

    if sample_count <= 0 or image_dir is None:
        return {}
    belt_map_path = output_dir / "belt_map.npy"
    phase_path = output_dir / "phase_estimates.csv"
    metadata_path = output_dir / "metadata.json"
    if not belt_map_path.is_file() or not phase_path.is_file() or not metadata_path.is_file():
        return {}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    region_data = metadata.get("belt_region") or {}
    try:
        top = int(region_data["top"])
        left = int(region_data["left"])
        height = int(region_data["height"])
        width = int(region_data["width"])
    except (KeyError, TypeError, ValueError):
        return {}

    with phase_path.open(newline="", encoding="utf-8") as handle:
        phase_rows = list(csv.DictReader(handle))
    if not phase_rows:
        return {}
    selected_positions = np.linspace(
        0,
        len(phase_rows) - 1,
        num=max(1, min(sample_count, len(phase_rows))),
        dtype=int,
    )
    belt_map = np.load(belt_map_path)
    rows: list[dict[str, Any]] = []
    for position in selected_positions:
        row = phase_rows[int(position)]
        image_name = row.get("image", "").strip()
        if not image_name:
            continue
        image_path = image_dir / image_name
        if not image_path.is_file():
            continue
        frame = _read_gray(image_path)
        if frame.shape[0] < top + height or frame.shape[1] < left + width:
            continue
        observed = frame[top : top + height, left : left + width]
        expected = render_belt_view(belt_map, float(row["phase_px"]), height)
        if expected.shape[1] != width:
            continue
        shift = estimate_integer_xy_shift(
            observed,
            expected,
            max_shift_y_px=max_shift_px,
            max_shift_x_px=max_shift_px,
        )
        frame_index = finite_int(row.get("frame_index"))
        if frame_index is None:
            continue
        rows.append({
            "frame_index": frame_index,
            "image": image_name,
            "shift_y_px": shift.shift_y_px,
            "shift_x_px": shift.shift_x_px,
            "loss": shift.loss,
            "score": shift.score,
        })
    if not rows:
        return {}
    csv_path = output_dir / "xy_shift_diagnostics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    shifts_y = np.asarray([row["shift_y_px"] for row in rows], dtype=np.float64)
    shifts_x = np.asarray([row["shift_x_px"] for row in rows], dtype=np.float64)
    summary = {
        "samples": len(rows),
        "csv": str(csv_path),
        "median_shift_y_px": float(np.median(shifts_y)),
        "median_shift_x_px": float(np.median(shifts_x)),
        "max_abs_shift_y_px": float(np.max(np.abs(shifts_y))),
        "max_abs_shift_x_px": float(np.max(np.abs(shifts_x))),
        "nonzero_x_shift_share": float(np.mean(shifts_x != 0)),
        "nonzero_y_shift_share": float(np.mean(shifts_y != 0)),
    }
    (output_dir / "xy_shift_diagnostics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _read_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float64)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
