from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from beltmap.operational_improvements import summarize_flux, write_science_exports


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize particle flux from BeltMap velocity rows.")
    parser.add_argument("--velocities", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("science_outputs"))
    parser.add_argument("--frame-count", type=int, default=0)
    parser.add_argument("--frame-rate-hz", type=float, default=0.0)
    args = parser.parse_args(argv)
    rows = read_rows(args.velocities)
    frame_count = None if args.frame_count <= 0 else args.frame_count
    frame_rate = None if args.frame_rate_hz <= 0 else args.frame_rate_hz
    artifacts = write_science_exports(args.output_dir, rows, frame_count=frame_count, frame_rate_hz=frame_rate)
    summary = summarize_flux(rows, frame_count=frame_count, frame_rate_hz=frame_rate)
    print(json.dumps({"summary": summary.to_dict(), "artifacts": {k: str(v) for k, v in artifacts.items()}}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
