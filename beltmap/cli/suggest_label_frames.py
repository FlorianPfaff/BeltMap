from __future__ import annotations

import argparse
from pathlib import Path

from beltmap.postrun_improvements import write_label_plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="beltmap-suggest-label-frames",
        description="Suggest a diverse set of frames for sparse real-data annotation.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = args.output or args.output_dir / "label_plan.csv"
    rows = write_label_plan(args.output_dir, output_path=output, frame_count=args.frames)
    print(f"wrote {len(rows)} frame suggestions to {output}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
