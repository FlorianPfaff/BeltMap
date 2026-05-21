from __future__ import annotations

import argparse
import json
from pathlib import Path

from beltmap.postrun_improvements import write_map_uncertainty_outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="beltmap-map-uncertainty",
        description="Estimate belt-map phase coverage and uncertainty from phase_estimates.csv.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--write-full-counts", action="store_true")
    args = parser.parse_args(argv)
    summary = write_map_uncertainty_outputs(
        args.output_dir,
        report_dir=args.report_dir,
        scale=args.scale,
        write_full_counts=args.write_full_counts,
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
