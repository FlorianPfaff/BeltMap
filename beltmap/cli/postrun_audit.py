from __future__ import annotations

import argparse
import json
from pathlib import Path

from beltmap.postrun_improvements import write_postrun_audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="beltmap-postrun-audit",
        description="Write post-run BeltMap quality flags, worst-frame tables, map uncertainty, confidence scores, label plans, and a quality contract report.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--frame-rate-hz", type=float, default=None)
    parser.add_argument("--write-full-counts", action="store_true", help="Also write full 2-D belt_map_counts.npy and belt_map_uncertainty.npy arrays.")
    parser.add_argument("--contract", type=Path, help="Optional JSON quality contract.")
    args = parser.parse_args(argv)

    contract = None
    if args.contract is not None:
        contract = json.loads(args.contract.read_text(encoding="utf-8"))
    summary = write_postrun_audit(
        args.output_dir,
        report_dir=args.report_dir,
        top_n=args.top_n,
        frame_rate_hz=args.frame_rate_hz,
        write_full_counts=args.write_full_counts,
        contract=contract,
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
