from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from beltmap.operational_improvements import estimate_period_from_belt_map


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate a belt period from belt_map.npy.")
    parser.add_argument("--belt-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("belt_period_estimate.json"))
    parser.add_argument("--min-period-px", type=int, default=8)
    parser.add_argument("--max-period-px", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    belt_map = np.load(args.belt_map)
    estimate = estimate_period_from_belt_map(
        belt_map,
        min_period_px=args.min_period_px,
        max_period_px=None if args.max_period_px <= 0 else args.max_period_px,
    )
    payload = estimate.to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
