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


def _paths_refer_to_same_file(first: Path, second: Path) -> bool:
    """Return whether two path spellings identify the same filesystem object."""

    try:
        if first.resolve(strict=False) == second.resolve(strict=False):
            return True
    except OSError:
        pass

    try:
        return first.samefile(second)
    except (FileNotFoundError, OSError):
        return False


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if _paths_refer_to_same_file(args.belt_map, args.output):
        parser.error("--output must not refer to the same file as --belt-map")

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
