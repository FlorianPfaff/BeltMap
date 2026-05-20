from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from beltmap.operational_improvements import empirical_p_values, fdr_threshold_from_p_values, recommend_threshold


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Suggest a residual threshold from empirical residual tails.")
    parser.add_argument("--residual-npy", type=Path, required=True, help="2-D residual image or N-D residual stack saved as .npy")
    parser.add_argument("--output", type=Path, default=Path("threshold_suggestion.json"))
    parser.add_argument("--expected-false-pixels", type=float, default=1.0)
    parser.add_argument("--polarity", choices=["bright", "dark", "absolute"], default="bright")
    parser.add_argument("--fdr-alpha", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    residual = np.load(args.residual_npy)
    threshold = recommend_threshold(
        residual,
        expected_false_pixels_per_frame=args.expected_false_pixels,
        polarity=args.polarity,
    )
    payload = {
        "threshold": threshold,
        "polarity": args.polarity,
        "expected_false_pixels": args.expected_false_pixels,
    }
    if args.fdr_alpha > 0:
        p_values = empirical_p_values(residual, polarity=args.polarity)
        signal = residual if args.polarity == "bright" else (-residual if args.polarity == "dark" else abs(residual))
        payload["fdr_threshold"] = fdr_threshold_from_p_values(p_values, signal, alpha=args.fdr_alpha)
        payload["fdr_alpha"] = args.fdr_alpha
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
