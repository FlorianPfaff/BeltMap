from __future__ import annotations

import argparse
from pathlib import Path

from beltmap.operational_improvements import build_review_items, write_html_review


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a lightweight HTML review page for overlay images.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-html", type=Path, default=None)
    args = parser.parse_args(argv)
    overlays = sorted(args.output_dir.glob("*overlay*.png")) + sorted(args.output_dir.glob("residual_frame_*.png"))
    items = build_review_items(overlays)
    target = args.review_html or args.output_dir / "review.html"
    write_html_review(target, items)
    print(target)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
