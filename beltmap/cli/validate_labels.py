from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from beltmap.label_validation import validated_label_state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a BeltMap truth-label JSON and report whether it is safe "
            "to use for labeled metrics."
        )
    )
    parser.add_argument(
        "--truth-path",
        type=Path,
        required=True,
        help="Path to a truth-label JSON file.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format. Defaults to text.",
    )
    parser.add_argument(
        "--allow-invalid",
        action="store_true",
        help="Return exit code 0 even when the label file is not metric-ready.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validated_label_state(args.truth_path)
    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        sys.stdout.write(report.format_text())
    if report.is_valid_for_metrics or args.allow_invalid:
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
