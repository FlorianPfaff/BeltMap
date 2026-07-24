from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from beltmap.evaluation import RunSpec, write_evaluation


def parse_run_spec(value: str) -> RunSpec:
    if "=" in value:
        name, path_text = value.split("=", 1)
        name = name.strip()
        path_text = path_text.strip()
        if not name:
            raise argparse.ArgumentTypeError("run name before '=' must not be empty")
        if not path_text:
            raise argparse.ArgumentTypeError(
                "run output directory after '=' must not be empty"
            )
        return RunSpec(name=name, output_dir=Path(path_text))

    path = Path(value)
    name = path.name or str(path)
    return RunSpec(name=name, output_dir=path)


def _evaluation_output_paths(
    *,
    output_dir: Path,
    json_path: Path | None,
    csv_path: Path | None,
    markdown_path: Path | None,
) -> dict[str, Path]:
    return {
        "JSON": json_path or (output_dir / "evaluation_summary.json"),
        "CSV": csv_path or (output_dir / "evaluation_summary.csv"),
        "Markdown": markdown_path or (output_dir / "evaluation_summary.md"),
    }


def _validate_distinct_output_paths(
    *,
    output_dir: Path,
    json_path: Path | None,
    csv_path: Path | None,
    markdown_path: Path | None,
) -> None:
    seen: dict[str, tuple[str, Path]] = {}
    for label, path in _evaluation_output_paths(
        output_dir=output_dir,
        json_path=json_path,
        csv_path=csv_path,
        markdown_path=markdown_path,
    ).items():
        resolved = path.resolve(strict=False)
        key = os.path.normcase(str(resolved))
        previous = seen.get(key)
        if previous is not None:
            previous_label, _previous_path = previous
            raise ValueError(
                f"{previous_label} and {label} evaluation outputs must use distinct "
                f"paths; both resolve to {resolved}"
            )
        seen[key] = (label, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beltmap-evaluate",
        description=(
            "Compare BeltMap output directories and write JSON/CSV/Markdown "
            "ablation summaries."
        ),
    )
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run_spec,
        required=True,
        metavar="NAME=OUTPUT_DIR",
        help=(
            "BeltMap output directory to include. May be passed multiple times. "
            "Use NAME=PATH to give stable ablation labels; a bare path uses its "
            "directory name."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation"),
        help="Directory for generated evaluation artifacts. Default: evaluation",
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        help="Explicit JSON summary path. Default: OUTPUT_DIR/evaluation_summary.json",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        help="Explicit CSV summary path. Default: OUTPUT_DIR/evaluation_summary.csv",
    )
    parser.add_argument(
        "--markdown-path",
        type=Path,
        help="Explicit Markdown summary path. Default: OUTPUT_DIR/evaluation_summary.md",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print generated artifact paths as JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        _validate_distinct_output_paths(
            output_dir=args.output_dir,
            json_path=args.json_path,
            csv_path=args.csv_path,
            markdown_path=args.markdown_path,
        )
    except ValueError as exc:
        parser.error(str(exc))

    artifacts = write_evaluation(
        args.run,
        output_dir=args.output_dir,
        json_path=args.json_path,
        csv_path=args.csv_path,
        markdown_path=args.markdown_path,
    )

    if not args.quiet:
        print(
            json.dumps(
                {
                    "json": str(artifacts.json_path),
                    "csv": str(artifacts.csv_path),
                    "markdown": str(artifacts.markdown_path),
                },
                indent=2,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
