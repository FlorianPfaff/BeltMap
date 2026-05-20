from __future__ import annotations

import argparse
import json
from pathlib import Path

from beltmap.advanced_quality import quality_flags, write_provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beltmap-advanced-report",
        description="Write additional BeltMap failure-mode diagnostics and provenance.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="BeltMap output directory to inspect.")
    parser.add_argument("--image-dir", type=Path, default=None, help="Optional image directory used to build a lightweight dataset manifest hash.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = quality_flags(args.output_dir)
    provenance = write_provenance(args.output_dir / "provenance.json", image_dir=args.image_dir)
    (args.output_dir / "failure_modes.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    lines = ["# Advanced BeltMap diagnostics", ""]
    lines.append(f"Output directory: `{args.output_dir}`")
    lines.append("")
    lines.append("## Failure-mode flags")
    lines.append("")
    flags = diagnostics.get("flags", [])
    if not flags:
        lines.append("No high-level failure-mode flags were triggered.")
    else:
        for flag in flags:
            lines.append(f"- **{flag['code']}** ({flag['severity']}): {flag['message']}")
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- Git commit: `{provenance.git_commit or 'unknown'}`")
    lines.append(f"- Git dirty: `{provenance.git_dirty}`")
    lines.append(f"- Python: `{provenance.python_version.split()[0]}`")
    lines.append(f"- Dataset manifest SHA-256: `{provenance.input_manifest_sha256 or 'not recorded'}`")
    (args.output_dir / "advanced_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_dir / "advanced_diagnostics.md")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
