from __future__ import annotations

import argparse
import json
from pathlib import Path

from beltmap.operational_improvements import write_container_templates, write_quality_tooling_templates, write_workflow_templates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write optional workflow, container, and quality-tooling templates.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workflow", action="store_true")
    parser.add_argument("--container", action="store_true")
    parser.add_argument("--quality", action="store_true")
    args = parser.parse_args(argv)
    artifacts = {}
    if args.workflow or not any([args.workflow, args.container, args.quality]):
        artifacts.update({f"workflow_{k}": str(v) for k, v in write_workflow_templates(args.output_dir).items()})
    if args.container or not any([args.workflow, args.container, args.quality]):
        artifacts.update({f"container_{k}": str(v) for k, v in write_container_templates(args.output_dir).items()})
    if args.quality or not any([args.workflow, args.container, args.quality]):
        artifacts.update({f"quality_{k}": str(v) for k, v in write_quality_tooling_templates(args.output_dir).items()})
    print(json.dumps(artifacts, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
