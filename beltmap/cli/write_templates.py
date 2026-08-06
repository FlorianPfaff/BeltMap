from __future__ import annotations

import argparse
import json
from pathlib import Path

from beltmap.operational_improvements import (
    write_container_templates,
    write_quality_tooling_templates,
    write_workflow_templates,
)

WORKFLOW_CONFIG_NAME = "beltmap-workflow-config.yaml"
DEFAULT_WORKFLOW_CONFIG = "config: beltmap.toml\n"


def write_workflow_bundle(output_dir: Path) -> dict[str, Path]:
    """Write runnable workflow templates and their default Snakemake config."""

    artifacts = write_workflow_templates(output_dir)
    config_path = output_dir / WORKFLOW_CONFIG_NAME
    if not config_path.exists():
        config_path.write_text(DEFAULT_WORKFLOW_CONFIG, encoding="utf-8")
    return {**artifacts, "snakemake_config": config_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write optional workflow, container, and quality-tooling templates."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workflow", action="store_true")
    parser.add_argument("--container", action="store_true")
    parser.add_argument("--quality", action="store_true")
    args = parser.parse_args(argv)
    artifacts = {}
    write_all = not any([args.workflow, args.container, args.quality])
    if args.workflow or write_all:
        artifacts.update(
            {
                f"workflow_{key}": str(path)
                for key, path in write_workflow_bundle(args.output_dir).items()
            }
        )
    if args.container or write_all:
        artifacts.update(
            {
                f"container_{key}": str(path)
                for key, path in write_container_templates(args.output_dir).items()
            }
        )
    if args.quality or write_all:
        artifacts.update(
            {
                f"quality_{key}": str(path)
                for key, path in write_quality_tooling_templates(args.output_dir).items()
            }
        )
    print(json.dumps(artifacts, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
