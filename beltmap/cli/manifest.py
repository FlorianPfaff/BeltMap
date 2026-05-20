from __future__ import annotations

import argparse
import json
from pathlib import Path

from beltmap.operational_improvements import dataset_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a reproducibility manifest for an image directory.")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data_manifest.json"))
    args = parser.parse_args(argv)
    manifest = dataset_manifest(args.image_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps({"files": len(manifest.files), "manifest_sha256": manifest.manifest_sha256}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
