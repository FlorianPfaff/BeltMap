from __future__ import annotations

import argparse
import json
from pathlib import Path

from beltmap.operational_improvements import DatasetManifest, dataset_manifest


def _paths_alias(first: Path, second: Path) -> bool:
    """Return whether two path spellings identify the same filesystem object."""

    try:
        return first.samefile(second)
    except (FileNotFoundError, OSError):
        return first.resolve(strict=False) == second.resolve(strict=False)


def _reject_output_image_alias(
    parser: argparse.ArgumentParser,
    *,
    image_dir: Path,
    manifest: DatasetManifest,
    output: Path,
) -> None:
    """Prevent the manifest JSON from replacing one of its input images."""

    for record in manifest.files:
        image_path = image_dir / record.path
        if _paths_alias(output, image_path):
            parser.error(
                f"--output must not overwrite input image {image_path}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write a reproducibility manifest for an image directory."
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data_manifest.json"))
    args = parser.parse_args(argv)
    manifest = dataset_manifest(args.image_dir)
    _reject_output_image_alias(
        parser,
        image_dir=args.image_dir,
        manifest=manifest,
        output=args.output,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest.to_dict(), indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "files": len(manifest.files),
                "manifest_sha256": manifest.manifest_sha256,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
