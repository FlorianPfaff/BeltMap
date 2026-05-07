"""Packaged image-sequence driver entry point for BeltMap."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    """Run the BeltMap image-sequence driver.

    The implementation currently delegates to the legacy script module while the
    driver internals are being moved under :mod:`beltmap`. Keeping this public
    package entry point lets CLIs and users stop depending on the top-level
    ``scripts`` package.
    """

    from scripts import apply_beltmap_to_images as legacy_driver

    legacy_driver.refresh_runtime_paths = getattr(
        legacy_driver,
        "refresh_runtime_paths",
        lambda: None,
    )
    legacy_driver.DATA = Path(__import__("os").getenv("BELTMAP_IMAGE_DIR", "data/images"))
    legacy_driver.OUT = Path(__import__("os").getenv("BELTMAP_OUTPUT_DIR", "outputs"))
    legacy_driver.main()
