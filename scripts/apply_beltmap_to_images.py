"""Compatibility wrapper for the BeltMap image-sequence driver CLI."""

from __future__ import annotations

from beltmap.cli.apply import main as _beltmap_cli_main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_beltmap_cli_main())
