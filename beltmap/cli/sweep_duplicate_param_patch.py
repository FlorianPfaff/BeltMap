"""Reject duplicate parameter dimensions before ``beltmap-sweep`` executes."""

from __future__ import annotations

import argparse
from typing import Any

_PATCHED_ATTR = "_beltmap_sweep_duplicate_param_patched"
_ORIGINAL_ATTR = "_beltmap_sweep_duplicate_param_original_parse_args"


def _canonical_param_key(value: Any) -> str | None:
    """Return the dotted key as interpreted by the sweep CLI, when valid."""

    if not isinstance(value, str) or "=" not in value:
        return None
    raw_key = value.split("=", 1)[0].strip()
    parts = [part.strip() for part in raw_key.split(".")]
    if not parts or any(part == "" for part in parts):
        return None
    return ".".join(parts)


def duplicate_sweep_param_keys(values: Any) -> list[str]:
    """Return repeated semantic ``--param`` keys in first-repeat order."""

    if not isinstance(values, (list, tuple)):
        return []
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        key = _canonical_param_key(value)
        if key is None:
            continue
        if key in seen and key not in duplicates:
            duplicates.append(key)
        seen.add(key)
    return duplicates


_current_parse_args = argparse.ArgumentParser.parse_args
_original_parse_args = getattr(_current_parse_args, _ORIGINAL_ATTR, _current_parse_args)


def parse_args_reject_duplicate_sweep_params(
    self: argparse.ArgumentParser,
    args: list[str] | None = None,
    namespace: argparse.Namespace | None = None,
) -> argparse.Namespace:
    """Reject dimensions that ``dict(zip(...))`` would silently collapse."""

    parsed = _original_parse_args(self, args, namespace)
    if self.prog == "beltmap-sweep":
        duplicates = duplicate_sweep_param_keys(getattr(parsed, "param", None))
        if duplicates:
            repeated = ", ".join(repr(key) for key in duplicates)
            self.error(
                "sweep parameter keys must be unique; repeated keys would "
                f"collapse sweep dimensions: {repeated}"
            )
    return parsed


setattr(parse_args_reject_duplicate_sweep_params, _PATCHED_ATTR, True)
setattr(
    parse_args_reject_duplicate_sweep_params,
    _ORIGINAL_ATTR,
    _original_parse_args,
)
argparse.ArgumentParser.parse_args = parse_args_reject_duplicate_sweep_params
