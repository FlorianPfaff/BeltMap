from __future__ import annotations

import argparse
import itertools
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def parse_scalar(value: str) -> Any:
    text = value.strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if text.lower() in {"none", "null"}:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def parse_param(value: str) -> tuple[str, list[Any]]:
    if "=" not in value:
        raise ValueError("parameters must be KEY=VALUE1,VALUE2")
    key, raw_values = value.split("=", 1)
    values = [parse_scalar(item) for item in raw_values.split(",")]
    return key.strip(), values


def set_dotted(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    target = data
    for key in keys[:-1]:
        child = target.setdefault(key, {})
        if not isinstance(child, dict):
            raise ValueError(f"cannot set nested key below non-table {key!r}")
        target = child
    target[keys[-1]] = value


def toml_value(value: Any) -> str:
    if value is None:
        return '""'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    return json.dumps(str(value))


def write_toml(data: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    flat_items = [(key, value) for key, value in data.items() if not isinstance(value, dict)]
    for key, value in flat_items:
        lines.append(f"{key} = {toml_value(value)}")
    for section, values in data.items():
        if not isinstance(values, dict):
            continue
        lines.append("")
        lines.append(f"[{section}]")
        for key, value in values.items():
            if isinstance(value, dict):
                nested_section = f"{section}.{key}"
                lines.append("")
                lines.append(f"[{nested_section}]")
                for nested_key, nested_value in value.items():
                    lines.append(f"{nested_key} = {toml_value(nested_value)}")
            else:
                lines.append(f"{key} = {toml_value(value)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beltmap-sweep",
        description="Generate or execute BeltMap parameter sweep configurations.",
    )
    parser.add_argument("--base-config", type=Path, required=True, help="Base TOML config.")
    parser.add_argument("--param", action="append", default=[], help="Dotted key and comma-separated values, e.g. detection.threshold=3.5,4.0.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/sweeps"), help="Directory for generated run configs and outputs.")
    parser.add_argument("--execute", action="store_true", help="Run beltmap-apply and beltmap-validate for each generated config.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base = tomllib.loads(args.base_config.read_text(encoding="utf-8"))
    params = [parse_param(item) for item in args.param]
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    keys = [key for key, _values in params]
    value_grid = itertools.product(*(values for _key, values in params)) if params else [()]
    for run_index, values in enumerate(value_grid):
        config = json.loads(json.dumps(base))
        run_dir = args.output_root / f"run_{run_index:03d}"
        set_dotted(config, "paths.output_dir", str(run_dir))
        overrides = dict(zip(keys, values))
        for key, value in overrides.items():
            set_dotted(config, key, value)
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "beltmap.toml"
        write_toml(config, config_path)
        manifest.append({"run_index": run_index, "config": str(config_path), "output_dir": str(run_dir), "overrides": overrides})
        if args.execute:
            subprocess.run(["beltmap-apply", "--config", str(config_path)], check=True)
            if shutil.which("beltmap-validate"):
                subprocess.run(["beltmap-validate", "--output-dir", str(run_dir)], check=True)
    manifest_path = args.output_root / "sweep_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
