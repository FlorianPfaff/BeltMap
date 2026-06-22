from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
PYPROJECT = ROOT / "pyproject.toml"

ROOT_SCRIPT_ALLOWLIST = {
    "__init__.py",
    "apply_beltmap_to_images.py",
    "compare_raw_baselines.py",
    "filter_run_by_detection_overlap.py",
    "sweep_detection_overlap_filter.py",
    "check_script_taxonomy.py",
}

PAPER_EXPERIMENT_HINTS = (
    "brick",
    "sandlime",
    "10g",
    "15g",
    "20g",
    "50vs50",
    "90vs10",
    "10vs90",
    "specificity",
    "full100",
    "pilot25",
    "yolo",
    "ghost",
    "paper",
    "zenodo_",
)


def _root_scripts() -> set[str]:
    return {path.name for path in SCRIPTS_DIR.glob("*.py")}


def _project_scripts() -> dict[str, str]:
    payload = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return dict(payload.get("project", {}).get("scripts", {}))


def _module_path_exists(module_path: str) -> bool:
    module_name = module_path.split(":", 1)[0]
    rel = Path(*module_name.split(".")).with_suffix(".py")
    return (ROOT / rel).is_file()


def check_root_scripts() -> list[str]:
    errors: list[str] = []
    unexpected = sorted(_root_scripts() - ROOT_SCRIPT_ALLOWLIST)
    if unexpected:
        errors.append(
            "Unexpected root-level scripts: "
            + ", ".join(unexpected)
            + ". Move paper-specific helpers to scripts/paper_experiments/ "
            "or dev-only helpers to scripts/dev_or_archive/, or update the allowlist "
            "with a reason in scripts/README.md."
        )

    suspicious = [
        name
        for name in sorted(_root_scripts())
        if name not in {"__init__.py", "check_script_taxonomy.py"}
        and any(hint in name.lower() for hint in PAPER_EXPERIMENT_HINTS)
    ]
    if suspicious:
        errors.append(
            "Root-level scripts look paper/dataset-specific: "
            + ", ".join(suspicious)
            + ". Put them under scripts/paper_experiments/."
        )
    return errors


def check_supported_cli_modules() -> list[str]:
    errors: list[str] = []
    for command, target in sorted(_project_scripts().items()):
        if not target.startswith("beltmap.cli."):
            errors.append(f"Console script {command!r} should target beltmap.cli.*, got {target!r}.")
            continue
        if not _module_path_exists(target):
            errors.append(f"Console script {command!r} points to missing module {target!r}.")
    return errors


def check_required_directories() -> list[str]:
    errors: list[str] = []
    for rel in [
        "scripts/README.md",
        "scripts/paper_experiments/README.md",
        "scripts/dev_or_archive/README.md",
    ]:
        if not (ROOT / rel).is_file():
            errors.append(f"Missing script taxonomy file: {rel}")
    return errors


def main() -> int:
    errors = []
    errors.extend(check_required_directories())
    errors.extend(check_root_scripts())
    errors.extend(check_supported_cli_modules())

    if errors:
        print("Script taxonomy check failed:\n", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Script taxonomy check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
