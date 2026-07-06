import importlib
import tomllib
from pathlib import Path


def test_ghost_repair_console_script_resolves_to_cli_module():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    target = pyproject["project"]["scripts"]["beltmap-ghost-repair"]
    module_name, separator, attribute = target.partition(":")

    assert separator == ":"
    module = importlib.import_module(module_name)

    resolved = module
    for part in attribute.split("."):
        resolved = getattr(resolved, part)

    assert callable(resolved)
