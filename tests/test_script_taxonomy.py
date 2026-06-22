from __future__ import annotations

from scripts import check_script_taxonomy


def test_script_taxonomy_check_passes() -> None:
    errors: list[str] = []
    errors.extend(check_script_taxonomy.check_required_directories())
    errors.extend(check_script_taxonomy.check_root_scripts())
    errors.extend(check_script_taxonomy.check_supported_cli_modules())

    assert errors == []
