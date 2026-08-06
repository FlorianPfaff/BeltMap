from __future__ import annotations

import json
from pathlib import Path

from beltmap.cli.write_templates import DEFAULT_WORKFLOW_CONFIG, main


def test_write_templates_creates_missing_snakemake_config(
    tmp_path: Path,
    capsys,
) -> None:
    assert main(["--output-dir", str(tmp_path), "--workflow"]) == 0

    payload = json.loads(capsys.readouterr().out)
    config_path = tmp_path / "beltmap-workflow-config.yaml"
    assert payload["workflow_snakemake_config"] == str(config_path)
    assert config_path.read_text(encoding="utf-8") == DEFAULT_WORKFLOW_CONFIG
    snakefile = (tmp_path / "Snakefile").read_text(encoding="utf-8")
    assert 'configfile: "beltmap-workflow-config.yaml"' in snakefile


def test_write_templates_preserves_existing_snakemake_config(
    tmp_path: Path,
    capsys,
) -> None:
    config_path = tmp_path / "beltmap-workflow-config.yaml"
    config_path.write_text("config: custom.toml\n", encoding="utf-8")

    assert main(["--output-dir", str(tmp_path), "--workflow"]) == 0

    capsys.readouterr()
    assert config_path.read_text(encoding="utf-8") == "config: custom.toml\n"
