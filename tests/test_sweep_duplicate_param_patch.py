import json
import subprocess
import sys
from pathlib import Path

import pytest

from beltmap.cli import sweep as cli_sweep
from beltmap.cli import sweep_duplicate_param_patch as duplicate_patch


def write_base_config(path: Path, image_dir: Path) -> None:
    path.write_text(
        f"""[paths]
image_dir = {json.dumps(str(image_dir))}
output_dir = "unused"

[detection]
threshold = 3.0
low_threshold = 0.0
""",
        encoding="utf-8",
    )


def duplicate_param_args(tmp_path: Path) -> tuple[list[str], Path]:
    base_config = tmp_path / "beltmap.toml"
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    write_base_config(base_config, image_dir)
    output_root = tmp_path / "sweep"
    return (
        [
            "--base-config",
            str(base_config),
            "--param",
            "detection.threshold=2.0,3.0",
            "--param",
            " detection . threshold =4.0,5.0",
            "--output-root",
            str(output_root),
        ],
        output_root,
    )


def test_duplicate_sweep_param_keys_use_dotted_key_semantics():
    assert duplicate_patch.duplicate_sweep_param_keys(
        [
            "detection.threshold=2.0",
            " detection . threshold =3.0",
            "tracking.max_distance=5.0",
        ]
    ) == ["detection.threshold"]


def test_sweep_main_rejects_duplicate_param_keys_before_writing(tmp_path, capsys):
    args, output_root = duplicate_param_args(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        cli_sweep.main(args)

    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "sweep parameter keys must be unique" in stderr
    assert "detection.threshold" in stderr
    assert not output_root.exists()


def test_sweep_module_rejects_duplicate_param_keys_before_writing(tmp_path):
    args, output_root = duplicate_param_args(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "-m", "beltmap.cli.sweep", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "sweep parameter keys must be unique" in result.stderr
    assert "detection.threshold" in result.stderr
    assert not output_root.exists()
