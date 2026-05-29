import subprocess
import sys


def test_tracking_import_defaults_to_headless_matplotlib_backend():
    code = (
        "import os; "
        "os.environ.pop('MPLBACKEND', None); "
        "import beltmap.tracking; "
        "print(os.environ.get('MPLBACKEND'))"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "Agg"
