import json

import numpy as np

from beltmap import _driver_runtime as runtime


def test_runtime_jsonable_replaces_nonfinite_values_recursively():
    converted = runtime.jsonable(
        {
            "native": [float("nan"), float("inf"), -float("inf")],
            "numpy_scalar": np.float64(np.nan),
            "numpy_array": np.asarray([1.0, np.inf]),
        }
    )

    assert converted == {
        "native": [None, None, None],
        "numpy_scalar": None,
        "numpy_array": [1.0, None],
    }
    json.dumps(converted, allow_nan=False)


def test_runtime_emit_writes_strict_json(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "OUT", tmp_path)
    monkeypatch.setattr(runtime, "rss_mb", lambda: None)

    runtime.emit(
        "test",
        "non-finite telemetry",
        metric=np.float64(np.nan),
        values=np.asarray([1.0, np.inf]),
    )

    latest_text = (tmp_path / "progress_latest.json").read_text(encoding="utf-8")
    jsonl_text = (tmp_path / "progress.jsonl").read_text(encoding="utf-8")
    assert "NaN" not in latest_text
    assert "Infinity" not in latest_text
    assert "NaN" not in jsonl_text
    assert "Infinity" not in jsonl_text

    payload = json.loads(latest_text)
    assert payload["metric"] is None
    assert payload["values"] == [1.0, None]
