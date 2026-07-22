from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from beltmap.operational_improvements import StreamingFrameState, discover_new_stream_frames


def parse_nonnegative_int(value: str) -> int:
    """Parse a non-negative integer CLI argument."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def load_state(path: Path) -> StreamingFrameState:
    """Load and validate a persisted streaming snapshot state."""

    state = StreamingFrameState()
    if not path.is_file():
        return state

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"stream state is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("stream state must be a JSON object")

    seen_paths = payload.get("seen_paths", [])
    if not isinstance(seen_paths, list) or any(
        not isinstance(item, str) for item in seen_paths
    ):
        raise ValueError("stream state seen_paths must be a JSON array of strings")

    last_scan = payload.get("last_scan_unix_s", 0.0)
    if isinstance(last_scan, bool):
        raise ValueError("stream state last_scan_unix_s must be finite and non-negative")
    try:
        last_scan_value = float(last_scan)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "stream state last_scan_unix_s must be finite and non-negative"
        ) from exc
    if not math.isfinite(last_scan_value) or last_scan_value < 0.0:
        raise ValueError("stream state last_scan_unix_s must be finite and non-negative")

    state.seen_paths = set(seen_paths)
    state.last_scan_unix_s = last_scan_value
    return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Snapshot new image files for a future streaming BeltMap driver.")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--state", type=Path, default=Path("stream_state.json"))
    parser.add_argument("--max-new", type=parse_nonnegative_int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    state = load_state(args.state)
    new_paths = discover_new_stream_frames(args.image_dir, state, max_new=None if args.max_new == 0 else args.max_new)
    args.state.parent.mkdir(parents=True, exist_ok=True)
    args.state.write_text(json.dumps({"seen_paths": sorted(state.seen_paths), "last_scan_unix_s": state.last_scan_unix_s}, indent=2), encoding="utf-8")
    print(json.dumps({"new_frames": [str(path) for path in new_paths], "state": str(args.state)}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
