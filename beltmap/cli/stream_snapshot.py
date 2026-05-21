from __future__ import annotations

import argparse
import json
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Snapshot new image files for a future streaming BeltMap driver.")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--state", type=Path, default=Path("stream_state.json"))
    parser.add_argument("--max-new", type=parse_nonnegative_int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    state = StreamingFrameState()
    if args.state.is_file():
        payload = json.loads(args.state.read_text(encoding="utf-8"))
        state.seen_paths = set(payload.get("seen_paths", []))
        state.last_scan_unix_s = float(payload.get("last_scan_unix_s", 0.0))
    new_paths = discover_new_stream_frames(args.image_dir, state, max_new=None if args.max_new == 0 else args.max_new)
    args.state.write_text(json.dumps({"seen_paths": sorted(state.seen_paths), "last_scan_unix_s": state.last_scan_unix_s}, indent=2), encoding="utf-8")
    print(json.dumps({"new_frames": [str(path) for path in new_paths], "state": str(args.state)}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
