#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python examples/synthetic_sequence/generate.py --output-dir data/images --frames 12
beltmap-apply --config examples/synthetic_sequence/beltmap.toml --print-config
beltmap-validate --output-dir outputs --quiet
beltmap-benchmark \
  --output-dir outputs \
  --truth-path data/images/synthetic_metadata.json \
  --quiet
python examples/synthetic_sequence/validate_outputs.py \
  --output-dir outputs \
  --expected-frames 12 \
  --expected-velocity 2.0
