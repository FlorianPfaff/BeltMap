from __future__ import annotations

import argparse
import json
from pathlib import Path

from beltmap.postrun_improvements import (
    DEFAULT_QUALITY_CONTRACT,
    evaluate_quality_contract,
    write_quality_contract_template,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="beltmap-quality-contract",
        description="Evaluate or write a compact BeltMap run quality contract.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--contract", type=Path, help="JSON contract file. Defaults to the built-in operational contract.")
    parser.add_argument("--write-template", type=Path, help="Write the default operational contract and exit.")
    parser.add_argument("--write-synthetic-template", type=Path, help="Write a synthetic-regression contract template and exit.")
    args = parser.parse_args(argv)

    if args.write_template is not None:
        write_quality_contract_template(args.write_template)
        return 0
    if args.write_synthetic_template is not None:
        write_quality_contract_template(args.write_synthetic_template, synthetic=True)
        return 0

    contract = DEFAULT_QUALITY_CONTRACT
    if args.contract is not None:
        contract = json.loads(args.contract.read_text(encoding="utf-8"))
    results = evaluate_quality_contract(args.output_dir, contract=contract)
    print(json.dumps([result.__dict__ for result in results], indent=2), flush=True)
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
