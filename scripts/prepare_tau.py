from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tool_prediction.prepare import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe gate and tool prediction datasets.")
    parser.add_argument("--input", type=Path, action="append", required=True, help="Repeat for each actor trajectory file.")
    parser.add_argument("--upstream-repo", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261026)
    args = parser.parse_args()
    report = prepare_dataset(args.input, args.upstream_repo, args.out_dir, args.seed)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

