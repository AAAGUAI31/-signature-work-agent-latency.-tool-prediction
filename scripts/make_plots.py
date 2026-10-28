from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tool_prediction.plots import make_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Create accuracy, latency, confusion, and reliability figures.")
    parser.add_argument("--reports", type=Path, required=True, help="metrics.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = make_plots(args.reports, args.output_dir)
    print(json.dumps([str(path) for path in paths], indent=2))


if __name__ == "__main__":
    main()

