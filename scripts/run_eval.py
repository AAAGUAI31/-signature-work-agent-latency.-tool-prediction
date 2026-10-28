from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tool_prediction.metrics import evaluate_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate next-action predictions with task-cluster bootstrap CIs.")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", action="append", required=True, metavar="METHOD=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261026)
    args = parser.parse_args()
    paths = {}
    for item in args.predictions:
        if "=" not in item:
            parser.error("--predictions must be METHOD=PATH")
        method, path = item.split("=", 1)
        if not method or method in paths:
            parser.error(f"invalid or duplicate method: {method!r}")
        paths[method] = Path(path)
    report = evaluate_files(
        args.gold, paths, args.output, args.split, args.bootstrap_samples, args.seed
    )
    summary = {
        method: {
            "gate_accuracy": result["gate"]["accuracy"],
            "tool_hit_at_1": result["tool"]["hit_at_1"],
            "full_action_accuracy": result["full_action"]["accuracy"],
            "coverage": result["coverage"],
        }
        for method, result in report["methods"].items()
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

