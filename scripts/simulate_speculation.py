from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tool_prediction.metrics import simulate_speculation


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay the client-side speculative-tool latency model.")
    parser.add_argument("--accuracy", type=float, required=True)
    parser.add_argument("--predictor-seconds", type=float, required=True)
    parser.add_argument("--actor-seconds", type=float, required=True)
    parser.add_argument("--tool-seconds", type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(simulate_speculation(args.accuracy, args.predictor_seconds, args.actor_seconds, args.tool_seconds), indent=2))


if __name__ == "__main__":
    main()

