from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tool_prediction.windowing import window_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply an exact-token deterministic history window.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--semif-repo", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    args = parser.parse_args()
    accepted, rejected = window_dataset(
        args.input, args.output, args.rejected, args.semif_repo, args.model, args.revision, args.max_tokens
    )
    print(json.dumps({"accepted": accepted, "rejected": rejected}))


if __name__ == "__main__":
    main()

