from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tool_prediction.io import read_jsonl, write_jsonl
from tool_prediction.model_runners import run_semif_direct


def main() -> None:
    parser = argparse.ArgumentParser(description="Run warmed SemIf direct scoring for gate and tool rows.")
    parser.add_argument("--models-config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--semif-repo", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261026)
    args = parser.parse_args()
    config = json.loads(args.models_config.read_text(encoding="utf-8"))["model"]
    with tempfile.TemporaryDirectory(prefix="tau-tool-semif-") as directory:
        combined = Path(directory) / "input.jsonl"
        rows = list(read_jsonl(args.data_dir / f"{args.split}_gate.jsonl"))
        rows += list(read_jsonl(args.data_dir / f"{args.split}_tool_all.jsonl"))
        write_jsonl(combined, rows)
        count = run_semif_direct(
            combined,
            args.output,
            args.semif_repo,
            config["name"],
            config["revision"],
            config["device"],
            config["dtype"],
            config["max_tokens"],
            args.seed,
        )
    print(json.dumps({"rows": count, "output": str(args.output)}))


if __name__ == "__main__":
    main()

