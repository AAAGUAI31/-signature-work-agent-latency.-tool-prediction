from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tool_prediction.baselines import run_count_baseline
from tool_prediction.io import read_jsonl, write_jsonl
from tool_prediction.model_runners import run_same_weight_generation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frequency, Markov, or same-weight generation baselines.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--models-config", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--baseline", required=True, help="Comma-separated: frequency,markov,generation")
    parser.add_argument("--semif-repo", type=Path)
    parser.add_argument("--seed", type=int, default=20261026)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    methods = [value.strip() for value in args.baseline.split(",") if value.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for method in methods:
        output = args.out_dir / f"{args.split}.jsonl"
        if len(methods) > 1:
            output = args.out_dir / f"{args.split}_{method}.jsonl"
        if method in {"frequency", "markov"}:
            results[method] = run_count_baseline(args.data_dir, args.split, output, method)
        elif method == "generation":
            if args.models_config is None or args.semif_repo is None:
                parser.error("generation requires --models-config and --semif-repo")
            models = json.loads(args.models_config.read_text(encoding="utf-8"))
            model = models["model"]
            with tempfile.TemporaryDirectory(prefix="tau-tool-generation-") as directory:
                combined = Path(directory) / "input.jsonl"
                rows = list(read_jsonl(args.data_dir / f"{args.split}_gate.jsonl"))
                rows += list(read_jsonl(args.data_dir / f"{args.split}_tool_all.jsonl"))
                write_jsonl(combined, rows)
                results[method] = run_same_weight_generation(
                    combined,
                    output,
                    args.semif_repo,
                    model["name"],
                    model["revision"],
                    model["device"],
                    model["dtype"],
                    model["max_tokens"],
                    args.seed,
                )
        else:
            parser.error(f"unknown baseline: {method}")
    print(json.dumps({"split": args.split, "rows": results, "seed": config.get("seed")}))


if __name__ == "__main__":
    main()

