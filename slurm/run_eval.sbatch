#!/bin/bash
#SBATCH --job-name=tau-eval
#SBATCH --output=reports/slurm-eval-%j.out
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

ROOT="$HOME/projects/jev/experiments/tool_prediction"

source "$HOME/projects/SemIf/.venv/bin/activate"
cd "$ROOT"

python scripts/run_eval.py \
  --gold data/splits/gold.jsonl \
  --predictions frequency=predictions/frequency/test.jsonl \
  --predictions markov=predictions/markov/test.jsonl \
  --predictions semif=predictions/semif_direct/test.jsonl \
  --predictions generation=predictions/qwen_generate/test.jsonl \
  --output reports/metrics.json \
  --split test