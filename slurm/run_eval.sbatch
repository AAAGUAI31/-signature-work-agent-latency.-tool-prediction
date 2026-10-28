#!/bin/bash
#SBATCH --job-name=tau-eval-v2
#SBATCH --output=reports/slurm-eval-%j.out
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

ROOT="$HOME/projects/jev/experiments/tool_prediction"
SEMIF_ROOT="$HOME/projects/jev/SemIf-OpenJev"

source "$SEMIF_ROOT/.venv/bin/activate"
cd "$ROOT"

python scripts/run_eval.py \
  --gold data/splits_v2/gold.jsonl \
  --predictions frequency=predictions/frequency_v2/test.jsonl \
  --predictions markov=predictions/markov_v2/test.jsonl \
  --predictions semif=predictions/semif_direct_v2/test.jsonl \
  --predictions generation=predictions/qwen_generate_v2/test.jsonl \
  --output reports/metrics_v2.json \
  --split test
