#!/bin/bash
#SBATCH --job-name=tau-prepare-v2
#SBATCH --output=reports/slurm-prepare-v2-%j.out
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

ROOT="$HOME/projects/jev/experiments/tool_prediction"
SEMIF_ROOT="$HOME/projects/jev/SemIf-OpenJev"

source "$SEMIF_ROOT/.venv/bin/activate"
cd "$ROOT"

PYTHONPATH="$ROOT/src" python -m pytest -q

python scripts/prepare_tau.py \
  --input ../../tau-bench/historical_trajectories/gpt-4o-retail.json \
  --upstream-repo ../../tau-bench \
  --out-dir data/splits_v2 \
  --seed 20261026

python scripts/validate_dataset.py --data-dir data/splits_v2

python scripts/run_baselines.py \
  --config configs/retail_v1.yaml \
  --data-dir data/splits_v2 \
  --out-dir predictions/frequency_v2 \
  --split test \
  --baseline frequency

python scripts/run_baselines.py \
  --config configs/retail_v1.yaml \
  --data-dir data/splits_v2 \
  --out-dir predictions/markov_v2 \
  --split test \
  --baseline markov
