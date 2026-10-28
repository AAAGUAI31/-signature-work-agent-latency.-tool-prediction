#!/bin/bash
#SBATCH --job-name=tau-generation-v2
#SBATCH --partition=common-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=reports/slurm-generation-%j.out

set -euo pipefail

SEMIF_ROOT="$HOME/projects/SemIf"
ROOT="$HOME/projects/jev/experiments/tool_prediction"
test -f "$SEMIF_ROOT/src/semif_phase1/core.py"
source "$SEMIF_ROOT/.venv/bin/activate"
cd "$ROOT"

export http_proxy=http://proxy-dku.oit.duke.edu:3128
export https_proxy=http://proxy-dku.oit.duke.edu:3128
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export HF_HOME=/work/jz466/huggingface
export TOKENIZERS_PARALLELISM=false

echo "=== GPU ==="
nvidia-smi

echo "=== Python ==="
which python

python scripts/run_baselines.py \
  --config configs/retail_v1.yaml \
  --models-config configs/models.yaml \
  --data-dir data/splits_v2 \
  --out-dir predictions/qwen_generate_v2 \
  --semif-repo "$SEMIF_ROOT" \
  --split test \
  --baseline generation
