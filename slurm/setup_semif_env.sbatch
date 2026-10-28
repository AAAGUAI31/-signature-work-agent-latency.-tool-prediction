#!/bin/bash
#SBATCH --job-name=semif-env
#SBATCH --output=reports/slurm-setup-semif-env-%j.out
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

ROOT="$HOME/projects/jev/experiments/tool_prediction"
SEMIF_ROOT="$HOME/projects/jev/SemIf-OpenJev"

cd "$ROOT"
test -f "$SEMIF_ROOT/pyproject.toml"
if [[ -e "$SEMIF_ROOT/.venv" ]]; then
  echo "$SEMIF_ROOT/.venv already exists; move it aside before requesting a clean environment"
  exit 2
fi

export http_proxy=http://proxy-dku.oit.duke.edu:3128
export https_proxy=http://proxy-dku.oit.duke.edu:3128
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export HF_HOME=/work/jz466/huggingface

python3 -m venv "$SEMIF_ROOT/.venv"
source "$SEMIF_ROOT/.venv/bin/activate"

python -m pip install --upgrade pip
python -m pip install -e "${SEMIF_ROOT}[test]"
python -m pip install -e "${ROOT}[test,plots]"

python -c 'import torch, transformers; print({"python_torch": torch.__version__, "transformers": transformers.__version__, "torch_cuda_build": torch.version.cuda})'
PYTHONPATH="$ROOT/src" python -m pytest -q
