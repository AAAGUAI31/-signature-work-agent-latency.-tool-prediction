#!/bin/bash
set -euo pipefail

ROOT="$HOME/projects/jev/experiments/tool_prediction"
SEMIF_ROOT="$HOME/projects/jev/SemIf-OpenJev"
VENV_ROOT="/work/$USER/venvs/semif-openjev"
SETUP_TMP="/work/$USER/tmp/semif-openjev"

cd "$ROOT"
test -f "$SEMIF_ROOT/pyproject.toml"
if [[ -e "$SEMIF_ROOT/.venv" ]]; then
  echo "$SEMIF_ROOT/.venv already exists; move it aside before requesting a clean environment"
  exit 2
fi
if [[ -e "$VENV_ROOT" ]]; then
  echo "$VENV_ROOT already exists; move it aside before requesting a clean environment"
  exit 2
fi

export HF_HOME=/work/jz466/huggingface
export TMPDIR="$SETUP_TMP"
export PIP_DISABLE_PIP_VERSION_CHECK=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

mkdir -p "$(dirname "$VENV_ROOT")" "$SETUP_TMP"

python3 -m venv "$VENV_ROOT"
ln -s "$VENV_ROOT" "$SEMIF_ROOT/.venv"
source "$SEMIF_ROOT/.venv/bin/activate"

python -m pip install --no-cache-dir --upgrade pip setuptools wheel
python -m pip install --no-cache-dir --no-build-isolation -e "${SEMIF_ROOT}[test]"
python -m pip install --no-cache-dir --no-build-isolation -e "${ROOT}[test,plots]"

python -c 'import torch, transformers; print({"python_torch": torch.__version__, "transformers": transformers.__version__, "torch_cuda_build": torch.version.cuda})'
PYTHONPATH="$ROOT/src" python -m pytest -q
