#!/bin/bash
set -euo pipefail

ROOT="$HOME/projects/jev/experiments/tool_prediction"
SEMIF_VENV="$HOME/projects/jev/SemIf-OpenJev/.venv/bin/python"
cd "$ROOT"
test -x "$SEMIF_VENV"

prepare_job="$(sbatch --parsable slurm/prepare_v2.sbatch)"
semif_job="$(sbatch --parsable --dependency="afterok:$prepare_job" slurm/run_semif.sbatch)"
generation_job="$(sbatch --parsable --dependency="afterok:$prepare_job" slurm/run_generation.sbatch)"
eval_job="$(sbatch --parsable --dependency="afterok:$semif_job:$generation_job" slurm/run_eval.sbatch)"

printf 'prepare=%s\nsemif=%s\ngeneration=%s\neval=%s\n' \
  "$prepare_job" "$semif_job" "$generation_job" "$eval_job"
