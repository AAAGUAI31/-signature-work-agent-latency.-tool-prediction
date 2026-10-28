# SemIf / Jev-style next-tool prediction

This directory implements the offline E0-E4 experiment for original tau-bench Retail. It treats the recorded actor's next assistant action as ground truth, keeps reference task actions and reward metadata out of model inputs, and evaluates a two-stage `call_tool/respond` gate followed by the canonical 16-way Retail tool selector.

The checked upstream snapshots are:

- tau-bench `59a200c6d575d595120f1cb70fea53cef0632f6b`
- SemIf-OpenJev `23cf1f39fc9534fe81437200959b6dfc7106e45a`
- Qwen3.5-4B `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`

SemIf results produced here are open-model Jev-style typed decisions, not measurements of the proprietary TypeSafe Jev product.

## Setup

From this directory:

```bash
SEMIF_ROOT="$HOME/projects/jev/SemIf-OpenJev"
source "$SEMIF_ROOT/.venv/bin/activate"
pip install -e '.[test,plots]'
pip install -e "$SEMIF_ROOT"
```

GPU model runs must be submitted on a DKUCC compute node. The GPU scripts request one A40 from `common-gpu`; CPU preparation and evaluation use the cluster default CPU partition.

Create the dedicated DKUCC environment once from the login node:

```bash
bash scripts/setup_semif_env.sh
```

The setup script uses the login node's configured package index, creates the physical environment under `/work/$USER/venvs/semif-openjev`, and links it as `~/projects/jev/SemIf-OpenJev/.venv`. Downloads use `/work/$USER/tmp/semif-openjev` and disable the pip cache so large CUDA wheels do not fill the home or login-node temporary filesystem. It installs the pinned OpenJev and experiment dependencies and runs the test suite. Do not copy an existing virtual environment between directories. Model inference remains restricted to Slurm GPU jobs.

## Data preparation

```bash
python scripts/inspect_raw.py \
  --input ../../tau-bench/historical_trajectories/gpt-4o-retail.json \
  --output reports/raw_audit.json

python scripts/prepare_tau.py \
  --input ../../tau-bench/historical_trajectories/gpt-4o-retail.json \
  --upstream-repo ../../tau-bench \
  --out-dir data/splits \
  --seed 20261026

python scripts/validate_dataset.py --data-dir data/splits
```

Preparation writes one gate row per valid assistant action, one scored tool row per recorded tool call, and an additional `*_tool_all.jsonl` inference set for measuring conditional two-stage latency without using labels. Ground truth remains in `gold.jsonl` and is never embedded in a scorer input.

## Baselines and models

CPU baselines:

```bash
python scripts/run_baselines.py \
  --config configs/retail_v1.yaml \
  --data-dir data/splits \
  --out-dir predictions/frequency \
  --split test \
  --baseline frequency

python scripts/run_baselines.py \
  --config configs/retail_v1.yaml \
  --data-dir data/splits \
  --out-dir predictions/markov \
  --split test \
  --baseline markov
```

The GPU runs use the same checkpoint, revision, state, deterministic option order, dtype, device, and batch size. SemIf reads the restricted option logits at the final position; generation uses the same SemIf prompt and generates exactly one option letter with greedy decoding. Both warm the loaded model once before recorded rows. Gate answer slots are counterbalanced deterministically by sample ID. Model inference applies the same exact-token history window to both methods, preserving the policy and the most recent complete user-started blocks when a full trajectory exceeds the configured input budget.

```bash
bash slurm/submit_v2.sh
```

The submission script schedules CPU data preparation and baselines, the two GPU methods after preparation succeeds, and final CPU evaluation after both GPU jobs succeed. It preserves the original outputs and writes the new run to `data/splits_v2`, `predictions/*_v2`, and `reports/metrics_v2.json`.

For context-budget experiments, window each gate and tool input separately. The command uses the exact SemIf prompt tokenizer, retains the policy/system message and most recent complete user-started blocks, records omissions, and writes every unrepresentable row to the rejected sidecar.

```bash
python scripts/window_dataset.py \
  --input data/splits/test_gate.jsonl \
  --output data/interim/test_gate_1024.jsonl \
  --rejected data/interim/test_gate_1024_rejected.jsonl \
  --semif-repo "$HOME/projects/jev/SemIf-OpenJev" \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --max-tokens 1024
```

## Evaluation and calibration

```bash
python scripts/run_eval.py \
  --gold data/splits/gold.jsonl \
  --predictions frequency=predictions/frequency/test.jsonl \
  --predictions markov=predictions/markov/test.jsonl \
  --predictions semif=predictions/semif_direct/test.jsonl \
  --predictions generation=predictions/qwen_generate/test.jsonl \
  --output reports/metrics.json \
  --split test

python scripts/calibrate_predictions.py \
  --gold data/splits/gold.jsonl \
  --val-predictions predictions/semif_direct/val.jsonl \
  --test-predictions predictions/semif_direct/test.jsonl \
  --calibrated-out predictions/semif_direct/test_calibrated.jsonl \
  --report reports/calibration.json

python scripts/make_plots.py \
  --reports reports/metrics.json \
  --output-dir reports/figures
```

`run_eval.py` joins by sample ID, counts missing and failed predictions in coverage and accuracy denominators, reports Gate Accuracy/Macro F1, Tool Hit@1/3/Macro F1, Full-action Accuracy, latency percentiles, reliability metrics, and task-cluster bootstrap confidence intervals.

## Tests

```bash
pytest -q
```

The tests cover leakage boundaries, grouped splits, action parsing, canonical options, ID-based joins, exact-call semantics, failure coverage, bootstrap grouping, and the speculative latency equation.

## Known limits

- E0-E4 are implemented. Argument candidate generation and online tool execution remain separate second-stage work.
- `window_dataset.py` needs the pinned tokenizer files; on a new machine, cache them before running an offline Slurm job.
- Client-side replay assumes no shared-resource contention. It reports wasted tool work separately and does not claim an online speedup.
