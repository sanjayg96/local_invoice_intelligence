#!/usr/bin/env bash
set -euo pipefail

uv run python benchmarks/docile/scripts/eval_runner.py \
  --pipeline blank_ensemble \
  --provider ollama \
  --model qwen3:14b \
  --blank-ensemble-model qwen3:14b \
  --ollama-thinking false \
  --output results/eval_report_qwen3_14b_blank_ensemble_qwen3_14b.json

uv run python benchmarks/docile/scripts/evaluate_metrics.py --report results/eval_report_qwen3_14b_blank_ensemble_qwen3_14b.json
