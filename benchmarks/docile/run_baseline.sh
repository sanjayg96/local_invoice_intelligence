#!/usr/bin/env bash
set -euo pipefail

uv run python benchmarks/docile/scripts/eval_runner.py \
  --pipeline baseline \
  --provider ollama \
  --model qwen3:14b \
  --ollama-thinking false \
  --output results/eval_report_qwen3:14b_baseline.json

uv run python benchmarks/docile/scripts/evaluate_metrics.py --report results/eval_report_qwen3_14b_baseline.json
