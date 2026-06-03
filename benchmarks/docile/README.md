# DocILE Benchmark

This folder is for reproducing the benchmark evidence cited in the project README.

The original benchmark scripts are preserved under `benchmarks/docile/scripts/`:

- `benchmarks/docile/scripts/eval_runner.py`
- `benchmarks/docile/scripts/evaluate_metrics.py`
- `benchmarks/docile/scripts/baseline_plus.py`
- `benchmarks/docile/scripts/analyze_baseline_plus.py`
- `benchmarks/docile/scripts/rca_runner.py`

Run the best local direct baseline:

```bash
bash benchmarks/docile/run_baseline.sh
```

Run the blank-only 3B ensemble experiment:

```bash
bash benchmarks/docile/run_blank_ensemble.sh
```

This keeps `qwen3:14b` as the primary extractor and calls `llama3.2:3b`
only for fields where the primary final value is blank/null. The 3B candidate
is accepted only when it passes the existing field validation and evidence
grounding checks.

Generated result JSON files are intentionally not checked in. Re-run the scripts locally after downloading DocILE.
