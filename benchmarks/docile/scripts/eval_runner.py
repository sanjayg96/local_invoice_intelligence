import os
import json
import concurrent.futures
import time
import argparse
import statistics
from tqdm import tqdm
from docile.dataset import Dataset
import traceback

from config import (
    DOCILE_DATASET_PATH, 
    SUBSET_JSON_PATH, 
    EVAL_RESULTS_PATH, 
    VISION_MODEL,
    PIPELINE_MODE,
    ENABLE_THINKING,
    THERMAL_COOLDOWN_SECONDS,
    THERMAL_COOLDOWN_BATCH_SIZE,
    MODEL_TIMEOUT_SECONDS,
    OCR_BACKEND,
    BASELINE_PLUS_PRIMARY_MODEL,
    BASELINE_PLUS_RESCUE_MODEL,
    BLANK_ENSEMBLE_MODEL,
    MODEL_PROVIDER,
    OPENAI_MODEL,
    OLLAMA_THINKING,
)
from ollama_controls import thinking_label
from schema import InvoiceExtractionBase, InvoiceExtractionWithReasoning
from evaluate_metrics import FIELDS, score_record, summarize_scores

# Import our decoupled extraction engine
# from extractor import pdf_to_base64_images, run_2_step_extraction
from extractor import run_2_step_extraction
from baseline_plus import extract_baseline_plus, extract_blank_ensemble, flatten_baseline_plus_result


DEFAULT_FIELDS = ["vendor_name", "vendor_address", "amount_total_gross", "date_issue"]


def extract_ground_truth(doc) -> dict:
    """Extracts the text values for our specific target fields from DocILE."""
    targets = ["vendor_name", "vendor_address", "amount_total_gross", "date_issue"]
    gt_data = {t: None for t in targets}
    
    for field in doc.annotation.fields:
        if field.fieldtype in targets:
            if gt_data[field.fieldtype]:
                gt_data[field.fieldtype] += " " + field.text
            else:
                gt_data[field.fieldtype] = field.text
    return gt_data


def parse_args():
    parser = argparse.ArgumentParser(description="Run invoice extraction evaluation.")
    parser.add_argument(
        "--pipeline",
        choices=["baseline_plus", "blank_ensemble", "baseline"],
        default=PIPELINE_MODE,
        help=(
            "Extraction pipeline to run. Use baseline for direct extraction, "
            "blank_ensemble for 3B retries on blank primary fields only, or "
            "baseline_plus for broader validation/rescue."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of documents to process from the sorted subset.",
    )
    parser.add_argument(
        "--output",
        default=str(EVAL_RESULTS_PATH),
        help="Path for the evaluation report JSON.",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=THERMAL_COOLDOWN_SECONDS,
        help="Thermal cooldown seconds between batches. Set to 0 to disable.",
    )
    parser.add_argument(
        "--cooldown-batch-size",
        type=int,
        default=THERMAL_COOLDOWN_BATCH_SIZE,
        help="Number of documents per thermal cooldown batch.",
    )
    parser.add_argument(
        "--live-metrics-every",
        type=int,
        default=10,
        help="Print rolling extraction accuracy every N documents. Set to 0 to disable checkpoint prints.",
    )
    parser.add_argument(
        "--ocr-backend",
        choices=["tesseract", "rapidocr"],
        default=OCR_BACKEND,
        help="OCR fallback backend for pages with weak pdfplumber text. Defaults to config.OCR_BACKEND.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Primary model for extraction. Defaults to qwen3:14b for Ollama or OPENAI_MODEL for OpenAI.",
    )
    parser.add_argument(
        "--rescue-model",
        default=BASELINE_PLUS_RESCUE_MODEL,
        help="Ollama model for baseline_plus rescue extraction.",
    )
    parser.add_argument(
        "--blank-ensemble-model",
        default=BLANK_ENSEMBLE_MODEL,
        help="Ollama model used by blank_ensemble to retry only blank/null primary fields.",
    )
    parser.add_argument(
        "--ollama-thinking",
        choices=["auto", "false", "true", "low", "medium", "high"],
        default=OLLAMA_THINKING,
        help="Ollama built-in thinking mode. Use false to disable Qwen3 reasoning, true/low/medium/high to test it, or auto for model default.",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai"],
        default="ollama",
        help=(
            "Model provider. Defaults to ollama for local inference; pass openai "
            "explicitly for API benchmarking."
        ),
    )
    parser.add_argument(
        "--shadow-identity-rescue",
        action="store_true",
        help=(
            "For baseline_plus diagnostics, run rescue on soft vendor name/address "
            "issues without applying those candidates."
        ),
    )
    return parser.parse_args()


def _latency_summary(results_log: list[dict]) -> dict:
    latencies = [
        item.get("pipeline_metadata", {}).get("latency_s")
        for item in results_log
        if item.get("pipeline_metadata", {}).get("latency_s") is not None
    ]
    if not latencies:
        return {}
    ordered = sorted(float(value) for value in latencies)
    p95_index = min(len(ordered) - 1, int(len(ordered) * 0.95))
    return {
        "count": len(ordered),
        "avg_s": round(statistics.mean(ordered), 3),
        "p50_s": round(statistics.median(ordered), 3),
        "p95_s": round(ordered[p95_index], 3),
        "max_s": round(max(ordered), 3),
    }


def _format_live_summary(score_lists: dict[str, list[float]]) -> str:
    summary = summarize_scores(score_lists)
    fields = summary["fields"]
    total = summary["total_average"]
    if total is None:
        return "avg=N/A"
    return (
        f"avg={total:.1f}% "
        f"name={(fields.get('vendor_name') or 0):.1f}% "
        f"addr={(fields.get('vendor_address') or 0):.1f}% "
        f"amt={(fields.get('amount_total_gross') or 0):.1f}% "
        f"date={(fields.get('date_issue') or 0):.1f}%"
    )


def main():
    args = parse_args()
    if args.provider == "openai" and args.pipeline != "baseline":
        raise ValueError("OpenAI provider currently supports the direct baseline pipeline only. Use --pipeline baseline.")
    if args.model is None:
        args.model = OPENAI_MODEL if args.provider == "openai" else BASELINE_PLUS_PRIMARY_MODEL
    print(f"Loading DocILE dataset from {DOCILE_DATASET_PATH}...")
    dataset = Dataset("val", DOCILE_DATASET_PATH)
    
    with open(SUBSET_JSON_PATH, "r") as f:
        subset_ids = json.load(f)
        
    print(f"Loaded {len(subset_ids)} target documents.")
    
    # Dynamically select schema based on the config toggle
    ActiveSchema = InvoiceExtractionWithReasoning if ENABLE_THINKING else InvoiceExtractionBase
    print(f"Pipeline: {args.pipeline}")
    if args.pipeline == "baseline_plus":
        print("Baseline-plus extraction: full-transcript baseline + validation + targeted rescue")
        print(f"Models: primary={args.model}, rescue={args.rescue_model}")
    elif args.pipeline == "blank_ensemble":
        print("Blank ensemble extraction: Qwen primary + 3B field-only retries for blank/null fields")
        print(f"Models: primary={args.model}, blank_ensemble={args.blank_ensemble_model}")
    else:
        print(f"2-Step Pipeline: {args.model}")
    print(f"Provider: {args.provider}")
    print(f"OCR Backend: {args.ocr_backend}")
    print(f"Schema Scratchpad: {'ON' if ENABLE_THINKING else 'OFF'}")
    print(f"Ollama Thinking: {thinking_label(args.ollama_thinking)}")
    print(f"Thermal Cooldown: {args.cooldown_seconds}s per batch of {args.cooldown_batch_size} documents\n")
    
    results_log = []
    
    # Running a batch of 50 to test the pipeline and cooldown logic
    # test_subset = subset_ids[:100]
    test_subset = subset_ids[: args.limit] if args.limit else subset_ids
    
    # Initialize the executor OUTSIDE the loop
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    
    score_lists = {field: [] for field in FIELDS}
    progress = tqdm(test_subset, desc="Processing Invoices")

    for idx, doc_id in enumerate(progress):
        doc = dataset[doc_id]
        ground_truth = extract_ground_truth(doc)
        
        pdf_path = os.path.join(DOCILE_DATASET_PATH, "pdfs", f"{doc_id}.pdf")
        
        # 1. Image Conversion
        # images = pdf_to_base64_images(pdf_path)
        
        try:
            pipeline_result = None
            provider_metadata = {}
            processing_started = time.perf_counter()
            if args.pipeline == "baseline_plus":
                future = executor.submit(
                    extract_baseline_plus,
                    pdf_path,
                    DEFAULT_FIELDS,
                    primary_model=args.model,
                    rescue_model=args.rescue_model,
                    ocr_backend=args.ocr_backend,
                    ollama_thinking=args.ollama_thinking,
                    shadow_identity_rescue=args.shadow_identity_rescue,
                )
            elif args.pipeline == "blank_ensemble":
                future = executor.submit(
                    extract_blank_ensemble,
                    pdf_path,
                    DEFAULT_FIELDS,
                    primary_model=args.model,
                    ensemble_model=args.blank_ensemble_model,
                    ocr_backend=args.ocr_backend,
                    ollama_thinking=args.ollama_thinking,
                )
            else:
                # 2. Multi-Step Extraction
                future = executor.submit(
                    run_2_step_extraction,
                    pdf_path,
                    ActiveSchema,
                    args.ocr_backend,
                    args.ollama_thinking,
                    args.model,
                    args.provider,
                )
            
            # The strict watchdog using your custom timeout parameter
            response = future.result(timeout=MODEL_TIMEOUT_SECONDS)
            processing_latency_s = time.perf_counter() - processing_started
            if args.pipeline in {"baseline_plus", "blank_ensemble"}:
                pipeline_result = response
                predicted_data = flatten_baseline_plus_result(pipeline_result)
            else:
                predicted_data = json.loads(response['message']['content'])
                provider_metadata = response.get("provider_metadata", {})
                provider_response_latency_s = provider_metadata.get("latency_s")
                provider_metadata = {
                    **provider_metadata,
                    "pipeline": "baseline",
                    "provider": args.provider,
                    "model": args.model,
                    "ocr_backend": args.ocr_backend,
                    "latency_s": round(processing_latency_s, 4),
                }
                if provider_response_latency_s is not None:
                    provider_metadata["provider_response_latency_s"] = provider_response_latency_s
            
        except concurrent.futures.TimeoutError:
            print(f"\n[TIMEOUT] Document {doc_id} permanently stalled. Abandoning thread.")
            predicted_data = {"error": f"Timeout - Pipeline locked up within {MODEL_TIMEOUT_SECONDS} seconds"}
            pipeline_result = None
            executor.shutdown(wait=False, cancel_futures=True)
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            
        except Exception as e:
            print(f"\n[ERROR] Processing {doc_id}:")
            traceback.print_exc()  # <--- This will print the EXACT line of the crash
            predicted_data = {"error": "Pipeline Failure"}
            pipeline_result = None
            provider_metadata = {}

        # 3. Log Results
        record = {
            "doc_id": doc_id,
            "ground_truth": ground_truth,
            "predicted": predicted_data
        }
        if pipeline_result:
            record["pipeline_metadata"] = pipeline_result.get("metadata", {})
            record["field_results"] = pipeline_result.get("fields", {})
            record["candidates"] = pipeline_result.get("candidates", {})
            record["evidence_windows"] = pipeline_result.get("evidence_windows", {})
        elif provider_metadata:
            record["pipeline_metadata"] = provider_metadata
        results_log.append(record)

        for field, score in score_record(record).items():
            score_lists[field].append(score)
        live_summary = _format_live_summary(score_lists)
        progress.set_postfix_str(live_summary)
        if args.live_metrics_every > 0 and (idx + 1) % args.live_metrics_every == 0:
            tqdm.write(f"[Rolling accuracy after {idx + 1} docs] {live_summary}")
        
        # 4. Apply custom thermal cooldown
        if (
            args.cooldown_seconds > 0
            and args.cooldown_batch_size > 0
            and (idx + 1) % args.cooldown_batch_size == 0
            and idx < len(test_subset) - 1
        ):
            print(f"\n[Thermal Control] Cooling down for {args.cooldown_seconds}s...")
            time.sleep(args.cooldown_seconds)

    with open(args.output, "w") as f:
        json.dump(results_log, f, indent=4)
        
    latency = _latency_summary(results_log)
    if latency:
        print(f"\nLatency summary: {latency}")
    print(
        "\n✅ Evaluation complete. Run "
        f"`uv run python benchmarks/docile/scripts/evaluate_metrics.py --report {args.output}` to score."
    )

if __name__ == "__main__":
    main()
