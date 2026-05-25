import os
import json
import concurrent.futures
import time
from tqdm import tqdm
from docile.dataset import Dataset

from config import (
    DOCILE_DATASET_PATH, 
    SUBSET_JSON_PATH, 
    EVAL_RESULTS_PATH, 
    VISION_MODEL,
    TEXT_MODEL,
    ENABLE_THINKING,
    THERMAL_COOLDOWN_SECONDS,
    THERMAL_COOLDOWN_BATCH_SIZE,
    MODEL_TIMEOUT_SECONDS
)
from schema import InvoiceExtractionBase, InvoiceExtractionWithReasoning

# Import our decoupled extraction engine
from extractor import pdf_to_base64_images, run_2_step_extraction


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


def main():
    print(f"Loading DocILE dataset from {DOCILE_DATASET_PATH}...")
    dataset = Dataset("val", DOCILE_DATASET_PATH)
    
    with open(SUBSET_JSON_PATH, "r") as f:
        subset_ids = json.load(f)
        
    print(f"Loaded {len(subset_ids)} target documents.")
    
    # Dynamically select schema based on the config toggle
    ActiveSchema = InvoiceExtractionWithReasoning if ENABLE_THINKING else InvoiceExtractionBase
    print(f"2-Step Pipeline: {VISION_MODEL} -> {TEXT_MODEL}")
    print(f"Thinking Mode: {'ON' if ENABLE_THINKING else 'OFF'}")
    print(f"Thermal Cooldown: {THERMAL_COOLDOWN_SECONDS}s per batch of {THERMAL_COOLDOWN_BATCH_SIZE} documents\n")
    
    results_log = []
    
    # Running a batch of 50 to test the pipeline and cooldown logic
    test_subset = subset_ids[:100]
    
    # Initialize the executor OUTSIDE the loop
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    
    for idx, doc_id in enumerate(tqdm(test_subset, desc="Processing Invoices")):
        doc = dataset[doc_id]
        ground_truth = extract_ground_truth(doc)
        
        pdf_path = os.path.join(DOCILE_DATASET_PATH, "pdfs", f"{doc_id}.pdf")
        
        # 1. Image Conversion
        images = pdf_to_base64_images(pdf_path)
        
        try:
            # 2. Multi-Step Extraction
            future = executor.submit(run_2_step_extraction, images, ActiveSchema)
            
            # The strict watchdog using your custom timeout parameter
            response = future.result(timeout=MODEL_TIMEOUT_SECONDS)
            predicted_data = json.loads(response['message']['content'])
            
        except concurrent.futures.TimeoutError:
            print(f"\n[TIMEOUT] Document {doc_id} permanently stalled. Abandoning thread.")
            predicted_data = {"error": f"Timeout - Pipeline locked up within {MODEL_TIMEOUT_SECONDS} seconds"}
            
        except Exception as e:
            print(f"\n[ERROR] Processing {doc_id}: {e}")
            predicted_data = {"error": str(e)}

        # 3. Log Results
        results_log.append({
            "doc_id": doc_id,
            "ground_truth": ground_truth,
            "predicted": predicted_data
        })
        
        # 4. Apply custom thermal cooldown
        if (idx + 1) % THERMAL_COOLDOWN_BATCH_SIZE == 0 and idx < len(test_subset) - 1:
            print(f"\n[Thermal Control] Cooling down for {THERMAL_COOLDOWN_SECONDS}s...")
            time.sleep(THERMAL_COOLDOWN_SECONDS)

    with open(EVAL_RESULTS_PATH, "w") as f:
        json.dump(results_log, f, indent=4)
        
    print(f"\n✅ Evaluation complete. Run `uv run python src/evaluate_metrics.py` to score.")

if __name__ == "__main__":
    main()