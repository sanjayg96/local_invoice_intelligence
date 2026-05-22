import os
import json
import concurrent.futures
import base64
import fitz  # PyMuPDF
import ollama
import time
from tqdm import tqdm
from docile.dataset import Dataset

from config import (
    DOCILE_DATASET_PATH, 
    SUBSET_JSON_PATH, 
    EVAL_RESULTS_PATH, 
    MODEL_NAME, 
    ENABLE_THINKING,
    THERMAL_COOLDOWN_SECONDS,
    THERMAL_COOLDOWN_BATCH_SIZE,
    MODEL_TIMEOUT_SECONDS
)
from schema import InvoiceExtractionBase, InvoiceExtractionWithReasoning

def pdf_to_base64_image(pdf_path: str, max_dimension: int = 1024) -> str:
    """Converts the first page of a PDF into a base64 encoded JPEG, scaling large files down."""
    doc = fitz.open(pdf_path)
    page = doc.load_page(0) 
    
    # Calculate scaling matrix to cap the longest edge
    rect = page.rect
    scale = min(1.0, max_dimension / max(rect.width, rect.height))
    mat = fitz.Matrix(scale, scale)
    
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("jpeg")
    doc.close()
    return base64.b64encode(img_bytes).decode('utf-8')

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


def call_ollama(model_name, system_prompt, b64_image, schema):
    """Wrapper function to execute the API call."""
    return ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Extract the vendor name, vendor address, total gross amount, and issue date.",
                "images": [b64_image]
            }
        ],
        format=schema.model_json_schema(), 
        options={
            "temperature": 0.0, 
            "num_ctx": 8192
        }
    )


def main():
    print(f"Loading DocILE dataset from {DOCILE_DATASET_PATH}...")
    dataset = Dataset("val", DOCILE_DATASET_PATH)
    
    with open(SUBSET_JSON_PATH, "r") as f:
        subset_ids = json.load(f)
        
    print(f"Loaded {len(subset_ids)} target documents.")
    
    # Dynamically select schema based on the config toggle
    ActiveSchema = InvoiceExtractionWithReasoning if ENABLE_THINKING else InvoiceExtractionBase
    print(f"Thinking Mode: {'ON' if ENABLE_THINKING else 'OFF'}")
    print(f"Thermal Cooldown: {THERMAL_COOLDOWN_SECONDS}s per batch of {THERMAL_COOLDOWN_BATCH_SIZE} documents\n")
    
    results_log = []
    
    # Running a batch of 5 to test the pipeline and cooldown logic
    test_subset = subset_ids[:50]
    # test_subset = subset_ids
    
    # Initialize the executor OUTSIDE the loop
    # We keep a pool of workers available so we don't block on cleanup
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    
    for idx, doc_id in enumerate(tqdm(test_subset, desc="Processing Invoices")):
        doc = dataset[doc_id]
        ground_truth = extract_ground_truth(doc)
        
        pdf_path = os.path.join(DOCILE_DATASET_PATH, "pdfs", f"{doc_id}.pdf")
        b64_image = pdf_to_base64_image(pdf_path)
        
        system_prompt = "You are a precise data extraction AI. Extract the requested fields from this document image."
        
        try:
            # Submit to our persistent thread pool
            future = executor.submit(call_ollama, MODEL_NAME, system_prompt, b64_image, ActiveSchema)
            
            # The strict 120-second watchdog
            response = future.result(timeout=MODEL_TIMEOUT_SECONDS)
            predicted_data = json.loads(response['message']['content'])
            
        except concurrent.futures.TimeoutError:
            print(f"\n[TIMEOUT] Document {doc_id} permanently stalled. Abandoning thread.")
            predicted_data = {"error": f"Timeout - VLM locked up within {MODEL_TIMEOUT_SECONDS} seconds"}
            # Notice we do NOT shut down the executor here. We just let the dead thread hang 
            # in the background and use one of our other max_workers for the next document.
            
        except Exception as e:
            print(f"\n[ERROR] Processing {doc_id}: {e}")
            predicted_data = {"error": str(e)}

        results_log.append({
            "doc_id": doc_id,
            "ground_truth": ground_truth,
            "predicted": predicted_data
        })
        
        # Apply thermal cooldown, skipping the sleep after the final document
        if (idx + 1) % THERMAL_COOLDOWN_BATCH_SIZE == 0 and idx < len(test_subset) - 1:
            print(f"\n[Thermal Control] Cooling down for {THERMAL_COOLDOWN_SECONDS}s...")
            time.sleep(THERMAL_COOLDOWN_SECONDS)

    with open(EVAL_RESULTS_PATH, "w") as f:
        json.dump(results_log, f, indent=4)
        
    print(f"\n✅ Evaluation complete. Results saved to {EVAL_RESULTS_PATH}")

if __name__ == "__main__":
    main()