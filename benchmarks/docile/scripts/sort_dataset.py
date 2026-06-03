import json
import fitz  # PyMuPDF
import os
from tqdm import tqdm
from config import DOCILE_DATASET_PATH, SUBSET_JSON_PATH, DATA_DIR

def main():
    sorted_output_path = DATA_DIR / "eval_subset_ids_sorted.json"

    print(f"Loading {SUBSET_JSON_PATH}...")
    with open(SUBSET_JSON_PATH, "r") as f:
        doc_ids = json.load(f)

    doc_sizes = []
    
    print("Calculating native PDF dimensions...")
    for doc_id in tqdm(doc_ids, desc="Scanning PDFs"):
        pdf_path = os.path.join(DOCILE_DATASET_PATH, "pdfs", f"{doc_id}.pdf")
        try:
            doc = fitz.open(pdf_path)
            page = doc.load_page(0)
            
            # Calculate total area (width * height)
            area = page.rect.width * page.rect.height
            doc_sizes.append((doc_id, area))
            doc.close()
        except Exception as e:
            print(f"Error reading {doc_id}: {e}")
            # Push corrupted files to the absolute end of the list
            doc_sizes.append((doc_id, float('inf'))) 

    # Sort the list by area, ascending (smallest first)
    doc_sizes.sort(key=lambda x: x[1])
    sorted_ids = [item[0] for item in doc_sizes]

    # Save the new sorted list
    with open(sorted_output_path, "w") as f:
        json.dump(sorted_ids, f, indent=4)
        
    print(f"\n✅ Saved {len(sorted_ids)} sorted IDs to {sorted_output_path}")
    print(f"Smallest doc area: {doc_sizes[0][1]:.2f} sq points")
    print(f"Largest doc area:  {doc_sizes[-2][1]:.2f} sq points (excluding errors)")

if __name__ == "__main__":
    main()