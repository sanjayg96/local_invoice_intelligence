import fitz
import ollama
import pytesseract
import pdfplumber
from PIL import Image
from config import TEXT_MODEL

# --- APPLE SILICON TESSERACT PATH ---
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract' 

def extract_text_hybrid(pdf_path: str) -> str:
    """
    The 'Hybrid Eyes'. Uses pdfplumber for perfect 2D spatial text extraction.
    Falls back to PyMuPDF + Tesseract OCR on a per-page basis if it's a flat scan.
    """
    master_text = ""
    
    # Open with both libraries (pdfplumber for text layout, fitz for fast image rendering)
    doc_plumber = pdfplumber.open(pdf_path)
    doc_fitz = fitz.open(pdf_path)
    
    for page_num in range(len(doc_plumber.pages)):
        page_plumber = doc_plumber.pages[page_num]
        
        # 1. The Fast Path: Born-Digital Layout Extraction
        # This is the command that pads text with spaces to match the physical layout
        text = page_plumber.extract_text(layout=True)
        
        # Clean up in case the page is completely devoid of digital text
        if text:
            text = text.strip()
        else:
            text = ""
        
        # 2. The Validation Gate: Is this a flat scanned image?
        if len(text) < 50: 
            # print(f"  [Fallback] Page {page_num + 1} is a flat scan. Triggering OCR...")
            
            # Use PyMuPDF to quickly render the page to an image
            page_fitz = doc_fitz.load_page(page_num)
            pix = page_fitz.get_pixmap(dpi=300)
            mode = "RGBA" if pix.alpha else "RGB"
            img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            
            if mode == "RGBA":
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
                
            # Run deterministic OCR (Tesseract preserves basic layout by default)
            text = pytesseract.image_to_string(img)
        
        master_text += f"\n--- START OF PAGE {page_num + 1} ---\n"
        master_text += text
        master_text += f"\n--- END OF PAGE {page_num + 1} ---\n"
            
    doc_plumber.close()
    doc_fitz.close()
    
    return master_text

def run_2_step_extraction(pdf_path: str, schema):
    """
    Step 1 (Perception): Hybrid router extracts text via pdfplumber or Tesseract.
    Step 2 (Analysis): Llama 3.1 reads the text and enforces the JSON schema.
    """
    
    # 1. Extract the text securely, regardless of document origin
    master_transcription = extract_text_hybrid(pdf_path)
    # print(pdf_path)
    # print(master_transcription)
    
    # 2. The Auditor (Llama 3.1 8B)
    system_prompt = (
        "You are an expert financial data auditor. You will be provided with a complete, "
        "multi-page text extraction of an invoice. The text has been formatted to preserve "
        "the original physical layout of the document.\n\n"
        "CRITICAL INSTRUCTION FOR `reasoning_process`: This field is your scratchpad. "
        "Review the entire text. Note the vendor. Verify the dates. "
        "If there are multiple pages, find the final total (usually on the last page) "
        "and explicitly differentiate it from line-item subtotals BEFORE populating the JSON fields."
    )
    
    llm_response = ollama.chat(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"DOCUMENT TRANSCRIPTION:\n{master_transcription}"}
        ],
        format=schema.model_json_schema(),
        options={
            "temperature": 0.0,
            "num_ctx": 8192,
            "num_predict": 1024  # Forces the model to stop and return after 1024 tokens, preventing server lockups. 
        }
    )
    
    return llm_response