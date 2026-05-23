import fitz  # PyMuPDF
import base64
import io
import ollama
from PIL import Image
from config import VISION_MODEL, TEXT_MODEL

def pdf_to_base64_images(pdf_path: str, max_dimension: int = 1024) -> list:
    """Converts PDF pages to JPEGs, padded to a perfect square to prevent GPU tensor crashes.
        This was done to fix the GGML_ASSERT error: GGML_ASSERT(a->ne[2] * 4 == b->ne[0])
    """
    doc = fitz.open(pdf_path)
    images = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num) 
        
        # 1. Render the page to a high-res pixmap
        pix = page.get_pixmap(dpi=150)
        
        # 2. Convert PyMuPDF pixmap to a Pillow Image
        # If the PDF has an alpha channel, we convert it to standard RGB
        mode = "RGBA" if pix.alpha else "RGB"
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        if mode == "RGBA":
            # Strip alpha channel and replace with white background
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
            
        # 3. Scale the image down so its longest edge matches max_dimension
        img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        
        # 4. Create a perfect white square canvas
        square_canvas = Image.new('RGB', (max_dimension, max_dimension), (255, 255, 255))
        
        # 5. Paste the scaled invoice into the exact center of the square
        offset_x = (max_dimension - img.width) // 2
        offset_y = (max_dimension - img.height) // 2
        square_canvas.paste(img, (offset_x, offset_y))
        
        # 6. Export to base64
        buffer = io.BytesIO()
        square_canvas.save(buffer, format="JPEG", quality=85)
        images.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))
        
    doc.close()
    return images

def run_2_step_extraction(images, schema):
    """
    Step 1: GLM-OCR transcribes pages to Markdown.
    Step 2: Llama 3.1 extracts structured JSON from the Markdown.
    """
    combined_markdown = ""
    
    # STEP 1: The Eyes (Vision)
    for idx, b64_img in enumerate(images):
        vision_response = ollama.chat(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": "Text Recognition:",
                "images": [b64_img]
            }],
            options={"temperature": 0.0}
        )
        combined_markdown += f"\n--- PAGE {idx + 1} ---\n"
        combined_markdown += vision_response['message']['content']
        
    # STEP 2: The Brain (LLM)
    system_prompt = (
        "You are an expert data extraction AI. You will be provided with the raw OCR markdown "
        "text extracted from an invoice. Analyze the text and extract the required fields into the JSON schema."
    )
    
    llm_response = ollama.chat(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"OCR DOCUMENT TEXT:\n{combined_markdown}"}
        ],
        format=schema.model_json_schema(),
        options={
            "temperature": 0.0,
            "num_ctx": 8192
        }
    )
    
    return llm_response