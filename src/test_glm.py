import ollama
import base64
import fitz

SAMPLE_PDF_PATH = "/Users/sanja/Projects/docile/data/docile/pdfs/0a1bf0ac3db840e1be10e064.pdf"

# def image_to_base64(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode("utf-8")

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

image_b64 = pdf_to_base64_image(SAMPLE_PDF_PATH)

print("Running GLM-OCR (0.9B)...")
response = ollama.chat(
    model='glm-ocr',
    messages=[{
        'role': 'user',
        # GLM-OCR requires very specific, rigid trigger prompts. 
        # "Text Recognition:" or "Table Recognition:" are the standards.
        'content': 'Table Recognition:', 
        'images': [image_b64]
    }]
)

print("\n--- RAW GLM-OCR MARKDOWN OUTPUT ---")
print(response['message']['content'])