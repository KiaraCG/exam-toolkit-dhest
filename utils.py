import fitz
import base64
from typing import Any

def extract_pdf_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = [f"--- page {i+1} ---\n{p.get_text()}" for i, p in enumerate(doc)]
    text = "\n".join(pages)
    if len(text.strip()) < 200 * len(doc):   # likely scanned, no text layer
        raise ValueError("Little or no extractable text; use image mode for this PDF")
    return text
    
def render_pdf_to_image_parts(pdf_bytes: bytes, label: str = "PDF") -> list[dict[str, Any]]:
    """Render each page of a PDF into a base64 JPEG image part for vision models."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    image_parts = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=96)
        img_data = pix.tobytes("jpeg")
        b64_encoded = base64.b64encode(img_data).decode("ascii")
        image_parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64_encoded}"
            }
        })
    print(f"  Rendered {len(image_parts)} pages for {label} into images", flush=True)
    return image_parts