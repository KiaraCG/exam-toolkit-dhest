import fitz
import base64
from typing import Any

def render_pdf_to_image_parts(pdf_bytes: bytes, label: str = "PDF") -> list[dict[str, Any]]:
    """Render each page of a PDF into a base64 JPEG image part for vision models."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    image_parts = []
    for i, page in enumerate(doc):
        # Render page to a high-res pixmap (150 DPI is ideal for text legibility)
        pix = page.get_pixmap(dpi=150)
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