import fitz
import base64
from typing import Any

def render_pdf_to_image_parts(pdf_bytes: bytes, label: str = "PDF", dpi: int = 150) -> list[dict[str, Any]]:
    """Render each page of a PDF into a base64 JPEG image part for vision models."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    image_parts = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        b64 = base64.b64encode(pix.tobytes("jpeg")).decode("ascii")
        image_parts.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    print(f"  Rendered {len(image_parts)} pages for {label} into images (dpi={dpi})", flush=True)
    return image_parts


def extract_pdf_text(pdf_bytes: bytes, label: str = "PDF") -> str:
    """Extract the text layer of a PDF, page by page. Raises ValueError if it looks scanned."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = [f"--- {label} page {i + 1} ---\n{p.get_text()}" for i, p in enumerate(doc)]
    text = "\n".join(pages)
    if len(text.strip()) < 200 * max(len(doc), 1):
        raise ValueError(f"{label} has little or no extractable text (scanned?); use images mode")
    print(f"  Extracted {len(text)} characters of text from {len(doc)} pages for {label}", flush=True)
    return text