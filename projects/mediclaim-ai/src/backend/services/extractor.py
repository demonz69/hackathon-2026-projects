"""
MediClaim AI — PDF Text Extraction
PyMuPDF for native PDFs, Tesseract OCR fallback for scanned documents.
"""

import fitz  # PyMuPDF
import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Try importing Tesseract — optional dependency
try:
    import pytesseract
    from PIL import Image
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    logger.warning("pytesseract not available — OCR fallback disabled")


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract text from a PDF file.
    1. Try PyMuPDF native text extraction first.
    2. If that yields very little text, fall back to Tesseract OCR.
    """
    text = _extract_with_pymupdf(pdf_bytes)

    # If native extraction got very little text, try OCR
    if len(text.strip()) < 50 and HAS_TESSERACT:
        logger.info("Native extraction yielded little text, trying OCR...")
        ocr_text = _extract_with_ocr(pdf_bytes)
        if len(ocr_text.strip()) > len(text.strip()):
            text = ocr_text

    return text


def _extract_with_pymupdf(pdf_bytes: bytes) -> str:
    """Extract text using PyMuPDF's native text extraction."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text("text")
        if page_text.strip():
            pages.append(f"--- Page {page_num + 1} ---\n{page_text}")
    doc.close()
    return "\n\n".join(pages)


def _extract_with_ocr(pdf_bytes: bytes) -> str:
    """Extract text by rendering PDF pages to images and running Tesseract OCR."""
    if not HAS_TESSERACT:
        return ""
        
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render at 300 DPI for good OCR accuracy
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        page_text = pytesseract.image_to_string(img)
        if page_text.strip():
            pages.append(f"--- Page {page_num + 1} (OCR) ---\n{page_text}")
    doc.close()
    return "\n\n".join(pages)
