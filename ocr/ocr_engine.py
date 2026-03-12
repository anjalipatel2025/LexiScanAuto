"""
LexiScan Auto — OCR Pipeline Engine
=====================================
Converts PDF documents (including scanned / image-based PDFs) into clean
text using a two-tier strategy:

1. **Native text extraction** via PyMuPDF (``fitz``) — fast, lossless for
   digitally-born PDFs.
2. **OCR fallback** via ``pdf2image`` + ``pytesseract`` — handles scanned
   documents where embedded text is absent.

The engine also provides text-cleaning and quality-evaluation utilities that
feed directly into the NER training pipeline.
"""

import os
import re
import string
from typing import Dict, List, Tuple

import fitz  # PyMuPDF

from utils.logger import configure_logger

logger = configure_logger("LexiScanAuto.OCR")

# ---------------------------------------------------------------------------
# Lazy imports — Tesseract + pdf2image are optional on dev machines but
# required in the Docker container.  We import them lazily so the module
# loads even when they are absent (tests, annotation review, etc.).
# ---------------------------------------------------------------------------
_TESSERACT_AVAILABLE = False
_PDF2IMAGE_AVAILABLE = False

try:
    import pytesseract
    _TESSERACT_AVAILABLE = True
except ImportError:
    pass

try:
    from pdf2image import convert_from_path
    _PDF2IMAGE_AVAILABLE = True
except ImportError:
    pass


# ───────────────────────────────────────────────────────────────────────────
#  Public helper functions (Week 1 deliverables)
# ───────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(
    pdf_path: str,
    dpi: int = 300,
    force_ocr: bool = False,
) -> str:
    """Extract text from a PDF file.

    Parameters
    ----------
    pdf_path : str
        Absolute or relative path to a ``.pdf`` file.
    dpi : int
        Resolution for rasterising pages before OCR (default 300).
    force_ocr : bool
        If *True*, always use Tesseract even when embedded text exists.

    Returns
    -------
    str
        Raw concatenated text from all pages.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    raw_pages: List[str] = []

    for page_idx in range(len(doc)):
        page = doc.load_page(page_idx)
        page_text = page.get_text("text") if not force_ocr else ""

        # If page yielded < 30 characters of text, treat as scanned
        if len(page_text.strip()) < 30 or force_ocr:
            page_text = _ocr_page(page, dpi) or page_text

        raw_pages.append(page_text)
        logger.debug(f"Page {page_idx + 1}/{len(doc)}: {len(page_text)} chars")

    doc.close()
    return "\n".join(raw_pages)


def clean_ocr_text(text: str) -> str:
    """Normalise and clean raw OCR / extracted text.

    * Collapses repeated whitespace.
    * Removes control characters.
    * Normalises line endings.
    * Strips leading / trailing whitespace.
    """
    # Remove non-printable control characters (keep newlines, tabs)
    text = re.sub(r"[^\S \n\t]+", "", text)
    # Collapse horizontal whitespace
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse vertical whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove lines that are purely whitespace
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return text.strip()


def evaluate_text_quality(text: str) -> Dict[str, float]:
    """Compute simple quality metrics for the extracted text.

    Returns
    -------
    dict
        ``text_length``  — character count.
        ``word_count``    — naive whitespace-split word count.
        ``noise_ratio``   — fraction of non-alphanumeric, non-space chars.
        ``alpha_ratio``   — fraction of alphabetic characters.
    """
    if not text:
        return {
            "text_length": 0,
            "word_count": 0,
            "noise_ratio": 1.0,
            "alpha_ratio": 0.0,
        }

    text_length = len(text)
    alpha_count = sum(1 for ch in text if ch.isalpha())
    alnum_or_space = sum(1 for ch in text if ch.isalnum() or ch.isspace())
    noise_chars = text_length - alnum_or_space

    return {
        "text_length": text_length,
        "word_count": len(text.split()),
        "noise_ratio": round(noise_chars / text_length, 4),
        "alpha_ratio": round(alpha_count / text_length, 4),
    }


# ───────────────────────────────────────────────────────────────────────────
#  OCRProcessor class (backward-compatible with existing code)
# ───────────────────────────────────────────────────────────────────────────

class OCRProcessor:
    """High-level class wrapping the three public helper functions."""

    def __init__(self, dpi: int = 300, force_ocr: bool = False):
        self.dpi = dpi
        self.force_ocr = force_ocr
        self.logger = configure_logger("LexiScanAuto.OCRProcessor")

    def process_pdf(self, pdf_path: str) -> Tuple[str, dict]:
        """Run the full OCR pipeline on *pdf_path*.

        Returns
        -------
        tuple[str, dict]
            ``(clean_text, quality_metrics)``
        """
        self.logger.info(f"Starting text extraction for: {pdf_path}")

        raw_text = extract_text_from_pdf(
            pdf_path, dpi=self.dpi, force_ocr=self.force_ocr,
        )
        self.logger.info("Extraction completed. Cleaning text...")

        cleaned = clean_ocr_text(raw_text)
        metrics = evaluate_text_quality(cleaned)

        self.logger.info(
            f"Processing complete. Text length: {metrics['text_length']}, "
            f"Noise ratio: {metrics['noise_ratio']}"
        )
        return cleaned, metrics


# ───────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ───────────────────────────────────────────────────────────────────────────

def _ocr_page(page: "fitz.Page", dpi: int = 300) -> str:
    """Rasterise a single ``fitz.Page`` and run Tesseract on it."""
    if not (_TESSERACT_AVAILABLE and _PDF2IMAGE_AVAILABLE):
        logger.warning(
            "Tesseract / pdf2image not installed — cannot OCR scanned page."
        )
        return ""

    try:
        # Render page to a pixmap, save as temp PNG, then OCR
        pix = page.get_pixmap(dpi=dpi)
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img)
        return text
    except Exception as exc:
        logger.error(f"OCR failed for page: {exc}")
        return ""


# ───────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ───────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python -m ocr.ocr_engine <path_to_pdf>")
        sys.exit(1)

    pdf = sys.argv[1]
    processor = OCRProcessor()
    text, metrics = processor.process_pdf(pdf)
    print(json.dumps({"text": text[:500], **metrics}, indent=2))