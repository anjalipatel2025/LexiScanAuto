import re
import logging
from typing import Dict, Tuple

import fitz  # PyMuPDF
from utils.logger import configure_logger

class OCRProcessor:
    """Class to process PDF files, extract text using PyMuPDF, and calculate quality metrics."""
    
    def __init__(self, dpi: int = 300):
        self.dpi = dpi  # Not strictly needed for text extraction, but keeping signature for backwards compatibility
        self.logger = configure_logger("LexiScanAuto.OCRProcessor")

    def _clean_text(self, text: str) -> str:
        """
        Cleans extracted text while preserving legal punctuation.
        Removes excessive whitespaces but keeps structure.
        """
        # Replace multiple spaces with a single space
        text = re.sub(r'[ \t]+', ' ', text)
        # Normalize line breaks (replace multiple newlines with a double newline for paragraphs)
        text = re.sub(r'\n+', '\n', text)
        # Strip leading/trailing whitespace
        return text.strip()

    def _calculate_metrics(self, text: str) -> Dict[str, float]:
        """
        Calculates basic text quality metrics.
        - text_length: Total number of characters.
        - noise_ratio: Ratio of non-alphanumeric characters to total characters.
        """
        if not text:
            return {"text_length": 0, "noise_ratio": 1.0}
            
        text_length = len(text)
        alphanumeric_count = sum(1 for char in text if char.isalnum() or char.isspace())
        noise_characters = text_length - alphanumeric_count
        noise_ratio = noise_characters / text_length if text_length > 0 else 1.0

        return {
            "text_length": text_length,
            "noise_ratio": round(noise_ratio, 4)
        }

    def process_pdf(self, pdf_path: str) -> Tuple[str, dict]:
        """
        Main pipeline to open PDF, extract text, and return cleaned text with metrics.
        """
        self.logger.info(f"Starting text extraction for: {pdf_path}")
        try:
            doc = fitz.open(pdf_path)
            raw_text = ""
            
            for i in range(len(doc)):
                self.logger.info(f"Processing page {i + 1}/{len(doc)}...")
                page = doc.load_page(i)
                text = page.get_text("text")
                raw_text += text + "\n"
            
            doc.close()
                
            self.logger.info("Extraction completed. Cleaning text...")
            cleaned_text = self._clean_text(raw_text)
            metrics = self._calculate_metrics(cleaned_text)
            
            self.logger.info(f"Processing complete. Text length: {metrics['text_length']}, Noise ratio: {metrics['noise_ratio']}")
            return cleaned_text, metrics
            
        except Exception as e:
            self.logger.error(f"Text extraction failed for {pdf_path}. Error: {e}")
            raise