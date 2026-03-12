import os
import tempfile
import fitz
from ocr.ocr_engine import extract_text_from_pdf, clean_ocr_text, evaluate_text_quality

def create_dummy_pdf(path: str, text: str):
    """Creates a basic text PDF for testing OCR extraction."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)
    doc.save(path)
    doc.close()

def test_extract_text_from_pdf():
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
        temp_path = tf.name

    try:
        test_string = "Confidentiality Agreement"
        create_dummy_pdf(temp_path, test_string)
        
        text = extract_text_from_pdf(temp_path)
        assert test_string in text
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_clean_ocr_text():
    dirty_text = "  This   is    dirty \n\n\ntest  \t data.   "
    clean_text = clean_ocr_text(dirty_text)
    assert clean_text == "This is dirty\ntest data."

def test_evaluate_text_quality():
    text = "Valid document with some #&* noise."
    metrics = evaluate_text_quality(text)
    
    assert "noise_ratio" in metrics
    assert "text_length" in metrics
    assert metrics["text_length"] > 0
    assert metrics["noise_ratio"] > 0.0 # #&* added noise
