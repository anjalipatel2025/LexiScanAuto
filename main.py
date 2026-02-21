import os
import json
import uuid
from datetime import datetime, timezone

from ocr.ocr_engine import OCRProcessor
from utils.logger import configure_logger

logger = configure_logger("LexiScanAuto.Main")

def save_annotation_output(text: str, metrics: dict, output_path: str, doc_name: str):
    """
    Saves the processed output to a JSONL file format 
    suitable for Doccano/Prodigy.
    """
    output_record = {
        "document_id": str(uuid.uuid4()),
        "document_name": doc_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ocr_noise_ratio": metrics["noise_ratio"],
        "text_length": metrics["text_length"],
        "text": text,
        "label": [] # Empty list for NER annotations
    }
    
    # Write as JSONL (JSON Lines)
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(output_record, ensure_ascii=False) + "\n")
        
    logger.info(f"Successfully saved output to {output_path}")

def run_pipeline(pdf_path: str):
    """
    Main orchestrator for the LexiScan Auto OCR pipeline.
    """
    logger.info("=== Starting LexiScan Auto Pipeline ===")
    
    if not os.path.exists(pdf_path):
        logger.error(f"File not found: {pdf_path}")
        return

    try:
        # Ensure directories exist
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/annotations", exist_ok=True)

        processor = OCRProcessor(dpi=300)
        
        # Extract text and metrics
        clean_text, metrics = processor.process_pdf(pdf_path)
        
        # Save output
        document_name = os.path.basename(pdf_path)
        output_file = os.path.join("data", "annotations", f"{document_name}.jsonl")
        
        save_annotation_output(clean_text, metrics, output_file, document_name)

        logger.info("=== LexiScan Auto Pipeline Completed Successfully ===")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    sample_pdf_path = os.path.join("data", "raw", "DM-1.pdf")
    run_pipeline(sample_pdf_path)