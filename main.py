import os
import json
import uuid
from datetime import datetime, timezone
import argparse

from ocr.ocr_engine import OCRProcessor
from ner.inference import NERInference
from utils.logger import configure_logger

logger = configure_logger("LexiScanAuto.Main")

def run_prediction(pdf_path: str):
    logger.info("=== Starting LexiScan Auto CLI ===")
    
    if not os.path.exists(pdf_path):
        logger.error(f"File not found: {pdf_path}")
        return

    try:
        # OCR
        logger.info("Extracting text via OCR...")
        processor = OCRProcessor(dpi=300)
        clean_text, metrics = processor.process_pdf(pdf_path)
        
        # NER + Rules
        logger.info("Extracting entities...")
        inference = NERInference()
        grouped_entities = inference.extract_grouped(clean_text)
        
        # Output
        output_record = {
            "document_id": str(uuid.uuid4()),
            "document_name": os.path.basename(pdf_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "entities": grouped_entities
        }
        
        print("\n" + "="*50)
        print("EXTRACTION RESULTS")
        print("="*50)
        print(json.dumps(output_record, indent=2))
        print("="*50 + "\n")

    except Exception as e:
        logger.error(f"Execution failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LexiScan Auto CLI Extraction")
    parser.add_argument("--pdf", type=str, help="Path to the PDF file", required=True)
    args = parser.parse_args()
    
    run_prediction(args.pdf)