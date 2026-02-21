import os
import shutil
import uuid
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
from pydantic import BaseModel

from ocr.ocr_engine import OCRProcessor
from ner.inference import NERInference
from utils.logger import configure_logger

logger = configure_logger("LexiScanAuto.API")

app = FastAPI(
    title="LexiScan Auto API",
    description="Production API for FinTech Legal Contract OCR and NER Extraction",
    version="1.0.0"
)

# Initialize engines on startup
try:
    logger.info("Initializing OCR components...")
    ocr_processor = OCRProcessor()
    
    logger.info("Initializing NER components...")
    ner_engine = NERInference()
    logger.info("Successfully loaded all LexiScan ML engines into memory.")
except Exception as e:
    logger.error(f"Failed to load ML engines on startup: {e}")
    ner_engine = None
    ocr_processor = None

class EntityResponse(BaseModel):
    entity: str
    value: str
    start_char: int
    end_char: int

class ExtractionResponse(BaseModel):
    document_id: str
    filename: str
    ocr_noise_ratio: float
    text_length: int
    full_text: str
    entities: List[EntityResponse]

@app.get("/")
def health_check():
    return {"status": "ok", "service": "LexiScan Auto API", "version": "1.0.0"}

@app.post("/api/v1/extract", response_model=ExtractionResponse)
async def extract_document(file: UploadFile = File(...)):
    """
    Production endpoint to upload a FinTech PDF contract, extract clean text via PyMuPDF OCR, 
    and identify structured financial Named Entities using SpaCy.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are supported.")
        
    if not ocr_processor or not ner_engine:
        raise HTTPException(status_code=503, detail="Machine Learning engines are not ready.")
        
    doc_id = str(uuid.uuid4())
    os.makedirs("data/raw", exist_ok=True)
    temp_path = os.path.join("data", "raw", f"temp_{doc_id}.pdf")
    
    try:
        # Write streaming upload to disk temporarily for the OCR engine
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info(f"Processing uploaded document: {file.filename} (ID: {doc_id})")
        
        # 1. OCR Extraction pipeline
        clean_text, metrics = ocr_processor.process_pdf(temp_path)
        
        if metrics["noise_ratio"] > 0.5:
             logger.warning(f"High noise ratio detected ({metrics['noise_ratio']}) in Document ID {doc_id}.")
        
        # 2. NER Extraction pipeline
        entities = ner_engine.extract_entities(clean_text)
        
        logger.info(f"Successfully processed {file.filename}. Found {len(entities)} entities.")
        
        return ExtractionResponse(
            document_id=doc_id,
            filename=file.filename,
            ocr_noise_ratio=metrics["noise_ratio"],
            text_length=metrics["text_length"],
            full_text=clean_text,
            entities=entities
        )
        
    except Exception as e:
        logger.error(f"Error processing document {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error executing pipeline: {str(e)}")
        
    finally:
        # Safely cleanup the temp file to keep production storage clean
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    logger.info("Starting LexiScan Auto API Server on port 8000...")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
