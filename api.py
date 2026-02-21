import os
import shutil
import json
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
   
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are supported.")
        
    if not ocr_processor or not ner_engine:
        raise HTTPException(status_code=503, detail="Machine Learning engines are not ready.")
        
    doc_id = str(uuid.uuid4())
    os.makedirs("data/raw", exist_ok=True)
    temp_path = os.path.join("data", "raw", f"temp_{doc_id}.pdf")
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info(f"Processing uploaded document: {file.filename} (ID: {doc_id})")
        
        clean_text, metrics = ocr_processor.process_pdf(temp_path)
        
        if metrics["noise_ratio"] > 0.5:
             logger.warning(f"High noise ratio detected ({metrics['noise_ratio']}) in Document ID {doc_id}.")
        
        entities = ner_engine.extract_entities(clean_text)
        
        logger.info(f"Successfully processed {file.filename}. Found {len(entities)} entities.")
        
        # 3. Auto-save output to data/annotations so the user can review it in Doccano!
        os.makedirs("data/annotations", exist_ok=True)
        annotation_path = os.path.join("data", "annotations", f"{file.filename}.jsonl")
        doccano_labels = [[ent["start_char"], ent["end_char"], ent["entity"]] for ent in entities]
        
        with open(annotation_path, "a", encoding="utf-8") as f:
            record = {
                "document_id": doc_id,
                "text": clean_text,
                "label": doccano_labels,
                "ocr_noise_ratio": metrics["noise_ratio"],
                "text_length": metrics["text_length"]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
        logger.info(f"Saved pre-annotated output to {annotation_path}")
        
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
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    logger.info("Starting LexiScan Auto API Server on port 8000...")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
