"""
LexiScan Auto — FastAPI REST API
==================================
Production API for the automated extraction of legal entities from PDFs.
"""

import os
import shutil
import uuid
from typing import Dict, List

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils.logger import configure_logger
from ocr.ocr_engine import extract_text_from_pdf, clean_ocr_text, evaluate_text_quality
from ner.inference import NERInference

logger = configure_logger("LexiScanAuto.API")

app = FastAPI(
    title="LexiScan Auto API",
    description="Automated legal entity extractor from unstructured PDF contracts.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global ML Engines ──────────────────────────────────────────────────────

ner_engine = None

@app.on_event("startup")
def load_models():
    """Load ML engines into memory on startup."""
    global ner_engine
    try:
        logger.info("Initializing NER components...")
        ner_engine = NERInference()
        logger.info("Successfully loaded ML engines.")
    except Exception as exc:
        logger.error(f"Failed to load ML engines on startup: {exc}")
        logger.warning("API will load without an active NER model.")

# ── Response Schema ───────────────────────────────────────────────────────

class ExtractionResponse(BaseModel):
    document_id: str
    filename: str
    metrics: Dict[str, float]
    entities: Dict[str, List[str]]

# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "LexiScan Auto API",
        "ner_model_loaded": ner_engine is not None
    }


@app.post("/extract", response_model=ExtractionResponse)
async def extract_document(file: UploadFile = File(...)):
    """Process a PDF contract pipeline: OCR → NER → Rules → JSON."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF files are supported."
        )

    if not ner_engine:
        raise HTTPException(
            status_code=503,
            detail="NER model not loaded. Please train the model and restart the server."
        )

    doc_id = str(uuid.uuid4())
    temp_dir = os.path.join("data", "raw")
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_path = os.path.join(temp_dir, f"temp_{doc_id}.pdf")

    try:
        # 1. Save uploaded file to disk explicitly
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Processing uploaded document: {file.filename} (ID: {doc_id})")

        # 2. OCR Pipeline
        logger.info("Running OCR pipeline...")
        raw_text = extract_text_from_pdf(temp_path, dpi=300)
        clean_text = clean_ocr_text(raw_text)
        metrics = evaluate_text_quality(clean_text)

        if metrics["noise_ratio"] > 0.5:
            logger.warning(
                f"High OCR noise ratio ({metrics['noise_ratio']}) detected "
                f"for Document ID {doc_id}."
            )

        # 3. NER + Rule-based validation + Grouping
        logger.info("Running NER inference and validation rules...")
        structured_entities = ner_engine.extract_grouped(clean_text)

        logger.info(f"Successfully processed {file.filename}.")
        
        return ExtractionResponse(
            document_id=doc_id,
            filename=file.filename,
            metrics=metrics,
            entities=structured_entities
        )

    except Exception as exc:
        logger.error(f"Error processing document {file.filename}: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Error executing extraction pipeline: {str(exc)}"
        )

    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    logger.info("Starting LexiScan Auto API Server on port 8000...")
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
