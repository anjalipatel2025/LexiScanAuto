from fastapi import FastAPI, UploadFile, File
import shutil
import os

# Importing previous weeks
from week1_ocr.ocr_pipeline import extract_text
from week2_ner.ner_pipeline import extract_entities
from week3_rules.rule_engine import apply_rules

# Create FastAPI app
app = FastAPI(title="LexiScanAuto API")

# Test route (very important to check API works)
@app.get("/")
def home():
    return {"message": "LexiScanAuto API is running successfully 🚀"}

# Main processing route
@app.post("/process/")
async def process_pdf(file: UploadFile = File(...)):

    # Save uploaded file temporarily
    file_location = f"temp_{file.filename}"

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Week 1 - OCR
        text = extract_text(file_location)

        # Week 2 - NER
        entities = extract_entities(text)

        # Week 3 - Rule Engine
        final_output = apply_rules(entities)

        response = {
            "status": "success",
            "data": final_output
        }

    except Exception as e:
        response = {
            "status": "error",
            "message": str(e)
        }

    finally:
        # Delete temp file
        if os.path.exists(file_location):
            os.remove(file_location)

    return response
