# LexiScan Auto

Automated Legal Entity Extractor.

An NLP system that automatically extracts key legal entities (PARTY, DATE, AMOUNT, JURISDICTION) from unstructured PDF contracts.

## Architecture & Tech Stack

*   **Python 3.11** - Core language
*   **FastAPI** - RESTful API serving the extraction model
*   **PyMuPDF / pdf2image & Tesseract** - robust two-tier hybrid text extraction
*   **SpaCy v3** - Custom Named Entity Recognition training and inference
*   **Docker** - Simple and robust deployment

## Quickstart

### 1. Setup

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

*Note on external dependencies: if you are running this natively, ensure that you have installed Tesseract OCR and Poppler on your system and they are appended to your PATH.*

### 2. NER Model Training

Given some Doccano-formatted `jsonl` data sets placed under `data/annotations/`, you can train the custom NER model:

```bash
python -m ner.train
```

Evaluate the model:

```bash
python -m ner.evaluate
```

This will save your custom SpaCy model directly to `models/lexiscan_ner/`.

### 3. Running the REST API

You can start the production API locally with:

```bash
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Then visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to access the interactive Swagger documentation.

## Running Tests
Run the test suite using pytest:
```bash
python -m pytest tests/
```

## Docker Deployment

Build and run the full stack container using Docker. The container perfectly pre-configures Tesseract OCR, Poppler, Python, and runs the application automatically.

```bash
docker build -t lexiscan-auto .
docker run -p 8000:8000 lexiscan-auto
```

## API Usage

**Endpoint:** `POST /extract`

**Using curl:**
```bash
curl -X 'POST' \
  'http://localhost:8000/extract' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/path/to/contract.pdf'
```

**Example API Response:**
```json
{
  "document_id": "99f3ef4d-4560-4b2a-8aa4-5e1c2b3d4e5f",
  "filename": "contract.pdf",
  "metrics": {
    "text_length": 450,
    "word_count": 80,
    "noise_ratio": 0.015,
    "alpha_ratio": 0.85
  },
  "entities": {
    "DATE": [
      "2023-10-12"
    ],
    "PARTY": [
      "Acme Corp",
      "John Doe"
    ],
    "AMOUNT": [
      "50000.00"
    ],
    "JURISDICTION": [
      "New York"
    ]
  }
}
```
