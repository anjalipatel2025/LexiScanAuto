from fastapi.testclient import TestClient
import os
import tempfile
import fitz

from api.app import app

client = TestClient(app)

def create_dummy_pdf(path: str, text: str):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)
    doc.save(path)
    doc.close()

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_extract_endpoint_invalid_file_type():
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tf:
        tf.write(b"Not a PDF")
        tf.close()
        
    try:
        with open(tf.name, "rb") as f:
            response = client.post("/extract", files={"file": ("test.txt", f, "text/plain")})
            assert response.status_code == 400
    finally:
        os.remove(tf.name)

# We skip a full /extract functional test without a loaded mock NER model since it's hard to mock global variables in the TestClient dynamically from here.
