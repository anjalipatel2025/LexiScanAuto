FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (Tesseract OCR, Poppler for pdf2image, build tools)
RUN apt-get update && apt-get install -y \
    build-essential \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download a small SpaCy english model to be used as a base if necessary
RUN python -m spacy download en_core_web_sm

# Copy the entire project code into the container
COPY . .

# Expose FastAPI production port
EXPOSE 8000

# Set Python path to ensure imports work correctly
ENV PYTHONPATH="/app"

# Start the uvicorn server serving the main FastAPI application
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
