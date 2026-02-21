FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# Include poppler-utils if you ever revert to pdf2image
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose production port
EXPOSE 8000

# Run the FastAPI server in production mode
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
