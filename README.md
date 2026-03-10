# 📄 LexiScanAuto

### Automated Document Scanning & Information Extraction System

LexiScanAuto is an **AI-powered document processing system** that automatically extracts structured information from scanned documents and images.
The system integrates **Optical Character Recognition (OCR)** and **Named Entity Recognition (NER)** to convert unstructured text from documents into meaningful, machine-readable data.

This project demonstrates how **computer vision, natural language processing, and API-based architecture** can be combined to build scalable document automation solutions.

The goal of LexiScanAuto is to **reduce manual document processing efforts and improve data extraction accuracy** in real-world applications such as invoices, forms, and identity documents.

---

# 📌 Project Overview

In many industries, large volumes of documents need to be processed manually. LexiScanAuto automates this workflow by:

1. Extracting text from scanned documents using **OCR**.
2. Processing the extracted text using **Natural Language Processing (NLP)**.
3. Identifying important entities such as names, dates, numbers, and identifiers using **NER models**.
4. Returning structured results through a **REST API**.

This makes the system suitable for applications such as:

* Document digitization
* Automated data entry
* Intelligent document processing
* AI-powered document workflows

---

# 🛠 Tech Stack

## Backend

* Python
* FastAPI / Flask API
* Uvicorn Server

## AI / Machine Learning

* OCR (Tesseract / OCR engines)
* Named Entity Recognition (NER)
* Pre-trained NLP models

## Data Processing

* Python utilities
* Text preprocessing pipelines

## Deployment

* Docker
* REST API architecture

---

# 🚀 Getting Started

Follow these instructions to run the project locally.

---

# ⚙ Prerequisites

Make sure the following are installed:

* Python 3.8+
* pip (Python package manager)
* Git
* Docker (optional for container deployment)

Check Python version:

```bash
python --version
```

---

# 📥 Installation

Clone the repository:

```bash
git clone https://github.com/sayalivadya1654-cell/LexiScanAuto.git
```

Move into the project directory:

```bash
cd LexiScanAuto
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

# ⚙ Configuration

Before running the application ensure:

* OCR models are available inside the **ocr/** directory.
* Named Entity Recognition models are stored inside **ner/**.
* Pre-trained model files are placed inside the **models/** directory.

Configuration parameters may be adjusted in:

```
api.py
```

or other configuration modules depending on deployment needs.

---

# ▶ Running the Application

Run the server using:

```bash
python main.py
```

or

```bash
python api.py
```

For Windows users you can also use:

```bash
start_server.bat
```

Once the server starts, the API will be available at:

```
http://localhost:8000
```

You can test endpoints using:

* Postman
* cURL
* Browser API requests

---

# ✨ Features

* Automated **document text extraction**
* AI-based **Named Entity Recognition**
* OCR-based **image-to-text conversion**
* REST API for easy integration
* Modular architecture for scalable AI systems
* Docker support for deployment
* Structured JSON output for extracted data

---

# 📂 Project Structure

```
LexiScanAuto
│
├── models/                # Pre-trained AI/ML models
│
├── ner/                   # Named Entity Recognition modules
│
├── ocr/                   # OCR processing pipeline
│
├── utils/                 # Utility functions and helper scripts
│
├── api.py                 # API endpoints for document processing
├── main.py                # Application entry point
│
├── requirements.txt       # Python dependencies
├── dockerfile             # Docker configuration for deployment
├── start_server.bat       # Script to start the server on Windows
│
└── README.md              # Project documentation
```

---

# 🔍 Document Processing Pipeline

The system processes documents through the following pipeline:

1️⃣ **Document Input**
User uploads a document or image.

2️⃣ **OCR Processing**
The OCR engine extracts raw text from the document.

3️⃣ **Text Preprocessing**
Noise removal, tokenization, and formatting.

4️⃣ **Named Entity Recognition**
The NER model detects entities such as:

* Names
* Dates
* Organizations
* Identification numbers
* Locations

5️⃣ **Structured Output**
The system returns extracted information in **JSON format**.

---

# 🤝 Contributing

Contributions are welcome!

If you want to improve the project:

1. Fork the repository
2. Create a new branch

```
git checkout -b feature-new-feature
```

3. Commit your changes

```
git commit -m "Added new feature"
```

4. Push to your branch

```
git push origin feature-new-feature
```

5. Open a Pull Request.

---

# 📜 License

This project is licensed under the **MIT License**.

---

⭐ If you find this project helpful, consider **starring the repository**.
