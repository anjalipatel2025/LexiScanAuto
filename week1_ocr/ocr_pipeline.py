import pytesseract
from pdf2image import convert_from_path

# Agar Windows hai to path set kare:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text(pdf_path):
    pages = convert_from_path(pdf_path)
    full_text = ""

    for page in pages:
        text = pytesseract.image_to_string(page)
        full_text += text + "\n"

    return full_text


if __name__ == "__main__":
    pdf_path = r"c:\Users\win11\Downloads\sample_contract (1).pdf"
    text = extract_text(pdf_path)
    print(text)

