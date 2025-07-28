import os
import pdfplumber
import logging
from chunker import split_text_to_chunks
from utils import clean_text, extract_title

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS_PATH = os.path.join(BASE_DIR, "documents")

# print(BASE_DIR,"base directory")
# print(DOCS_PATH,"DOCS PATH")

def process_pdf(doc, base_dir="documents"):
    path = os.path.join(DOCS_PATH, doc["filename"])
    # print(path,"path is found ")
    if not os.path.isfile(path):
        logging.warning(f"File '{path}' not found.")
        return []

    chunks = []
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if not text:
                    continue
                for chunk in split_text_to_chunks(text):
                    chunks.append({
                        "document": doc["filename"],
                        "page_number": i + 1,
                        "refined_text": chunk,
                        "section_title": extract_title(chunk)
                    })
    except Exception as e:
        logging.error(f"Error processing {doc['filename']}: {e}")
    return chunks
