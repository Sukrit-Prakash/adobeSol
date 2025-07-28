import os
import json
import time
import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Set
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256

import pdfplumber
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

# --- Constants and Configuration ---
INPUT_CONFIG_FILE = "input.json"
OUTPUT_FILE = "challenge1b_output2.json"
DOCUMENTS_DIR = "documents"
MODEL_NAME = "paraphrase-MiniLM-L6-v2"
MAX_SEQ_LENGTH = 256
EMBEDDING_BATCH_SIZE = 128
CHUNK_SIZE_SENTENCES = 3
MIN_CHUNK_LENGTH_CHARS = 100
MAX_CHUNK_LENGTH_CHARS = 1500
TOP_N_CANDIDATES = 5
DUPLICATE_CHECK_POOL = 25
HIGHLIGHT_KEYWORDS = ["sign", "form", "create", "tool", "e-signature", "fill"]

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def clean_text(text: str) -> str:
    return ' '.join(text.split())

def sentence_based_chunking(text: str, chunk_size: int = CHUNK_SIZE_SENTENCES) -> List[str]:
    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), max(1, chunk_size - 1)):
        chunk = " ".join(sentences[i:i + chunk_size])
        if MIN_CHUNK_LENGTH_CHARS <= len(chunk) <= MAX_CHUNK_LENGTH_CHARS:
            chunks.append(clean_text(chunk))
    return chunks

def extract_title(text: str, max_len: int = 80) -> str:
    match = re.search(r'^\s*(\d{1,2}\.|\u2022|\-|\*)?\s*([A-Z][^\n]{10,100})', text, re.MULTILINE)
    if match:
        return clean_text(match.group(2))[:max_len]
    for line in text.split('\n')[:3]:
        line = line.strip()
        if 10 < len(line) < 120:
            return clean_text(line)[:max_len]
    return clean_text(text.split('\n', 1)[0])[:max_len]

def text_signature(text: str) -> str:
    return sha256(text[:300].lower().encode()).hexdigest()

def highlight_keywords(text: str, keywords: List[str]) -> str:
    for kw in keywords:
        text = re.sub(fr"(?i)\\b({re.escape(kw)})\\b", r"**\\1**", text)
    return text

def process_document(doc_info: Dict[str, str]) -> List[Dict[str, Any]]:
    filename = doc_info.get("filename")
    file_path = os.path.join(DOCUMENTS_DIR, filename)
    chunks = []
    if not os.path.isfile(file_path):
        logging.warning(f"File '{file_path}' not found. Skipping.")
        return chunks

    try:
        with pdfplumber.open(file_path) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                try:
                    text = page.extract_text()
                    if not text:
                        continue
                    for chunk in sentence_based_chunking(text):
                        chunks.append({
                            "document": filename,
                            "page_number": page_idx,
                            "refined_text": chunk,
                        })
                except Exception as e:
                    logging.warning(f"Failed to parse page {page_idx} in {filename}: {e}")
    except Exception as e:
        logging.error(f"Failed to process {filename}: {e}")
    return chunks

def find_top_candidates(intent_embedding: np.ndarray, chunks: List[Dict[str, Any]], model: SentenceTransformer) -> List[Dict[str, Any]]:
    if not chunks:
        return []

    logging.info(f"Computing embeddings for {len(chunks)} chunks...")
    chunk_texts = [chunk["refined_text"] for chunk in chunks]
    chunk_embeddings = model.encode(chunk_texts, batch_size=EMBEDDING_BATCH_SIZE, normalize_embeddings=True, show_progress_bar=False)

    similarities = cosine_similarity(intent_embedding, chunk_embeddings)[0]

    for i, chunk in enumerate(chunks):
        chunk["score"] = float(similarities[i])
        chunk["section_title"] = extract_title(chunk["refined_text"])

    chunks.sort(key=lambda x: x["score"], reverse=True)
    seen_hashes: Set[str] = set()
    unique_candidates = []

    for candidate in chunks[:DUPLICATE_CHECK_POOL]:
        sig = text_signature(candidate["refined_text"])
        if sig not in seen_hashes:
            seen_hashes.add(sig)
            candidate["highlighted_text"] = highlight_keywords(candidate["refined_text"], HIGHLIGHT_KEYWORDS)
            candidate["confidence"] = bucket_confidence(candidate["score"])
            unique_candidates.append(candidate)
        if len(unique_candidates) >= TOP_N_CANDIDATES:
            break
    return unique_candidates[:TOP_N_CANDIDATES]

def bucket_confidence(score: float) -> str:
    if score >= 0.80:
        return "High"
    elif score >= 0.60:
        return "Medium"
    else:
        return "Low"

def format_output(top_candidates: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    output = {
        "metadata": {
            "input_documents": [d.get("filename", "N/A") for d in config.get("documents", [])],
            "persona": config.get("persona", {}).get("role", ""),
            "job_to_be_done": config.get("job_to_be_done", {}).get("task", ""),
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": [],
        "subsection_analysis": []
    }
    for rank, item in enumerate(top_candidates, start=1):
        output["extracted_sections"].append({
            "document": item["document"],
            "page_number": item["page_number"],
            "section_title": item["section_title"],
            "importance_rank": rank,
            "confidence": item["confidence"]
        })
        output["subsection_analysis"].append({
            "document": item["document"],
            "highlighted_text": item["highlighted_text"],
            "page_number": item["page_number"],
            "similarity_score": round(item["score"], 4)
        })
    return output

def main():
    start_time = time.time()

    try:
        with open(INPUT_CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        logging.critical(f"Error loading input config: {e}")
        return

    documents = config.get("documents", [])
    persona = config.get("persona", {}).get("role", "user")
    job = config.get("job_to_be_done", {}).get("task", "analyze documents")
    intent_query = (
        f"As a {persona}, my primary task is to {job}. "
        "I am looking for specific, step-by-step guidance or tools that help me complete this task effectively. "
        "Focus on tools like Fill & Sign, e-signatures, and form creation in Acrobat."
    )

    model = SentenceTransformer(MODEL_NAME, device='cuda' if SentenceTransformer(MODEL_NAME)._target_device == 'cuda' else 'cpu')
    model.max_seq_length = MAX_SEQ_LENGTH
    intent_embedding = model.encode([intent_query], normalize_embeddings=True).reshape(1, -1)

    with ThreadPoolExecutor() as executor:
        all_results = list(executor.map(process_document, documents))
    all_chunks = [chunk for sublist in all_results for chunk in sublist]

    if not all_chunks:
        logging.critical("No valid text chunks extracted.")
        return

    top_candidates = find_top_candidates(intent_embedding, all_chunks, model)
    final_output = format_output(top_candidates, config)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    total_time = time.time() - start_time
    logging.info(f"Completed in {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
