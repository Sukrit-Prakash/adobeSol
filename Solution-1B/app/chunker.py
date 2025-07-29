# chunker.py
import nltk
from nltk.tokenize import sent_tokenize
from utils import clean_text

CHUNK_SIZE = 3
MIN_CHARS = 100
MAX_CHARS = 1500

def split_text_to_chunks(text):
    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), CHUNK_SIZE - 1):
        chunk = " ".join(sentences[i:i+CHUNK_SIZE])
        chunk = clean_text(chunk)
        if MIN_CHARS <= len(chunk) <= MAX_CHARS:
            chunks.append(chunk)
    return chunks
