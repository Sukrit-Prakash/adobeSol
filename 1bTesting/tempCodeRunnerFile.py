
def extract_title(text: str, max_len: int = 80) -> str:
    match = re.search(r'^\s*(\d{1,2}\.|•|\-|\*)?\s*([A-Z][^\n]{10,100})', text, re.MULTILINE)
    if match:
        return clean_text(match.group(2))[:max_len]
    for line in text.split('\n')[:3]:
        line = line.strip()
        if 10 < len(line) < 120:
            return clean_text(line)[:max_len]
    return clean_text(text.split('\n', 1)[0])[:max_len]

def process_document(doc_info: Dict[str, str]) -> List[Dict[str, Any]]:
    filename = doc_info.get("filename")
    file_path = os.path.join(DOCUMENTS_DIR, filename)
    if not os.path.isfile(file_path):
        logging.warning(f"File '{file_path}' not found. Skipping.")
        return []

    chunks = []