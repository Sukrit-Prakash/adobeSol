import hashlib

def get_signature(text):
    return hashlib.sha256(text[:300].lower().encode()).hexdigest()

def deduplicate_chunks(chunks, top_n=5, pool=25):
    seen = set()
    result = []
    for chunk in chunks[:pool]:
        sig = get_signature(chunk["refined_text"])
        if sig not in seen:
            seen.add(sig)
            result.append(chunk)
            if len(result) >= top_n:
                break
    return result
