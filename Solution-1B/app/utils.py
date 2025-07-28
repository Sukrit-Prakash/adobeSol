import re

def clean_text(text):
    return ' '.join(text.split())

def extract_title(text, max_len=80):
    match = re.search(r'^\s*(\d{1,2}\.|•|\-|\*)?\s*([A-Z][^\n]{10,100})', text, re.MULTILINE)
    if match:
        return clean_text(match.group(2))[:max_len]
    lines = text.split('\n')
    for line in lines:
        if 10 < len(line.strip()) < 120:
            return clean_text(line)[:max_len]
    return clean_text(lines[0])[:max_len]
