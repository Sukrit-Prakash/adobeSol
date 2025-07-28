import fitz  # PyMuPDF
import re
from utils import get_body_font_size, map_sizes_to_levels

class OutlineExtractor:
    def __init__(self, pdf_path: str):
        self.doc = fitz.open(pdf_path)
        # Profile fonts on first page
        sizes = []
        for span in self.doc[0].get_text("dict")["blocks"]:
            for line in span.get("lines", []):
                for span in line.get("spans", []):
                    sizes.append(span['size'])
        self.body_size = get_body_font_size(sizes)
        # threshold to consider: >= 1.2 * body_size
        self.heading_threshold = self.body_size * 1.2
        self.size_to_level = map_sizes_to_levels(set(sizes))

    def extract(self) -> dict:
        title = self._get_title()
        outline = []
        for page_no, page in enumerate(self.doc, start=1):
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span['text'].strip()
                        if not text:
                            continue
                        size = span['size']
                        if self._is_heading(text, size):
                            lvl = self._assign_level(text, size)
                            outline.append({"level": lvl, "text": text, "page": page_no})
        return {"title": title, "outline": outline}

    def _get_title(self) -> str:
        # Largest single text span on page 1
        max_span = ('', 0)
        for block in self.doc[0].get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span['size'] > max_span[1]:
                        max_span = (span['text'].strip(), span['size'])
        return max_span[0]

    def _is_heading(self, text: str, size: float) -> bool:
        # size-based
        if size >= self.heading_threshold:
            return True
        # numbering pattern
        if re.match(r'^\d+(\.\d+)*\s+', text):
            return True
        return False

    def _assign_level(self, text: str, size: float) -> str:
        # by size
        if size in self.size_to_level:
            return self.size_to_level[size]
        # by numbering depth
        m = re.match(r'^(\d+(?:\.\d+)*)', text)
        if m:
            depth = m.group(1).count('.') + 1
            return f"H{min(depth,3)}"
        # default to H3
        return "H3"