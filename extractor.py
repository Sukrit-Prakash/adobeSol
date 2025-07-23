# # extractor.py


# extractor.py
import fitz  # PyMuPDF
import re
from utils import get_body_font_size, map_sizes_to_levels

class OutlineExtractor:
    def __init__(self, pdf_path: str):
        self.doc = fitz.open(pdf_path)
        self.sizes = []
        self.blocks = []

        # Extract font sizes on first page for profiling
        for block in self.doc[0].get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    self.sizes.append(span["size"])

        self.body_size = get_body_font_size(self.sizes)
        self.heading_threshold = self.body_size * 1.2
        self.size_to_level = map_sizes_to_levels(set(self.sizes))

    def extract(self) -> dict:
        title = self._get_title()
        flat_headings = []
        prev_y = None

        for page_no, page in enumerate(self.doc, start=1):
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span["text"].strip()
                        if not text:
                            continue

                        size = span["size"]
                        flags = span.get("flags", 0)
                        y0 = span.get("bbox", [0, 0, 0, 0])[1]  # top Y coordinate

                        if self._is_heading(text, size, flags, y0, prev_y):
                            level = self._assign_level(text, size)
                            flat_headings.append({
                                "level": level,
                                "text": text,
                                "page": page_no
                            })

                        prev_y = y0

        structured_outline = self._build_hierarchy(flat_headings)
        return {"title": title, "outline": structured_outline}

    def _get_title(self) -> str:
        max_span = ('', 0)
        for block in self.doc[0].get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span['size'] > max_span[1]:
                        max_span = (span['text'].strip(), span['size'])
        return max_span[0]

    # 🔍 Multi-feature scoring logic
    def _is_heading(self, text: str, size: float, flags: int, y0: float, prev_y: float) -> bool:
        score = 0
        if size >= self.heading_threshold:
            score += 2
        if re.match(r'^\d+(\.\d+)*', text):
            score += 1
        if flags & 2:  # Bold
            score += 1
        if text.isupper():
            score += 1
        if prev_y is not None and (y0 - prev_y) > 5:
            score += 1
        return score >= 3  # Threshold to classify as heading

    def _assign_level(self, text: str, size: float) -> str:
        if size in self.size_to_level:
            return self.size_to_level[size]
        m = re.match(r'^(\d+(?:\.\d+)*)', text)
        if m:
            depth = m.group(1).count('.') + 1
            return f"H{min(depth, 3)}"
        return "H3"

    # 🧱 Auto-group into hierarchy
    def _build_hierarchy(self, headings: list) -> list:
        hierarchy = []
        stack = []

        for heading in headings:
            level_num = int(heading["level"][1])
            while stack and int(stack[-1]["level"][1]) >= level_num:
                stack.pop()
            if stack:
                stack[-1].setdefault("children", []).append(heading)
            else:
                hierarchy.append(heading)
            stack.append(heading)
        print("Hierarchy built:", hierarchy)

        return hierarchy


# import fitz  # PyMuPDF
# import re
# from utils import get_body_font_size, map_sizes_to_levels

# class OutlineExtractor:
#     def __init__(self, pdf_path: str): # Corrected: __init__ instead of _init_
#         self.doc = fitz.open(pdf_path)
#         # Profile fonts on first page
#         sizes = []
#         for block in self.doc[0].get_text("dict")["blocks"]:
#             for line in block.get("lines", []):
#                 for span in line.get("spans", []):
#                     sizes.append(span['size'])
#         self.body_size = get_body_font_size(sizes)
#         # threshold to consider: >= 1.2 * body_size
#         self.heading_threshold = self.body_size * 1.2
#         self.size_to_level = map_sizes_to_levels(set(sizes))
    
#     def extract(self) -> dict:
#         title = self._get_title()
#         outline = []
#         for page_no, page in enumerate(self.doc, start=1):
#             blocks = page.get_text("dict")["blocks"]
#             for block in blocks:
#                 for line in block.get("lines", []):
#                     for span in line.get("spans", []):
#                         text = span['text'].strip()
#                         if not text:
#                             continue
#                         size = span['size']
#                         if self._is_heading(text, size):
#                             lvl = self._assign_level(text, size)
#                             outline.append({"level": lvl, "text": text, "page": page_no})
#         return {"title": title, "outline": outline}
    
#     def _get_title(self) -> str:
#         # Largest single text span on page 1
#         max_span = ('', 0)
#         for block in self.doc[0].get_text("dict")["blocks"]:
#             # print("Processing block:", block)
#             for line in block.get("lines", []):
#                 # print("Processing line:", line)
#                 for span in line.get("spans", []):
#                     # print("Processing span:", span)
#                     if span['size'] > max_span[1]:
#                         max_span = (span['text'].strip(), span['size'])
#         return max_span[0]
    
    
#     # Not all headings follow size/number patterns — especially in PDFs with inconsistent formatting.
#     def _is_heading(self, text: str, size: float) -> bool:
#         # size-based
#         if size >= self.heading_threshold:
#             return True
#         # numbering pattern
#         if re.match(r'^\d+(\.\d+)*\s+', text):
#             return True
#         return False
    
#     def _assign_level(self, text: str, size: float) -> str:
#         # by size
#         if size in self.size_to_level:
#             return self.size_to_level[size]
#         # by numbering depth
#         m = re.match(r'^(\d+(?:\.\d+)*)', text)
#         if m:
#             depth = m.group(1).count('.') + 1
#             return f"H{min(depth,3)}"
#         # default to H3
#         return "H3"

    
    #  shit after this code
    # import fitz  # PyMuPDF
# from config import HEADING_FONT_RELATIVE_THRESHOLD

# def extract_pdf_outline(pdf_path):
#     doc = fitz.open(pdf_path)
#     blocks = []
#     font_stats = []

#     for page_num, page in enumerate(doc, start=1):
#         for block in page.get_text("dict")["blocks"]:
#             if "lines" not in block:
#                 continue
#             for line in block["lines"]:
#                 text = "".join([span["text"] for span in line["spans"]]).strip()
#                 if not text:
#                     continue
#                 font_size = line["spans"][0]["size"]
#                 blocks.append({
#                     "text": text,
#                     "font": font_size,
#                     "page": page_num
#                 })
#                 font_stats.append(font_size)

#     if not blocks:
#         return None

#     # Get the most frequent font size
#     from statistics import mode
#     try:
#         base_font = mode(font_stats)
#     except:
#         base_font = sorted(font_stats)[len(font_stats) // 2]

#     # Identify title: largest font on page 1
#     title = ""
#     page1_blocks = [b for b in blocks if b["page"] == 1]
#     if page1_blocks:
#         title_block = max(page1_blocks, key=lambda b: b["font"])
#         title = title_block["text"]

#     # Determine headings
#     outline = []
#     for block in blocks:
#         font_ratio = block["font"] / base_font
#         if font_ratio >= HEADING_FONT_RELATIVE_THRESHOLD["H1"]:
#             level = "H1"
#         elif font_ratio >= HEADING_FONT_RELATIVE_THRESHOLD["H2"]:
#             level = "H2"
#         elif font_ratio >= HEADING_FONT_RELATIVE_THRESHOLD["H3"]:
#             level = "H3"
#         else:
#             continue
#         outline.append({
#             "level": level,
#             "text": block["text"],
#             "page": block["page"]
#         })

#     return {
#         "title": title,
#         "outline": outline
#     }
