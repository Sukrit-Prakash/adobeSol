# import fitz
# import csv
# import os

# def extract_features_from_pdf(pdf_path, label_map, output_csv, append=False):
#     doc = fitz.open(pdf_path)
#     rows = []
#     for page_num, page in enumerate(doc, start=1):
#         prev_y = None
#         blocks = page.get_text("dict")["blocks"]
#         for block in blocks:
#             for line in block.get("lines", []):
#                 for span in line.get("spans", []):
#                     text = span['text'].strip()
#                     if not text:
#                         continue
#                     font_size = span['size']
#                     font_name = span['font']
#                     flags = span['flags']
#                     length = len(text)
#                     y0 = span['bbox'][1]
#                     spacing = y0 - prev_y if prev_y is not None else 0
#                     prev_y = y0

#                     label = label_map.get(text, "BODY")  # default label
#                     rows.append([text, font_size, font_name, flags, length, spacing, label])

#     mode = 'a' if append else 'w'
#     with open(output_csv, mode, newline='', encoding='utf-8') as f:
#         writer = csv.writer(f)
#         if not append:
#             writer.writerow(["text", "font_size", "font_name", "flags", "text_length", "spacing", "label"])
#         writer.writerows(rows)

import fitz  # PyMuPDF
import csv
from pathlib import Path
import re
from collections import Counter

def label_by_heuristics(text, size, flags, font_size_distribution):
    # Use top N% of font sizes as H1/H2/H3
    percentile_90 = font_size_distribution[int(len(font_size_distribution) * 0.9)]
    percentile_75 = font_size_distribution[int(len(font_size_distribution) * 0.75)]

    # Bold, caps, etc.
    is_bold = flags & 2
    is_caps = text.isupper()

    if size >= percentile_90 and (is_bold or is_caps):
        return "H1"
    elif re.match(r'^\\d+\\.\\s+', text):
        return "H2"
    elif re.match(r'^\\d+(\\.\\d+)+\\s+', text):
        return "H3"
    else:
        return "BODY"

def extract_features_auto_label(pdf_path: str, output_csv: str, append=False):
    doc = fitz.open(pdf_path)
    all_spans = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    all_spans.append(span)

    sizes_sorted = sorted([s["size"] for s in all_spans])

    rows = []
    for page in doc:
        prev_y = None
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if not text:
                        continue
                    font_size = span["size"]
                    font_name = span["font"]
                    flags = span["flags"]
                    length = len(text)
                    y0 = span["bbox"][1]
                    spacing = y0 - prev_y if prev_y is not None else 0
                    prev_y = y0

                    label = label_by_heuristics(text, font_size, flags, sizes_sorted)
                    rows.append([
                        text, font_size, font_name, flags, length, spacing, label
                    ])

    mode = "a" if append else "w"
    with open(output_csv, mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not append:
            writer.writerow([
                "text", "font_size", "font_name", "flags", "text_length", "spacing", "label"
            ])
        writer.writerows(rows)

    print(f"Extracted {len(rows)} rows from {pdf_path}")
