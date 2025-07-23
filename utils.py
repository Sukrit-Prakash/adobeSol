
import statistics
from collections import Counter

# Determine body font size as the mode of all sizes
def get_body_font_size(sizes):
    try:
        return Counter(sizes).most_common(1)[0][0]
    except IndexError:
        return statistics.median(sizes)

# Map the top three sizes to heading levels H1, H2, H3
def map_sizes_to_levels(unique_sizes):
    # unique_sizes: list of distinct sizes, unsorted
    sorted_sizes = sorted(unique_sizes, reverse=True)
    mapping = {}
    for i, size in enumerate(sorted_sizes[:3]):
        mapping[size] = f"H{i+1}"
    return mapping



# import os
# import json

# def list_pdfs(input_dir):
#     return [f for f in os.listdir(input_dir) if f.endswith(".pdf")]

# def save_json(output_path, data):
#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(data, f, indent=2)

# def ensure_directories():
#     os.makedirs("/app/input", exist_ok=True)
#     os.makedirs("/app/output", exist_ok=True)
