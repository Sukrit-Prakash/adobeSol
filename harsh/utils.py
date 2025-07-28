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