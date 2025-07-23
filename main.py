
# import json
# from pathlib import Path
# from extractor import OutlineExtractor

# # Hardcoded paths
# input_file = Path("input/sample_input.pdf")
# output_dir = Path("output")
# output_file = output_dir / (input_file.stem + ".json")

# # Ensure output directory exists
# output_dir.mkdir(parents=True, exist_ok=True)

# # Run the extractor
# json_data = OutlineExtractor(str(input_file)).extract()
# # # json_data = OutlineExtractor(input_file).extract()
# # json_data = OutlineExtractor().extract()

# # Write output
# with open(output_file, 'w', encoding='utf-8') as f:
#     json.dump(json_data, f, ensure_ascii=False, indent=2)

# print(f"Processed: {input_file.name} -> {output_file.name}")

import json
from pathlib import Path
from extractor import OutlineExtractor # Assuming extractor.py is in the same directory

# Hardcoded paths
input_file = Path("input/sample2.pdf")
output_dir = Path("output")
output_file = output_dir / (input_file.stem + ".json")

# Ensure output directory exists
output_dir.mkdir(parents=True, exist_ok=True)

# Run the extractor
# Make sure 'input' directory exists and 'sample_input.pdf' is inside it
json_data = OutlineExtractor(str(input_file)).extract()

# Write output
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)

print(f"Processed: {input_file.name} -> {output_file.name}")



# import os
# import traceback
# from utils import list_pdfs, save_json, ensure_directories
# from extractor import extract_pdf_outline

# INPUT_DIR = "/app/input"
# OUTPUT_DIR = "/app/output"

# def main():
#     ensure_directories()
#     try:
#         pdfs = list_pdfs(INPUT_DIR)
#     except Exception as e:
#         print(f"Error listing PDFs: {e}")
#         traceback.print_exc()
#         return

#     for pdf in pdfs:
#         pdf_path = os.path.join(INPUT_DIR, pdf)
#         try:
#             result = extract_pdf_outline(pdf_path)
#             if result:
#                 out_file = pdf.replace(".pdf", ".json")
#                 output_path = os.path.join(OUTPUT_DIR, out_file)
#                 save_json(output_path, result)
#                 print(f"Processed: {pdf} → {out_file}")
#             else:
#                 print(f"Failed to process: {pdf}")
#         except Exception as e:
#             print(f"Error processing {pdf}: {e}")
#             traceback.print_exc()

# if __name__ == "__main__":
#     main()