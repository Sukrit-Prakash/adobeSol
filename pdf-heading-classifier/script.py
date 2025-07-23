from pathlib import Path
from feature_extractor import extract_features_auto_label

pdf_dir = Path("training_pdfs")
output_csv = "dataset.csv"
first = True

for pdf_file in pdf_dir.glob("*.pdf"):
    try:
        extract_features_auto_label(str(pdf_file), output_csv, append=not first)
        first = False
        print(f"Processed: {pdf_file}")
    except Exception as e:
        print(f"Error processing {pdf_file}: {e}")
