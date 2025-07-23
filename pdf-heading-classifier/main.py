from extractor import MLHeadingExtractor

pdf_path = "input/sample_input.pdf"
extractor = MLHeadingExtractor(pdf_path)
result = extractor.extract()

with open("output/sample.json", "w", encoding="utf-8") as f:
    import json
    json.dump(result, f, indent=2)
