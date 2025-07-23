# import joblib

# class MLHeadingExtractor:
#     def __init__(self, pdf_path):
#         self.doc = fitz.open(pdf_path)
#         self.model = joblib.load("heading_classifier.pkl")
#         self.font_enc = joblib.load("font_encoder.pkl")
#         self.label_enc = joblib.load("label_encoder.pkl")

#     def extract(self):
#         result = []
#         for page_num, page in enumerate(self.doc, start=1):
#             prev_y = None
#             for block in page.get_text("dict")["blocks"]:
#                 for line in block.get("lines", []):
#                     for span in line.get("spans", []):
#                         text = span['text'].strip()
#                         if not text:
#                             continue
#                         y0 = span['bbox'][1]
#                         spacing = y0 - prev_y if prev_y is not None else 0
#                         prev_y = y0

#                         x = [[
#                             span['size'],
#                             self.font_enc.transform([span['font']])[0],
#                             span['flags'],
#                             len(text),
#                             spacing
#                         ]]
#                         label_encoded = self.model.predict(x)[0]
#                         label = self.label_enc.inverse_transform([label_encoded])[0]

#                         if label != "BODY":
#                             result.append({
#                                 "level": label,
#                                 "text": text,
#                                 "page": page_num
#                             })
#         return result


import fitz
import joblib

class MLHeadingExtractor:
    def __init__(self, pdf_path: str):
        self.doc = fitz.open(pdf_path)
        self.model = joblib.load("heading_classifier.pkl")
        self.font_encoder = joblib.load("font_encoder.pkl")
        self.label_encoder = joblib.load("label_encoder.pkl")

    def extract(self) -> dict:
        result = []
        title = self._get_title()

        for page_num, page in enumerate(self.doc, start=1):
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
                        text_length = len(text)
                        y0 = span["bbox"][1]
                        spacing = y0 - prev_y if prev_y is not None else 0
                        prev_y = y0

                        try:
                            font_encoded = self.font_encoder.transform([font_name])[0]
                        except:
                            continue  # skip unknown font

                        features = [[font_size, font_encoded, flags, text_length, spacing]]
                        pred_label = self.model.predict(features)[0]
                        label = self.label_encoder.inverse_transform([pred_label])[0]

                        if label != "BODY":
                            result.append({
                                "level": label,
                                "text": text,
                                "page": page_num
                            })

        return {"title": title, "outline": self._build_hierarchy(result)}

    def _get_title(self) -> str:
        max_span = ("", 0)
        for block in self.doc[0].get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span["size"] > max_span[1]:
                        max_span = (span["text"].strip(), span["size"])
        return max_span[0]

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

        return hierarchy
