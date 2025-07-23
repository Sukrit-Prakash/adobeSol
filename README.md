# 📘 PDF Heading Extractor with Machine Learning

An intelligent tool that extracts and organizes hierarchical outlines (H1–H3) from PDF files using a combination of heuristics and a trained ML classifier.

---

## ✨ Features

- 🔍 Automatically identifies headings (`H1`, `H2`, `H3`) and body text
- 🧠 Uses `RandomForestClassifier` for font/spacing-based prediction
- 🧱 Hierarchically builds outlines based on heading levels
- 📝 Supports both rule-based and model-based extraction
- 📤 Outputs clean JSON outline for structured use

---

## 🧠 Architecture Overview

    +-------------------+
    |   Input PDF File  |
    +---------+---------+
              |
       PyMuPDF Parsing
              ↓
    +--------------------+
    |  Extracted Spans   |
    | (Text, Font, Size) |
    +--------------------+
              |
  ┌───────────┴────────────┐
  ↓                        ↓
              ↓
    +--------------------+
    |  Tagged Text Lines |
    +--------------------+
              ↓
    Outline Builder (Hierarchy via H1/H2/H3)
              ↓
    +--------------------+
    |   JSON Output      |
    +--------------------+

---

## 🔧 Project Structure

pdf-heading-classifier/
├── input/ # Input PDFs
├── output/ # Extracted JSON outputs
├── training_pdfs/ # Training PDFs
├── dataset.csv # Labeled feature dataset
├── balanced_dataset.csv # Class-balanced dataset
├── extractor.py # Heuristic/ML extraction logic
├── feature_extractor.py # Feature + auto-label extraction
├── train_model.py # Trains classifier + saves .pkl
├── script.py # Batch PDF → dataset.csv
├── main.py # Runs extractor on a single PDF
├── heading_classifier.pkl # Trained ML model
├── font_encoder.pkl # Encoded font name model
├── label_encoder.pkl # Encoded H1/H2/H3/BODY labels
└── README.md

---

## 🏗️ Model Features

The classifier is trained on:

| Feature        | Description                          |
|----------------|--------------------------------------|
| `font_size`     | Absolute font size of the span       |
| `font_name`     | Encoded font family                  |
| `flags`         | Bold/italic flags from PyMuPDF       |
| `text_length`   | Number of characters in text span    |
| `spacing`       | Vertical spacing from previous line  |

---

## 🧪 Getting Started

### 1. 🧹 Generate Training Dataset

```bash
python script.py
2. ⚖️ Balance the Dataset
bash
Copy
Edit
python balance_dataset.py
Produces balanced_dataset.csv.

3. 🤖 Train the Classifier
bash
Copy
Edit
python train_model.py
Trains and saves:

heading_classifier.pkl

font_encoder.pkl

label_encoder.pkl

4. 🏁 Run Extraction on New PDF
bash
Copy
Edit
python main.py
Outputs outline to output/filename.json.

Sample Output (JSON)

{
  "title": "Understanding AI",
  "outline": [
    {
      "level": "H1",
      "text": "1. Introduction",
      "page": 1,
      "children": [
        {
          "level": "H2",
          "text": "1.1 Background",
          "page": 1
        }
      ]
    }
  ]
}
```
