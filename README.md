# 📄 Solution 1B – Document-Driven Intelligence for PDF Processing

Welcome to **Solution 1B** of the Adobe India Hackathon!  
This solution implements an intelligent document processing pipeline that extracts semantic embeddings from PDFs, enables efficient document clustering & similarity analysis, and answers user queries contextually using local models — all in a lightweight, offline setup under **200MB**.

---

## 🚀 Features

- ✅ **Semantic understanding of PDF content** using Sentence-BERT.
- ✅ **Lightweight local processing** (<200MB; no internet required).
- ✅ **Optimized document embedding & indexing** with FAISS or cosine similarity.
- ✅ **Post-query semantic answer generation**.
- ✅ **Supports batch document analysis**.
- ✅ **Runs fully offline inside Docker**.

---

## 🏗️ Architecture Overview

```plaintext
                        +----------------------+
                        |   Input PDFs         |
                        +----------------------+
                                  |
                                  v
                        +----------------------+
                        |   PDF Text Extractor |
                        |    (PyMuPDF)         |
                        +----------------------+
                                  |
                                  v
                        +----------------------+
                        |  Sentence Chunking   |
                        |   (via NLTK)         |
                        +----------------------+
                                  |
                                  v
                        +----------------------+
                        | Embedding Generator  |
                        | (Sentence-BERT)      |
                        +----------------------+
                                  |
                                  v
                        +----------------------+
                        | Similarity Scorer    |
                        | (FAISS / Cosine)     |
                        +----------------------+
                                  |
                                  v
                    +------------------------------+
                    |  Relevant Chunk Retrieval     |
                    +------------------------------+
                                  |
                                  v
                      +------------------------+
                      |   Answer Generator     |
                      |   (Local Template /    |
                      |     Transformer)       |
                      +------------------------+

Setup Instructions
🔧 1. Clone the Repository
git clone https:/Sukrit-Prakash/adobeSol/github.com/.git
cd adobe-solution-1b
🐋 2. Build the Docker Image
docker build -t doc_processor .
▶️ 3. Run the Processor
Place your PDFs in the data/ folder and modify input.json as needed.

docker run --rm -v "$(pwd):/app" doc_processor python app/main.py --config input.json

| 🛠️ Tool                     | 🔍 Purpose                                     |
| ---------------------------- | ---------------------------------------------- |
| 📄 **PyMuPDF**               | High-performance PDF text extraction           |
| 🧠 **NLTK**                  | Sentence tokenization                          |
| 🔗 **sentence-transformers** | Embedding generation (MiniLM)                  |
| 📊 **scikit-learn**          | Cosine similarity computation                  |
| ⚡ **FAISS** *(optional)*     | Fast dense vector indexing (retrieval)         |
| 🐳 **Docker**                | Containerized execution & dependency isolation |


Solution-1B/
│
├── app/
│   ├── main.py              # Entry point
│   ├── pdf_processor.py     # PDF reading & text cleaning
│   ├── embedder.py          # Sentence-BERT wrapper
│   ├── similarity.py        # Similarity engine
│   ├── answer_generator.py  # Query response logic
│
├── data/                    # PDF files go here
├── input.json               # User query config
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker setup
└── README.md                # 📘 You're reading it!


Example input.json
{
  "query": "How do I export PDFs in Acrobat?",
  "input_dir": "data",
  "use_faiss": false,
  "top_k": 3
}


Sample Output (Terminal)
Query: How do I export PDFs in Acrobat?

Relevant Passages:
1. "To export a PDF, open it in Acrobat and go to File > Export To > Microsoft Word..."
2. "You can also convert PDFs to Excel or PowerPoint using the Export tool."
3. "The Export PDF tool allows selection of file types like DOCX, XLSX, etc."

Generated Answer:
"To export PDFs in Acrobat, use the 'Export PDF' tool under the File menu. You can choose formats such as Word, Excel, or PowerPoint based on your needs."
