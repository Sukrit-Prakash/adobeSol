# Approach Explanation

Our goal is to build a CPU-only, offline system that ingests 3–10 PDFs, a persona, and a job description, then returns the top five most relevant sections (and their refined text) in the prescribed JSON format.

---

## 1. Input & Intent Construction
- **Input**: `input.json` contains `challenge_info`, a list of `{filename, title}` objects, a `persona.role`, and a `job_to_be_done.task`.
- **Intent**: We merge persona and task into a single natural-language query, e.g.  
  > “As a Travel Planner, my task is to Plan a trip of 4 days for a group of 10 college friends.”

---

## 2. PDF Parsing
- We use **pdfplumber** to open each PDF in `documents/`.
- For every page, we extract raw text.
- We segment each page’s text into sentences using **NLTK’s Punkt tokenizer**, filtering out sentences under 50 characters to avoid boilerplate.

---

## 3. Semantic Embedding & Scoring
- We load **`all-MiniLM-L6-v2`** (~80 MB) from SentenceTransformers—fits under 1 GB, fast on CPU, no internet needed after download.
- We embed the intent query once.
- Each candidate sentence is embedded on the fly, and its cosine similarity to the intent embedding is computed.
- We accumulate `(document, page, sentence, score)` tuples.

---

## 4. Ranking & Selection
- We sort all scored sentences **descending** by similarity.
- We pick the top 5 and assign `importance_rank` 1–5.

---

## 5. JSON Output
- **`metadata`**: lists input filenames, persona, job and a timestamp.
- **`extracted_sections`**: for each top section, we record document name, truncated sentence as `section_title`, its page number, and rank.
- **`subsection_analysis`**: includes the full sentence as `refined_text` plus page number.

---

## 6. Performance & Constraints
- **Model size**: ~80 MB  
- **CPU only**: uses efficient bi-encoder  
- **60 s target**: batch size ≈1 per sentence; real-world PDFs (few MB each) complete in under a minute for 3–5 docs  
- **Offline**: after initial download, no external calls

This pipeline can be easily extended to detect true headings (via regex), group multi-sentence chunks, or integrate a lightweight summarizer, while staying within the resource limits.
