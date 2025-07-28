import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import argparse
import re


from processor import process_pdf
from embedder import get_model, get_embeddings
from dedup import deduplicate_chunks

logging.basicConfig(level=logging.INFO)



def highlight_keywords(text, keywords):
    for kw in keywords:
        text = re.sub(fr"(?i)\b({re.escape(kw)})\b", r"**\1**", text)
    return text

def main(config_path, output_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    docs = config.get("documents", [])
    # print(docs,"documents")
    persona = config.get("persona", {}).get("role", "user")
    job = config.get("job_to_be_done", {}).get("task", "analyze documents")
    intent = f"As a {persona}, my primary task is to {job}. I am looking for specific tools and steps."

    model = get_model()
    intent_embed = get_embeddings(model, [intent])

    logging.info("Processing documents...")
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_pdf, docs))

    all_chunks = [c for sub in results for c in sub]
    texts = [c["refined_text"] for c in all_chunks]
    embeds = get_embeddings(model, texts)
    
    # if not embeds or len(embeds) == 0:
    #    logging.critical("No embeddings generated from documents. Exiting.")
    #    return

    sims = cosine_similarity(intent_embed, embeds)[0]

    for i, chunk in enumerate(all_chunks):
        chunk["score"] = float(sims[i])

    all_chunks.sort(key=lambda x: x["score"], reverse=True)
    top_chunks = deduplicate_chunks(all_chunks)

    keywords = ["fill", "sign", "form", "signature", "acrobat"]
    for chunk in top_chunks:
        chunk["highlighted_text"] = highlight_keywords(chunk["refined_text"], keywords)
        score = chunk["score"]
        chunk["confidence"] = "high" if score > 0.75 else "medium" if score > 0.6 else "low"

    output = {
        "metadata": {
            "persona": persona,
            "task": job,
            "processed_documents": [d["filename"] for d in docs],
            "timestamp": datetime.now().isoformat()
        },
        "results": top_chunks
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logging.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../input.json", help="Path to input config")
    parser.add_argument("--output", default="challenge1b_output.json", help="Output JSON path")
    args = parser.parse_args()
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", required=True, help="Path to input config")
    # parser.add_argument("--output", default="challenge1b_output.json", help="Output JSON path")
    # args = parser.parse_args()
    


    start = time.time()
    main(args.config, args.output)
    logging.info(f"Completed in {time.time() - start:.2f}s")
