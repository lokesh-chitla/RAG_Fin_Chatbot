import os
import json
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
DOCS_JSON_PATH = os.path.join(DATA_DIR, "all_docs.json")
CHUNKS_JSON_PATH = os.path.join(DATA_DIR, "pdf_chunks.json")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_pdfs.bin")

if not os.path.exists(DOCS_JSON_PATH):
    raise FileNotFoundError("Need doc-level JSON for BM25 coarse retrieval!")

with open(DOCS_JSON_PATH, "r", encoding="utf-8") as f:
    documents = json.load(f)

bm25_corpus = [doc["text"].split() for doc in documents]
bm25 = BM25Okapi(bm25_corpus)

if not os.path.exists(CHUNKS_JSON_PATH):
    raise FileNotFoundError(f"No chunk metadata found at {CHUNKS_JSON_PATH}!")

with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
    chunk_metadata = json.load(f)

if not os.path.exists(FAISS_INDEX_PATH):
    raise FileNotFoundError("FAISS index not found! Run embedder.py first.")

faiss_index = faiss.read_index(FAISS_INDEX_PATH)
model = SentenceTransformer("all-MiniLM-L6-v2")

def multi_stage_retrieve(query, top_k_coarse=3, top_k_fine=3):
    # 1) BM25
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    doc_ranking = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
    top_doc_indices = doc_ranking[:top_k_coarse]

    candidate_chunk_indices = []
    for i in top_doc_indices:
        doc_id = documents[i].get("doc_id", None)  # Ensure doc_id is set
        if doc_id is None:
            continue
        for idx, meta in enumerate(chunk_metadata):
            if meta.get("doc_id") == doc_id:
                candidate_chunk_indices.append(idx)

    # 2) FAISS
    query_emb = model.encode([query], convert_to_numpy=True)
    query_emb = query_emb / np.linalg.norm(query_emb)
    distances, indices = faiss_index.search(query_emb, top_k_fine * 10)

    all_hits = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx in candidate_chunk_indices:
            all_hits.append((idx, dist))

    all_hits.sort(key=lambda x: x[1])
    final_hits = all_hits[:top_k_fine]

    results = []
    for idx, dist in final_hits:
        info = chunk_metadata[idx]
        results.append({
            "chunk_id": idx,
            "distance": dist,
            "text": info["text"],
            "pdf_file": info["pdf_file"]
        })
    return results
