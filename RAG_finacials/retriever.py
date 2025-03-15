import os
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"  # use relative path
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_pdfs.bin")
CHUNKS_JSON_PATH = os.path.join(DATA_DIR, "pdf_chunks.json")
TABLES_JSON_PATH = os.path.join(DATA_DIR, "financial_tables.json")

# Check for faiss index
if not os.path.exists(FAISS_INDEX_PATH):
    raise FileNotFoundError(f"❌ FAISS index file {FAISS_INDEX_PATH} not found! Run embedder.py first.")

# Load index
faiss_pdf = faiss.read_index(FAISS_INDEX_PATH)
print(f"✅ FAISS index loaded with {faiss_pdf.ntotal} entries.")

# Load chunk metadata
with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
    chunk_metadata = json.load(f)

# Load table data (optional)
if os.path.exists(TABLES_JSON_PATH):
    with open(TABLES_JSON_PATH, "r", encoding="utf-8") as f:
        financial_tables = json.load(f)
else:
    financial_tables = []

# Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_similar_documents(query, top_k=3):
    """Basic single-stage retrieval. Returns top PDF chunks plus structured data hits."""
    query_emb = model.encode([query], convert_to_numpy=True)
    query_emb = query_emb / np.linalg.norm(query_emb)  # normalize

    distances, indices = faiss_pdf.search(query_emb, top_k)

    results = {"PDF Results": [], "Structured Financial Data": []}

    # Grab chunk text
    for i in range(top_k):
        idx = indices[0][i]
        dist = distances[0][i]
        chunk_info = chunk_metadata[idx]
        snippet = chunk_info["text"]
        pdf_file_name = chunk_info["pdf_file"]
        results["PDF Results"].append(
            f"[{pdf_file_name}] chunk #{idx}, distance={dist:.4f}\n{snippet}"
        )

    # Simple substring search in table data
    structured_hits = []
    for entry in financial_tables:
        if "data" in entry:
            for row in entry["data"]:
                row_str = " | ".join(str(x) for x in row.values())
                if query.lower() in row_str.lower():
                    structured_hits.append(row)

    if structured_hits:
        results["Structured Financial Data"] = structured_hits
    else:
        results["Structured Financial Data"] = ["No structured data found."]

    return results

if __name__ == "__main__":
    # Quick test
    test_q = "What is TCS's net profit?"
    docs = retrieve_similar_documents(test_q)
    print("PDF Results:\n", docs["PDF Results"])
    print("Structured Data:\n", docs["Structured Financial Data"])
