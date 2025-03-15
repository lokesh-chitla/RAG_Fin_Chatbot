import os
import streamlit as st
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from transformers import pipeline

# === Load Models & Data ===
st.sidebar.title("Loading Models...")

# Load Zero-Shot Classification Model
try:
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
except Exception as e:
    st.sidebar.error(f"Error loading zero-shot classifier: {e}")
    classifier = None

# Embedding model for retrieval
try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    st.sidebar.error(f"Error loading embedder: {e}")
    embedder = None

# Paths
DATA_DIR = "data"
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_pdfs.bin")
CHUNKS_JSON_PATH = os.path.join(DATA_DIR, "pdf_chunks.json")
DOC_JSON_PATH = os.path.join(DATA_DIR, "all_docs.json")

# Load FAISS index
try:
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"FAISS index file {FAISS_INDEX_PATH} not found!")
    faiss_pdf = faiss.read_index(FAISS_INDEX_PATH)
except Exception as e:
    st.sidebar.error(f"Error loading FAISS index: {e}")
    faiss_pdf = None

# Load PDF Chunk Metadata
try:
    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        chunk_metadata = json.load(f)
except Exception as e:
    st.sidebar.error(f"Error loading chunk metadata: {e}")
    chunk_metadata = None

# Load BM25 for Coarse Retrieval
try:
    with open(DOC_JSON_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)
    bm25_corpus = [doc["text"].split() for doc in documents]
    bm25 = BM25Okapi(bm25_corpus)
except Exception as e:
    st.sidebar.error(f"Error loading BM25 data: {e}")
    bm25 = None

# === Financial Query Classification ===
def is_financial_query(query: str) -> bool:
    """
    Uses Hugging Face's zero-shot classification model to determine if a query is financial.
    Returns True if the query is financial, False otherwise.
    """
    if classifier is None:
        st.sidebar.warning("Zero-shot classifier not loaded. Falling back to simple keyword matching.")
        return any(keyword in query.lower() for keyword in ["finance", "stock", "investment"])  # Simple fallback
    
    candidate_labels = ["financial", "non-financial"]
    result = classifier(query, candidate_labels)
    return result['labels'][0] == 'financial'

# === Multi-Stage Retrieval Function ===
def multi_stage_retrieve(query):
    """
    Multi-stage retrieval combining BM25 and FAISS.
    1. BM25 retrieves top documents.
    2. FAISS refines retrieval using embeddings on BM25-filtered docs.
    """
    try:
        # Step 1: BM25 Retrieval
        bm25_scores = bm25.get_scores(query.split())
        top_doc_indices = np.argsort(bm25_scores)[-5:][::-1]  # Get top 5 docs
        filtered_indices = [idx for idx in top_doc_indices if idx < len(chunk_metadata)]

        # Step 2: FAISS Refinement (Search only within BM25 top docs)
        filtered_embeddings = np.array([embedder.encode(chunk_metadata[idx]["text"]) for idx in filtered_indices])
        temp_index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
        temp_index.add(filtered_embeddings)
        query_emb = embedder.encode([query], convert_to_numpy=True)
        distances, indices = temp_index.search(query_emb, 3)

        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                "pdf_file": chunk_metadata[filtered_indices[idx]]["pdf_file"],
                "chunk_id": filtered_indices[idx],
                "distance": distances[0][i],
                "text": chunk_metadata[filtered_indices[idx]]["text"]
            })
        return results
    except Exception as e:
        st.error(f"Error in multi-stage retrieval: {e}")
        return []

# === Response Generation ===
def generate_response(query, mode="multi-stage"):
    """
    Generates an answer using retrieved data.
    """
    if embedder is None or faiss_pdf is None or chunk_metadata is None or bm25 is None:
        return "One or more models or data failed to load. Please check the sidebar for errors."

    if not is_financial_query(query):
        return "This is not a financial query. Please ask something related to finance."

    # --- Step 1: Retrieve relevant chunks ---
    if mode == "multi-stage":
        results = multi_stage_retrieve(query)
    else:
        # Basic FAISS retrieval (fallback)
        try:
            query_emb = embedder.encode([query], convert_to_numpy=True)
            query_emb = query_emb / np.linalg.norm(query_emb)
            distances, indices = faiss_pdf.search(query_emb, 3)
            results = [
                {
                    "pdf_file": chunk_metadata[idx]["pdf_file"],
                    "chunk_id": idx,
                    "distance": distances[0][i],
                    "text": chunk_metadata[idx]["text"]
                }
                for i, idx in enumerate(indices[0])
            ]
        except Exception as e:
            st.error(f"Error in basic FAISS retrieval: {e}")
            return "Error during basic FAISS retrieval."

    # --- Step 2: Format Retrieved Data ---
    top_chunks = [
        f"<b>[{r['pdf_file']}] chunk #{r['chunk_id']}, distance={r['distance']:.4f}</b><br>{r['text']}"
        for r in results
    ]
    return "\n\n".join(top_chunks) if top_chunks else "No relevant data found."

# === Streamlit UI ===
def main():
    st.title("RAG Financial Chatbot")

    # Mode selection
    retrieval_mode = st.sidebar.selectbox("Retrieval Mode", ["multi-stage", "basic"])

    user_query = st.text_input("Enter your query:")

    if st.button("Submit"):
        if user_query.strip():
            with st.spinner("Processing..."):
                answer = generate_response(user_query, mode=retrieval_mode)
                st.markdown("### Answer")
                st.write(answer)
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
