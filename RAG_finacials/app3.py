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

# Load SentenceTransformer for embeddings
try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    st.sidebar.error(f"Error loading embedder: {e}")
    embedder = None

# Define paths for data files
DATA_DIR = "data"
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_pdfs.bin")
CHUNKS_JSON_PATH = os.path.join(DATA_DIR, "pdf_chunks.json")
DOC_JSON_PATH = os.path.join(DATA_DIR, "all_docs.json")

# Load FAISS index for vector search
try:
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"FAISS index file {FAISS_INDEX_PATH} not found!")
    faiss_pdf = faiss.read_index(FAISS_INDEX_PATH)
except Exception as e:
    st.sidebar.error(f"Error loading FAISS index: {e}")
    faiss_pdf = None

# Load PDF chunk metadata
try:
    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        chunk_metadata = json.load(f)
except Exception as e:
    st.sidebar.error(f"Error loading chunk metadata: {e}")
    chunk_metadata = None

# Load BM25 model for keyword-based retrieval
try:
    with open(DOC_JSON_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)
    bm25_corpus = [doc["text"].lower().split() for doc in documents]
    bm25 = BM25Okapi(bm25_corpus)
except Exception as e:
    st.sidebar.error(f"Error loading BM25 data: {e}")
    bm25 = None

# === Financial Query Classification ===
classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

def is_financial_query(query: str) -> bool:
    """
    Determines if the query is financial using zero-shot classification.
    Returns True if classified as financial, otherwise False.
    """
    candidate_labels = ["financial", "non-financial"]
    result = classifier(query, candidate_labels)
    return result['labels'][0] == 'financial'

# === Multi-Stage Retrieval ===
def multi_stage_retrieve(query: str):
    """
    Multi-stage retrieval combining BM25 (coarse search) and FAISS (fine search).
    - BM25 retrieves top documents.
    - FAISS refines retrieval using embeddings.
    Returns a list of relevant document chunks.
    """
    if not bm25 or not chunk_metadata or not embedder or not faiss_pdf:
        return []
    
    try:
        # Step 1: BM25 keyword search
        bm25_scores = bm25.get_scores(query.lower().split())
        top_doc_indices = np.argsort(bm25_scores)[-5:][::-1]  # Select top 5 docs
        filtered_indices = [idx for idx in top_doc_indices if idx < len(chunk_metadata)]

        # Step 2: FAISS embedding similarity search
        filtered_embeddings = np.array([embedder.encode(chunk_metadata[idx]["text"]) for idx in filtered_indices])
        query_emb = embedder.encode([query], convert_to_numpy=True).squeeze()
        similarities = np.dot(filtered_embeddings, query_emb)  # Cosine similarity
        sorted_indices = np.argsort(similarities)[-3:][::-1]  # Select top 3 matches

        results = [
            {
                "pdf_file": chunk_metadata[filtered_indices[idx]]["pdf_file"],
                "chunk_id": filtered_indices[idx],
                "distance": similarities[idx],
                "text": chunk_metadata[filtered_indices[idx]]["text"]
            }
            for idx in sorted_indices
        ]
        return results
    except Exception as e:
        st.error(f"Error in multi-stage retrieval: {e}")
        return []

# === Response Generation ===
def generate_response(query: str, mode: str = "multi-stage") -> str:
    """
    Generates a response based on retrieved document chunks.
    Uses multi-stage retrieval by default, falling back to FAISS-only retrieval.
    """
    if not all([embedder, faiss_pdf, chunk_metadata, bm25]):
        return "One or more models or data failed to load. Please check the sidebar for errors."

    if not is_financial_query(query):
        return "This is not a financial query. Please ask something related to finance."

    # Retrieve relevant documents
    if mode == "multi-stage":
        results = multi_stage_retrieve(query)
    else:
        # FAISS-only retrieval fallback
        try:
            query_emb = embedder.encode([query], convert_to_numpy=True)
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
            st.error(f"Error in FAISS retrieval: {e}")
            return "Error during FAISS retrieval."

    # Format response
    if not results:
        return "No relevant data found."
    
    top_result = results[0]  # Best match
    return f"**Answer:** {top_result['text']}\n\n**Confidence Score:** {top_result['distance']:.4f}"

# === Streamlit UI ===
def main():
    """
    Streamlit UI for financial chatbot.
    Allows users to enter queries and retrieve financial information.
    """
    st.title("RAG Financial Chatbot")

    # User selects retrieval mode
    retrieval_mode = st.sidebar.selectbox("Retrieval Mode", ["multi-stage", "basic"])
    user_query = st.text_input("Enter your query:")

    # Process query on button click
    if st.button("Submit", disabled=not user_query.strip()):
        with st.spinner("Processing..."):
            answer = generate_response(user_query, mode=retrieval_mode)
            st.markdown("### Answer")
            st.write(answer, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
