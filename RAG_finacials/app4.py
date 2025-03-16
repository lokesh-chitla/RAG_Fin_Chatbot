import os
import streamlit as st
import faiss
import json
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from huggingface_hub import login

# Configure logging
logger = logging.getLogger(__name__)

# === Load Models & Data ===
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

# Load Flan-T5 for response generation
try:
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    slm_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    slm_pipeline = pipeline("text2text-generation", model=slm_model, tokenizer=tokenizer)
except Exception:
    slm_pipeline = None

# Load SentenceTransformer for generating query embeddings
try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    embedder = None

# Define paths for data files
DATA_DIR = "data"
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_pdfs.bin")
CHUNKS_JSON_PATH = os.path.join(DATA_DIR, "pdf_chunks.json")
DOC_JSON_PATH = os.path.join(DATA_DIR, "all_docs.json")

# Load FAISS index for vector search
try:
    faiss_pdf = faiss.read_index(FAISS_INDEX_PATH)
except Exception:
    faiss_pdf = None

# Load PDF chunk metadata
try:
    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        chunk_metadata = json.load(f)
except Exception:
    chunk_metadata = None

# Load BM25 model for keyword-based retrieval
try:
    with open(DOC_JSON_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)
    bm25_corpus = [doc["text"].lower().split() for doc in documents]
    bm25 = BM25Okapi(bm25_corpus)
except Exception:
    bm25 = None

# === Financial Query Classification ===
try:
    classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
except Exception:
    classifier = None

def is_financial_query(query: str) -> bool:
    if not classifier:
        return False
    candidate_labels = ["financial", "non-financial"]
    try:
        result = classifier(query, candidate_labels)
        return result['labels'][0] == 'financial'
    except Exception:
        return False

# === Basic Retrieval ===
def basic_retrieve(query: str):
    if not embedder or not faiss_pdf or not chunk_metadata:
        return []
    try:
        query_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        distances, indices = faiss_pdf.search(query_emb, 3)
        return [{
            "pdf_file": chunk_metadata[idx]["pdf_file"],
            "chunk_id": idx,
            "similarity": round(1 / (1 + distances[0][i]) if distances[0][i] > 0 else 1.0, 4),
            "text": chunk_metadata[idx]["text"]
        } for i, idx in enumerate(indices[0]) if idx < len(chunk_metadata)]
    except Exception:
        return []

# === Multi-Stage Retrieval ===
def multi_stage_retrieve(query: str):
    if not bm25 or not documents:
        return []
    bm25_results = bm25.get_top_n(query.lower().split(), documents, n=5)
    return basic_retrieve(bm25_results[0]['text']) if bm25_results else []

# === Response Generation ===
def generate_response(query: str, mode: str = "multi-stage"):
    if not all([embedder, faiss_pdf, chunk_metadata, bm25]):
        return "One or more models or data failed to load. Please check the settings.", "N/A"
    if not is_financial_query(query):
        return "This is not a financial query. Please ask something related to finance.", "N/A"
    results = multi_stage_retrieve(query) if mode == "multi-stage" else basic_retrieve(query)
    if not results:
        return "No relevant data found.", "N/A"
    top_result = results[0]
    return top_result['text'], f"{top_result['similarity']:.4f}"

# === Streamlit UI ===
def main():
    st.set_page_config(page_title="RAG Financial Chatbot", layout="wide")
    st.title("💰 RAG Financial Chatbot")
    st.markdown("AI-powered retrieval system for financial queries.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Settings")
        retrieval_mode = st.selectbox("Retrieval Mode", ["multi-stage", "basic"], index=0)
        query = st.text_area("📌 Enter your query:", height=150)
        search_clicked = st.button("🔍 Search")
    
    with col2:
        st.header("Results")
        if search_clicked and query.strip():
            with st.spinner("Retrieving data..."):
                answer, confidence = generate_response(query, mode=retrieval_mode)
            st.markdown("### Answer")
            st.info(answer)
            st.markdown("### Confidence Score")
            st.success(confidence)

if __name__ == "__main__":
    main()
