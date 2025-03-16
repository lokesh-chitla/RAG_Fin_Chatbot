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
DATA_DIR = "data"  # Directory where data files are stored
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_pdfs.bin")  # Path to FAISS index file
CHUNKS_JSON_PATH = os.path.join(DATA_DIR, "pdf_chunks.json")  # Path to PDF chunk metadata
DOC_JSON_PATH = os.path.join(DATA_DIR, "all_docs.json")  # Path to document data for BM25

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
    bm25_corpus = [doc["text"].lower().split() for doc in documents]  # Preprocess text for BM25
    bm25 = BM25Okapi(bm25_corpus)  # Initialize BM25 model
except Exception as e:
    st.sidebar.error(f"Error loading BM25 data: {e}")
    bm25 = None

# === Financial Query Classification ===
classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

def is_financial_query(query: str) -> bool:
    """
    Determines if the query is financial using zero-shot classification.
    
    Args:
        query (str): The user's query to classify.
    
    Returns:
        bool: True if the query is classified as financial, otherwise False.
    """
    candidate_labels = ["financial", "non-financial"]  # Labels for classification
    result = classifier(query, candidate_labels)  # Perform classification
    return result['labels'][0] == 'financial'  # Return True if top label is "financial"

# === Basic Retrieval ===
def basic_retrieve(query: str):
    """
    Retrieves relevant document chunks based on the query using FAISS for vector search.
    
    Args:
        query (str): The user's query to retrieve results for.
    
    Returns:
        list: A list of dictionaries containing retrieval results, including PDF file, chunk ID,
              similarity score, and text. Returns an empty list if an error occurs or models are not loaded.
    """
    if not embedder or not faiss_pdf or not chunk_metadata:
        return []  # Return empty list if required models or data are not loaded
    
    try:
        # Encode the query into an embedding
        query_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        
        # Search the FAISS index for the top 3 most similar chunks
        distances, indices = faiss_pdf.search(query_emb, 3)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(chunk_metadata):  # Ensure the index is within bounds
                # Calculate similarity score (inverse of distance)
                similarity_score = 1 / (1 + distances[0][i]) if distances[0][i] > 0 else 1.0
                
                # Append result with metadata and similarity score
                results.append({
                    "pdf_file": chunk_metadata[idx]["pdf_file"],  # PDF file name
                    "chunk_id": idx,  # Chunk ID
                    "similarity": round(similarity_score, 4),  # Similarity score rounded to 4 decimal places
                    "text": chunk_metadata[idx]["text"]  # Text content of the chunk
                })

        # Debugging: Uncomment to log retrieved results
        # st.write("DEBUG: Retrieved results →", results)
        
        return results
    except Exception as e:
        st.error(f"Error in basic retrieval: {e}")
        return []  # Return empty list if an error occurs
# === Response Generation ===
def generate_response(query: str, mode: str = "multi-stage") -> str:
    """
    Generates a response based on retrieved document chunks.
    """
    if not all([embedder, faiss_pdf, chunk_metadata, bm25]):
        return "One or more models or data failed to load. Please check the sidebar for errors."

    if not is_financial_query(query):
        return "This is not a financial query. Please ask something related to finance."

    results = multi_stage_retrieve(query) if mode == "multi-stage" else basic_retrieve(query)
    
    if not results:
        return "No relevant data found."
    
    top_result = results[0]  # Best match
    similarity_score = top_result.get("similarity", 0.0)  # Default to 0.0 instead of "N/A"

    return f"**Answer:** {top_result['text']}\n\n**Confidence Score:** {similarity_score:.4f}"

# === Multi-Stage Retrieval ===
def multi_stage_retrieve(query: str):
    if not bm25 or not chunk_metadata or not embedder or not faiss_pdf:
        return []
    
    try:
        bm25_scores = bm25.get_scores(query.lower().split())
        top_doc_indices = np.argsort(bm25_scores)[-5:][::-1]
        filtered_indices = [idx for idx in top_doc_indices if idx < len(chunk_metadata)]

        query_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).squeeze()
        filtered_embeddings = np.array([embedder.encode(chunk_metadata[idx]["text"], convert_to_numpy=True, normalize_embeddings=True) for idx in filtered_indices])
        
        similarities = np.dot(filtered_embeddings, query_emb)
        sorted_indices = np.argsort(similarities)[-3:][::-1]

        results = [
            {
                "pdf_file": chunk_metadata[filtered_indices[idx]]["pdf_file"],
                "chunk_id": filtered_indices[idx],
                "similarity": round(similarities[idx], 4),  # Ensure similarity is always there
                "text": chunk_metadata[filtered_indices[idx]]["text"]
            }
            for idx in sorted_indices
        ]

        #st.write("DEBUG: Retrieved results →", results)  # Debugging
        return results
    except Exception as e:
        st.error(f"Error in multi-stage retrieval: {e}")
        return []

# === Streamlit UI ===
def main():
    """
    Streamlit UI for financial chatbot.
    Users enter queries, which are processed upon pressing Enter.
    """
    st.title("RAG Financial Chatbot")

    retrieval_mode = st.sidebar.selectbox("Retrieval Mode", ["multi-stage", "basic"])
    user_query = st.text_area("Enter your query:", key="query", height=100)  # Larger input box

    # Automatically process query when user presses Enter
    if user_query.strip():
        with st.spinner("Processing..."):
            answer = generate_response(user_query, mode=retrieval_mode)
            
            # Split the answer into "Answer" and "Confidence Score"
            if isinstance(answer, str) and "**Answer:**" in answer and "**Confidence Score:**" in answer:
                answer_text = answer.split("**Answer:**")[1].split("**Confidence Score:**")[0].strip()
                confidence_score = answer.split("**Confidence Score:**")[1].strip()
            else:
                answer_text = answer
                confidence_score = "N/A"

            # Display results sequentially
            st.markdown("### Answer")
            st.info(answer_text)  # Display answer in a box

            st.markdown("### Confidence Score")
            st.success(confidence_score)  # Display confidence score in a box below the answer

if __name__ == "__main__":
    main()
