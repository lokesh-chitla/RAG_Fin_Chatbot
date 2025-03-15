import os
import faiss
import numpy as np
import pdfplumber
import json
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_pdfs.bin")
TABLES_JSON_PATH = os.path.join(DATA_DIR, "financial_tables.json")
CHUNKS_JSON_PATH = os.path.join(DATA_DIR, "pdf_chunks.json")

model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_path, chunk_size=512):
    text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pg_text = page.extract_text()
            if pg_text:
                text.append(pg_text)
    full_text = "\n".join(text)
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
    return chunks

def extract_tables_from_pdf(pdf_path, pdf_name):
    tables_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            extracted_tables = page.extract_tables()
            for table in extracted_tables:
                df = pd.DataFrame(table)
                df = df.rename(columns=df.iloc[0]).drop(df.index[0])  
                df.dropna(how='all', inplace=True)
                if not df.empty:
                    table_dict = {
                        "pdf_name": pdf_name,
                        "page": page_idx+1,
                        "data": df.to_dict(orient="records")
                    }
                    tables_data.append(table_dict)
    return tables_data

if __name__ == "__main__":
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise ValueError("No PDFs found in the data folder!")

    all_text_chunks = []
    chunk_metadata = []
    all_tables = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(DATA_DIR, pdf_file)
        print("Processing:", pdf_file)
        text_chunks = extract_text_from_pdf(pdf_path)
        for chunk in text_chunks:
            all_text_chunks.append(chunk)
            # Store chunk metadata for retrieval
            chunk_metadata.append({
                "pdf_file": pdf_file,
                "text": chunk,
                # Optional: doc_id if you want multi-stage or matching
                # "doc_id": ...
            })
        # Extract tables
        tables_data = extract_tables_from_pdf(pdf_path, pdf_file)
        all_tables.extend(tables_data)

    print("Generating embeddings...")
    embeddings = model.encode(all_text_chunks, convert_to_numpy=True)
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings)
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    print("Saved FAISS index:", FAISS_INDEX_PATH)

    with open(TABLES_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_tables, f, indent=4)
    print("Saved tables:", TABLES_JSON_PATH)

    with open(CHUNKS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(chunk_metadata, f, indent=4)
    print("Saved chunk metadata:", CHUNKS_JSON_PATH)
    print("All done!")
