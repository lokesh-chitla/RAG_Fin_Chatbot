import os
import json
import pdfplumber

DATA_DIR = "data"
DOC_JSON_PATH = os.path.join(DATA_DIR, "all_docs.json")

def create_doc_level_json():
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    documents = []
    doc_id_counter = 1
    for pdf_file in pdf_files:
        pdf_path = os.path.join(DATA_DIR, pdf_file)
        all_text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    all_text.append(txt)
        doc_text = "\n".join(all_text)
        documents.append({
            "doc_id": doc_id_counter,
            "text": doc_text,
            "pdf_file": pdf_file
        })
        doc_id_counter += 1

    with open(DOC_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=4)
    print("Created:", DOC_JSON_PATH)

if __name__ == "__main__":
    create_doc_level_json()
