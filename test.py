import faiss
import os

DATA_DIR = r"C:\Users\bapun\Downloads\Order_approval\data"
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_pdfs.bin")

# Check if the file exists
if not os.path.exists(FAISS_INDEX_PATH):
    print("❌ FAISS index file not found!")
else:
    # Load FAISS index
    index = faiss.read_index(FAISS_INDEX_PATH)
    print(f"✅ FAISS index loaded with {index.ntotal} entries.")
