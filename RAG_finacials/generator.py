import os

from retriever import retrieve_similar_documents  # Basic retrieval
# We'll try to import multi-stage. If it fails (no doc-level?), we handle that.
try:
    from multi_stage_retriever import multi_stage_retrieve
    MULTI_STAGE_AVAILABLE = True
except ImportError:
    # If multi_stage_retriever or doc-level JSON is missing, we skip
    MULTI_STAGE_AVAILABLE = False

from transformers import pipeline

# 1️⃣ Initialize a small open-source model pipeline (Flan-T5)
model_name = "google/flan-t5-small"
generator_pipeline = pipeline("text2text-generation", model=model_name)

def generate_response(query, mode="basic"):
    """
    1) Decide which retriever to call (basic vs. multi-stage).
    2) Build a final prompt from retrieved docs.
    3) Truncate prompt if needed, then call Flan-T5 to generate an answer.
    """

    # --- Stage 1: Retrieve top chunks ---
    if mode == "multi-stage" and MULTI_STAGE_AVAILABLE:
        # If user selected multi-stage but doc-level JSON or code is missing, handle that:
        retrieved_hits = multi_stage_retrieve(query)
        # Turn those chunk hits into lines:
        top_chunks = []
        for r in retrieved_hits:
            snippet = f"[{r['pdf_file']}] chunk #{r['chunk_id']}, distance={r['distance']:.4f}\n{r['text']}"
            top_chunks.append(snippet)

        # We won't do structured data in multi-stage unless you specifically incorporate it.
        structured_data = ["No structured data for multi-stage. (Optional to add)"]

    else:
        # Default to single-stage
        results = retrieve_similar_documents(query)
        top_chunks = results["PDF Results"]
        structured_data = results["Structured Financial Data"]

    # --- Stage 2: Build a big prompt ---
    prompt_intro = "You are a financial Q&A assistant. Use the data below.\n\n"
    context_chunks = "\n\n---\n\n".join(top_chunks)

    if structured_data and structured_data[0] != "No structured data found.":
        structured_text = "\n".join(str(row) for row in structured_data)
        prompt_tables = f"\n\nStructured Data:\n{structured_text}\n\n"
    else:
        prompt_tables = "\n\n(No structured data)\n\n"

    final_prompt = (
        f"{prompt_intro}"
        f"Query: {query}\n\n"
        f"Relevant PDF Chunks:\n{context_chunks}\n"
        f"{prompt_tables}"
        "Provide a concise, accurate answer:\n"
    )

    # --- Stage 3: Truncate if needed to avoid T5's 512 token limit
    max_prompt_tokens = 450
    prompt_words = final_prompt.split()
    if len(prompt_words) > max_prompt_tokens:
        final_prompt = " ".join(prompt_words[:max_prompt_tokens])
        final_prompt += "\n\n(Truncated prompt)\n"

    # --- Stage 4: Call the pipeline
    output = generator_pipeline(final_prompt, max_length=256)
    answer = output[0]["generated_text"]
    return answer

# Test
if __name__ == "__main__":
    test_query = "What was TCS's net profit in 2023?"
    print("--- BASIC MODE ---")
    print(generate_response(test_query, mode="basic"))

    print("\n--- MULTI-STAGE MODE ---")
    print(generate_response(test_query, mode="multi-stage"))
