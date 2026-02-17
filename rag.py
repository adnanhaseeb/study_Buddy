import os
import json
import numpy as np
import openai
from dotenv import load_dotenv
from vectorstore import load_faiss_index, load_metadata, search_index

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-3.5-turbo"  # Or your preferred model
openai.api_base = "https://openrouter.ai/api/v1"

def get_embedding(text: str) -> np.ndarray:
    response = openai.Embedding.create(input=text, model=EMBEDDING_MODEL)
    return np.array([response["data"][0]["embedding"]], dtype="float32")

def get_chunk_texts(metadata, indices, jsonl_path="data/ingested.jsonl"):
    # Load all chunks from JSONL
    with open(jsonl_path, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    return [chunks[i]["text"] for i in indices]

def answer_question(question: str, top_k: int = 5):
    # 1. Embed question
    query_emb = get_embedding(question)

    # 2. Retrieve top-K chunks
    index = load_faiss_index("data/faiss.index")
    metadata = load_metadata("data/metadata.json")
    D, I = search_index(index, query_emb, top_k=top_k)
    indices = I[0].tolist()
    retrieved_texts = get_chunk_texts(metadata, indices)

    # 3. Build context for LLM
    context = "\n\n".join(retrieved_texts)
    prompt = (
        f"Answer the following question using ONLY the provided context. "
        f"If the answer is not present, reply exactly: 'Not found in document.'\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    # 4. Call LLM
    response = openai.ChatCompletion.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=500,
    )
    answer = response["choices"][0]["message"]["content"].strip()
    return answer, retrieved_texts

if __name__ == "__main__":
    question = input("Ask a question: ")
    answer, retrieved = answer_question(question)
    print("\nAnswer:\n", answer)
    print("\nRetrieved context:\n", "\n---\n".join(retrieved))