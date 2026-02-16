import os
import json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict
import openai
from dotenv import load_dotenv
load_dotenv()
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
print("OPENAI_API_KEY loaded:", os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-ada-002"  # Or your preferred model

def load_chunks(jsonl_path: Path) -> List[Dict]:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def get_embedding(text: str) -> List[float]:
    response = openai.Embedding.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response["data"][0]["embedding"]

def main(
    input_jsonl="data/ingested.jsonl",
    faiss_index_path="data/faiss.index",
    metadata_path="data/metadata.json"
):
    # Load chunks
    chunks = load_chunks(Path(input_jsonl))
    embeddings = []
    metadatas = []

    for chunk in chunks:
        emb = get_embedding(chunk["text"])
        embeddings.append(emb)
        metadatas.append(chunk["metadata"])

    # Convert to numpy array
    embeddings_np = np.array(embeddings).astype("float32")

    # Build FAISS index
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)
    faiss.write_index(index, faiss_index_path)

    # Save metadata for later retrieval
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadatas, f, ensure_ascii=False, indent=2)

    print(f"Embeddings and metadata saved. FAISS index: {faiss_index_path}")

if __name__ == "__main__":
    openai.api_key = os.getenv("OPENAI_API_KEY")
    main()