import faiss
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple
import openai
from dotenv import load_dotenv
import os

def save_faiss_index(embeddings: np.ndarray, index_path: str):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_path)

def load_faiss_index(index_path: str) -> faiss.IndexFlatL2:
    return faiss.read_index(index_path)

def save_metadata(metadata: List[Dict], metadata_path: str):
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def load_metadata(metadata_path: str) -> List[Dict]:
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)

def search_index(index: faiss.IndexFlatL2, query_embedding: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    # query_embedding shape: (1, dim)
    D, I = index.search(query_embedding, top_k)
    return D, I

# Example usage:
if __name__ == "__main__":
    # Load environment and OpenAI key
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Load index and metadata
    index = load_faiss_index("data/faiss.index")
    metadata = load_metadata("data/metadata.json")

    # Get a query from the user
    query = input("Enter your test query: ")

    # Generate embedding for the query
    EMBEDDING_MODEL = "text-embedding-ada-002"
    response = openai.Embedding.create(input=query, model=EMBEDDING_MODEL)
    query_embedding = np.array([response["data"][0]["embedding"]], dtype="float32")

    # Search index
    D, I = search_index(index, query_embedding, top_k=5)

    print("\nTop 5 results:")
    for rank, (idx, dist) in enumerate(zip(I[0], D[0]), 1):
        print(f"\nResult {rank}:")
        print("Distance:", dist)
        print("Metadata:", metadata[idx])