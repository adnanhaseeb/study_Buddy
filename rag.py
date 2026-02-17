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

def answer_question(question: str, top_k: int = 5, similarity_threshold: float = 0.8):
    # 1. Embed question
    query_emb = get_embedding(question)

    # 2. Retrieve top-K chunks
    index = load_faiss_index("data/faiss.index")
    metadata = load_metadata("data/metadata.json")
    D, I = search_index(index, query_emb, top_k=top_k)
    
    # 3. Filter by similarity threshold (lower distance = higher similarity)
    relevant_indices = []
    relevant_distances = []
    for i, distance in enumerate(D[0]):
        if distance <= similarity_threshold:  
            relevant_indices.append(I[0][i])
            relevant_distances.append(distance)
    
    # If no relevant chunks found, return "Not found in document"
    if not relevant_indices:
        return "Not found in document.", []
    
    retrieved_texts = get_chunk_texts(metadata, relevant_indices)

    # 4. Build enhanced context for LLM with strict grounding
    context = "\n\n".join(retrieved_texts)
    prompt = (
        f"You are a precise document assistant. Follow these rules STRICTLY:\n"
        f"1. Answer ONLY using information explicitly stated in the provided context\n"
        f"2. Do NOT use any external knowledge or make assumptions\n" 
        f"3. If the context does not contain the answer, respond EXACTLY: 'Not found in document.'\n"
        f"4. Do not say 'based on the context' or similar phrases - just give the direct answer\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )

    # 5. Call LLM with low temperature for consistency
    response = openai.ChatCompletion.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.05,  # Very low temperature for grounded responses
        max_tokens=500,
    )
    answer = response["choices"][0]["message"]["content"].strip()
    
    # 6. Post-processing validation: Check if LLM followed grounding rules
    answer_lower = answer.lower()
    context_lower = context.lower()
    
    # If answer contains information not in context, enforce grounding
    if (len(answer) > 50 and  # Only check substantial answers
        "not found in document" not in answer_lower and
        not any(word in context_lower for word in answer_lower.split()[:10])):  # Check first 10 words
        return "Not found in document.", retrieved_texts
    
    return answer, retrieved_texts

if __name__ == "__main__":
    print("StudyBuddy RAG System with Grounding Enforcement")
    print("=" * 50)
    
    while True:
        question = input("\nAsk a question (or 'quit' to exit): ")
        if question.lower() in ['quit', 'exit', 'q']:
            break
            
        answer, retrieved = answer_question(question)
        
        print(f"\nQuestion: {question}")
        print(f"Answer: {answer}")
        
        if retrieved:
            print(f"\nRetrieved {len(retrieved)} relevant chunks:")
            for i, chunk in enumerate(retrieved, 1):
                print(f"\nChunk {i}: {chunk[:200]}{'...' if len(chunk) > 200 else ''}")
        else:
            print("\nNo relevant chunks found above similarity threshold.")