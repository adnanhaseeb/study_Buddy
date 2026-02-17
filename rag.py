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

def load_all_chunks(jsonl_path="data/ingested.jsonl"):
    """Load all document chunks from the ingested JSONL file."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def chunk_text_for_summary(chunks, max_chars=8000):
    """Group chunks into batches that fit within token limits."""
    batches = []
    current_batch = []
    current_length = 0
    
    for chunk in chunks:
        chunk_text = chunk["text"]
        chunk_length = len(chunk_text)
        
        # If adding this chunk exceeds limit, start new batch
        if current_length + chunk_length > max_chars and current_batch:
            batches.append(current_batch)
            current_batch = [chunk]
            current_length = chunk_length
        else:
            current_batch.append(chunk)
            current_length += chunk_length
    
    # Add the last batch if not empty
    if current_batch:
        batches.append(current_batch)
    
    return batches

def generate_document_summary(document_title="document", max_summary_length=500):
    """Generate a comprehensive summary of the entire document."""
    try:
        # Load all chunks
        all_chunks = load_all_chunks()
        if not all_chunks:
            return "No document content found to summarize."
        
        # Group chunks into manageable batches
        chunk_batches = chunk_text_for_summary(all_chunks)
        
        if len(chunk_batches) == 1:
            # Single batch - direct summarization
            content = "\n\n".join([chunk["text"] for chunk in chunk_batches[0]])
            
            prompt = (
                f"Create a comprehensive summary of the following document content. "
                f"The summary should be concise but cover all key points. "
                f"Use ONLY the information provided in the document.\n\n"
                f"Document Content:\n{content}\n\n"
                f"Summary:"
            )
            
            response = openai.ChatCompletion.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=max_summary_length,
            )
            
            return response["choices"][0]["message"]["content"].strip()
        
        else:
            # Multiple batches - summarize each batch then combine
            batch_summaries = []
            
            for i, batch in enumerate(chunk_batches):
                content = "\n\n".join([chunk["text"] for chunk in batch])
                
                prompt = (
                    f"Summarize the key points from this section of the document. "
                    f"Focus on the main ideas and important details. "
                    f"Use ONLY the information provided.\n\n"
                    f"Section Content:\n{content}\n\n"
                    f"Section Summary:"
                )
                
                response = openai.ChatCompletion.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=300,
                )
                
                batch_summaries.append(response["choices"][0]["message"]["content"].strip())
            
            # Combine batch summaries into final summary
            combined_content = "\n\n".join(batch_summaries)
            
            final_prompt = (
                f"Create a comprehensive summary by combining these section summaries. "
                f"Ensure all key points are covered while maintaining conciseness. "
                f"Remove any redundancy between sections.\n\n"
                f"Section Summaries:\n{combined_content}\n\n"
                f"Final Comprehensive Summary:"
            )
            
            response = openai.ChatCompletion.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.1,
                max_tokens=max_summary_length,
            )
            
            return response["choices"][0]["message"]["content"].strip()
    
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def generate_key_points(top_k=10):
    """Extract key points from the document using chunk importance."""
    try:
        all_chunks = load_all_chunks()
        if not all_chunks:
            return "No document content found."
        
        # For simplicity, take first few chunks and last few chunks
        # In a more sophisticated version, we could rank by importance
        important_chunks = all_chunks[:top_k//2] + all_chunks[-top_k//2:]
        content = "\n\n".join([chunk["text"] for chunk in important_chunks])
        
        prompt = (
            f"Extract the key points and main ideas from this document content. "
            f"Present them as a bulleted list of the most important information. "
            f"Use ONLY the information provided in the document.\n\n"
            f"Document Content:\n{content}\n\n"
            f"Key Points:"
        )
        
        response = openai.ChatCompletion.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=400,
        )
        
        return response["choices"][0]["message"]["content"].strip()
    
    except Exception as e:
        return f"Error generating key points: {str(e)}"

if __name__ == "__main__":
    print("StudyBuddy RAG System")
    print("=" * 30)
    
    while True:
        print("\nOptions:")
        print("1. Ask a question")
        print("2. Generate document summary")  
        print("3. Generate key points")
        print("4. Quit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            question = input("\nAsk a question: ")
            answer, retrieved = answer_question(question)
            
            print(f"\nQuestion: {question}")
            print(f"Answer: {answer}")
            
            if retrieved:
                print(f"\nRetrieved {len(retrieved)} relevant chunks:")
                for i, chunk in enumerate(retrieved, 1):
                    print(f"\nChunk {i}: {chunk[:200]}{'...' if len(chunk) > 200 else ''}")
            else:
                print("\nNo relevant chunks found above similarity threshold.")
        
        elif choice == "2":
            print("\nGenerating document summary...")
            summary = generate_document_summary()
            print(f"\nDocument Summary:\n{summary}")
        
        elif choice == "3":
            print("\nExtracting key points...")
            key_points = generate_key_points()
            print(f"\nKey Points:\n{key_points}")
        
        elif choice == "4":
            print("Goodbye!")
            break
        
        else:
            print("Invalid option. Please select 1-4.")