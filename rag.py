import os
import json
import csv
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

def get_chunks_with_metadata(indices, jsonl_path="data/ingested.jsonl"):
    """Get chunk texts along with their metadata for citations."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    
    results = []
    for i in indices:
        chunk = chunks[i]
        results.append({
            'text': chunk['text'],
            'metadata': chunk['metadata']
        })
    return results

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
        return "Not found in document.", [], []
    
    retrieved_chunks = get_chunks_with_metadata(relevant_indices)
    retrieved_texts = [chunk['text'] for chunk in retrieved_chunks]
    citations = [f"[{chunk['metadata']['filename']}, Page {chunk['metadata']['page_number']}, Chunk {chunk['metadata']['chunk_index']}]" 
                for chunk in retrieved_chunks]

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
        return "Not found in document.", retrieved_texts, citations
    
    return answer, retrieved_texts, citations

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

def generate_flashcards(num_flashcards=15, output_file="data/flashcards.csv"):
    """Generate Q&A flashcards from the document content."""
    try:
        all_chunks = load_all_chunks()
        if not all_chunks:
            return "No document content found for flashcard generation."
        
        # Select diverse chunks for flashcard generation
        step = max(1, len(all_chunks) // num_flashcards)
        selected_chunks = all_chunks[::step][:num_flashcards]
        
        flashcards = []
        
        for i, chunk in enumerate(selected_chunks):
            content = chunk["text"]
            metadata = chunk["metadata"]
            
            # Generate question-answer pair from chunk
            prompt = (
                f"Create a clear, specific question and answer based on the following content. "
                f"The question should test understanding of key information. "
                f"Format your response as:\n"
                f"Question: [your question]\n"
                f"Answer: [your answer]\n\n"
                f"Content: {content}"
            )
            
            response = openai.ChatCompletion.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=200,
            )
            
            result = response["choices"][0]["message"]["content"].strip()
            
            # Parse the response
            lines = result.split('\n')
            question = ""
            answer = ""
            
            for line in lines:
                if line.startswith("Question:"):
                    question = line.replace("Question:", "").strip()
                elif line.startswith("Answer:"):
                    answer = line.replace("Answer:", "").strip()
            
            if question and answer:
                citation = f"{metadata['filename']} (Page {metadata['page_number']}, Chunk {metadata['chunk_index']})"
                flashcards.append({
                    'Question': question,
                    'Answer': answer,
                    'Citation': citation,
                    'Tags': 'StudyBuddy'
                })
        
        # Export to CSV (Anki compatible format)
        if flashcards:
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['Question', 'Answer', 'Citation', 'Tags']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(flashcards)
            
            return f"Generated {len(flashcards)} flashcards and saved to {output_file}"
        else:
            return "No flashcards could be generated from the document content."
    
    except Exception as e:
        return f"Error generating flashcards: {str(e)}"

def export_summary_with_citations(output_file="data/summary_with_citations.txt"):
    """Generate and export a summary with full citation information."""
    try:
        summary = generate_document_summary()
        all_chunks = load_all_chunks()
        
        # Create citation list
        citations = []
        for i, chunk in enumerate(all_chunks):
            metadata = chunk["metadata"]
            citation = f"[{i+1}] {metadata['filename']}, Page {metadata['page_number']}, Chunk {metadata['chunk_index']}"
            citations.append(citation)
        
        # Combine summary with citations
        full_content = f"Document Summary\\n{'='*50}\\n\\n{summary}\\n\\n"
        full_content += f"Citations\\n{'='*20}\\n"
        full_content += "\\n".join(citations[:20])  # Limit to first 20 citations
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_content)
        
        return f"Summary with citations exported to {output_file}"
    
    except Exception as e:
        return f"Error exporting summary: {str(e)}"

if __name__ == "__main__":
    print("StudyBuddy RAG System")
    print("=" * 30)
    
    while True:
        print("\nOptions:")
        print("1. Ask a question")
        print("2. Generate document summary")  
        print("3. Generate key points")
        print("4. Generate flashcards")
        print("5. Export summary with citations")
        print("6. Quit")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == "1":
            question = input("\nAsk a question: ")
            answer, retrieved, citations = answer_question(question)
            
            print(f"\nQuestion: {question}")
            print(f"Answer: {answer}")
            
            if retrieved and citations:
                print(f"\nRetrieved {len(retrieved)} relevant chunks:")
                for i, (chunk, citation) in enumerate(zip(retrieved, citations), 1):
                    print(f"\nChunk {i}: {chunk[:150]}{'...' if len(chunk) > 150 else ''}")
                    print(f"Citation: {citation}")
            elif not retrieved:
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
            num_cards = input("\nNumber of flashcards to generate (default 15): ").strip()
            if not num_cards:
                num_cards = 15
            else:
                try:
                    num_cards = int(num_cards)
                except ValueError:
                    print("Invalid number, using default of 15")
                    num_cards = 15
            
            print(f"\nGenerating {num_cards} flashcards...")
            result = generate_flashcards(num_cards)
            print(f"\n{result}")
        
        elif choice == "5":
            print("\nExporting summary with citations...")
            result = export_summary_with_citations()
            print(f"\n{result}")
        
        elif choice == "6":
            print("Goodbye!")
            break
        
        else:
            print("Invalid option. Please select 1-6.")