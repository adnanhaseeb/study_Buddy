import streamlit as st
import os
import tempfile
import pandas as pd
from pathlib import Path
import json
from rag import (
    answer_question, 
    generate_document_summary, 
    generate_key_points, 
    generate_flashcards,
    export_summary_with_citations
)
from ingest import ingest
from embeddings import main as generate_embeddings

# Page configuration
st.set_page_config(
    page_title="StudyBuddy - Personal RAG Learning Assistant",
    page_icon="SB",
    layout="wide"
)

# Initialize session state
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def check_system_ready():
    """Check if the system has processed documents and is ready for queries."""
    required_files = [
        "data/ingested.jsonl",
        "data/faiss.index", 
        "data/metadata.json"
    ]
    return all(os.path.exists(f) for f in required_files)

def process_uploaded_files(uploaded_files):
    """Process uploaded files through the ingestion and embedding pipeline."""
    if not uploaded_files:
        return False, "No files uploaded."
    
    try:
        # Create temporary directory for uploaded files
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        # Save uploaded files
        saved_files = []
        for uploaded_file in uploaded_files:
            file_path = temp_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            saved_files.append(file_path)
        
        # Process through ingestion pipeline
        total_chunks = ingest(
            input_path=temp_dir,
            output_path=Path("data/ingested.jsonl"),
            chunk_size=600,
            chunk_overlap=100
        )
        
        # Generate embeddings
        generate_embeddings(
            input_jsonl="data/ingested.jsonl",
            faiss_index_path="data/faiss.index",
            metadata_path="data/metadata.json"
        )
        
        # Clean up temp files
        for file_path in saved_files:
            file_path.unlink()
        temp_dir.rmdir()
        
        return True, f"Successfully processed {len(saved_files)} files with {total_chunks} chunks."
    
    except Exception as e:
        return False, f"Error processing files: {str(e)}"

def main():
    st.title("StudyBuddy - Personal RAG Learning Assistant")
    st.markdown("Upload your study materials and get AI-powered assistance!")
    
    # Sidebar for file upload and system status
    with st.sidebar:
        st.header("Document Management")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="Upload your study materials to get started"
        )
        
        if st.button("Process Documents", type="primary"):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    success, message = process_uploaded_files(uploaded_files)
                    if success:
                        st.success(message)
                        st.session_state.documents_processed = True
                        st.rerun()
                    else:
                        st.error(message)
            else:
                st.warning("Please upload files first!")
        
        # System status
        st.header("System Status") 
        if check_system_ready():
            st.success("System Ready")
            st.session_state.documents_processed = True
        else:
            st.warning("Upload and process documents first")
            st.session_state.documents_processed = False
        
        # Quick stats
        if os.path.exists("data/ingested.jsonl"):
            with open("data/ingested.jsonl", "r") as f:
                chunks = len(f.readlines())
            st.metric("Document Chunks", chunks)
    
    # Main interface
    if not st.session_state.documents_processed:
        st.info("Please upload and process your documents using the sidebar to get started.")
        
        # Show example of what the app can do
        st.header("What StudyBuddy Can Do")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Smart Q&A")
            st.write("Ask questions about your documents and get precise, grounded answers with citations.")
            
            st.subheader("Document Summaries")
            st.write("Generate comprehensive summaries of your study materials.")
        
        with col2:
            st.subheader("Flashcard Generation")
            st.write("Automatically create study flashcards from your documents.")
            
            st.subheader("Export & Citations")
            st.write("Export summaries and flashcards with full citation information.")
    
    else:
        # Main functionality tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Q&A Chat", "Summary", "Flashcards", "Export"])
        
        with tab1:
            st.header("Ask Questions About Your Documents")
            
            # Question input
            question = st.text_input(
                "Enter your question:",
                placeholder="What is the main topic discussed in the document?"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                ask_button = st.button("Ask", type="primary")
            
            if ask_button and question:
                with st.spinner("Searching for answer..."):
                    answer, retrieved_texts, citations = answer_question(question)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer,
                        "citations": citations
                    })
            
            # Display chat history
            if st.session_state.chat_history:
                st.subheader("Chat History")
                for i, chat in enumerate(reversed(st.session_state.chat_history)):
                    with st.expander(f"Q: {chat['question'][:50]}...", expanded=(i==0)):
                        st.markdown(f"**Question:** {chat['question']}")
                        st.markdown(f"**Answer:** {chat['answer']}")
                        if chat['citations']:
                            st.markdown("**Sources:**")
                            for citation in chat['citations']:
                                st.text(f"• {citation}")
        
        with tab2:
            st.header("Document Summary & Key Points")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Generate Summary", type="primary"):
                    with st.spinner("Generating summary..."):
                        summary = generate_document_summary()
                        st.session_state.summary = summary
            
            with col2:
                if st.button("Generate Key Points", type="secondary"):
                    with st.spinner("Extracting key points..."):
                        key_points = generate_key_points()
                        st.session_state.key_points = key_points
            
            # Display results
            if hasattr(st.session_state, 'summary'):
                st.subheader("Document Summary")
                st.write(st.session_state.summary)
            
            if hasattr(st.session_state, 'key_points'):
                st.subheader("Key Points")
                st.write(st.session_state.key_points)
        
        with tab3:
            st.header("Generate Study Flashcards")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                num_flashcards = st.slider(
                    "Number of flashcards to generate:",
                    min_value=5,
                    max_value=50,
                    value=15,
                    step=5
                )
            
            with col2:
                if st.button("Generate Flashcards", type="primary"):
                    with st.spinner(f"Generating {num_flashcards} flashcards..."):
                        result = generate_flashcards(num_flashcards)
                        st.success(result)
                        st.session_state.flashcards_generated = True
            
            # Display flashcards if generated
            if os.path.exists("data/flashcards.csv"):
                st.subheader("Generated Flashcards")
                
                try:
                    df = pd.read_csv("data/flashcards.csv")
                    st.dataframe(df, width="stretch")
                    
                    # Download button
                    with open("data/flashcards.csv", "rb") as file:
                        st.download_button(
                            label="Download Flashcards (CSV)",
                            data=file,
                            file_name="studybuddy_flashcards.csv",
                            mime="text/csv",
                            key="download_flashcards_tab"
                        )
                except Exception as e:
                    st.error(f"Error loading flashcards: {e}")
        
        with tab4:
            st.header("Export & Download")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Summary Export")
                if st.button("Export Summary with Citations", type="secondary"):
                    with st.spinner("Generating export..."):
                        result = export_summary_with_citations()
                        st.success(result)
                
                # Download summary if exists
                if os.path.exists("data/summary_with_citations.txt"):
                    with open("data/summary_with_citations.txt", "r", encoding="utf-8") as file:
                        st.download_button(
                            label="Download Summary (TXT)",
                            data=file.read(),
                            file_name="studybuddy_summary.txt",
                            mime="text/plain",
                            key="download_summary_export"
                        )
            
            with col2:
                st.subheader("Flashcard Export")
                if os.path.exists("data/flashcards.csv"):
                    with open("data/flashcards.csv", "rb") as file:
                        st.download_button(
                            label="Download Flashcards (CSV)",
                            data=file,
                            file_name="studybuddy_flashcards.csv",
                            mime="text/csv",
                            key="download_flashcards_export"
                        )
                    
                    # Show Anki import instructions
                    with st.expander("How to import to Anki"):
                        st.markdown("""
                        **To import flashcards into Anki:**
                        1. Download the CSV file above
                        2. Open Anki and click "Import"
                        3. Select the downloaded CSV file
                        4. Map fields: Question → Front, Answer → Back
                        5. Set Citation and Tags as additional fields
                        6. Click "Import"
                        """)
                else:
                    st.info("Generate flashcards first to enable download.")
            
            # Chat export
            if st.session_state.chat_history:
                st.subheader("Chat History Export")
                chat_data = []
                for chat in st.session_state.chat_history:
                    chat_data.append({
                        "Question": chat["question"],
                        "Answer": chat["answer"],
                        "Citations": "; ".join(chat["citations"]) if chat["citations"] else "None"
                    })
                
                chat_df = pd.DataFrame(chat_data)
                csv = chat_df.to_csv(index=False)
                
                st.download_button(
                    label="Download Chat History (CSV)",
                    data=csv,
                    file_name="studybuddy_chat_history.csv",
                    mime="text/csv",
                    key="download_chat_history"
                )

if __name__ == "__main__":
    main()