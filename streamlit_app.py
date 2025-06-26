#!/usr/bin/env python3
"""
Batch PDF Processor for RAG System
Reads all PDFs from a folder, processes them, generates embeddings, and stores in Pinecone.
"""

import os
import hashlib
import json
import re
from pathlib import Path
from typing import List, Dict, Any
import argparse
from datetime import datetime

import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from tqdm import tqdm

# --- CONFIG ---
INDEX_NAME = "prebuilt-demo"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
BATCH_SIZE = 100  # Number of vectors to upsert at once

# Load secrets (for Streamlit deployment)
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
except:
    # Fallback for local development - set these environment variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY and PINECONE_API_KEY environment variables")

# --- INIT ---
pinecone = Pinecone(api_key=PINECONE_API_KEY)

def create_index_if_not_exists():
    """Create Pinecone index if it doesn't exist."""
    if not pinecone.has_index(INDEX_NAME):
        print(f"Creating index: {INDEX_NAME}")
        pinecone.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Index {INDEX_NAME} created successfully!")
    else:
        print(f"Using existing index: {INDEX_NAME}")

def get_file_hash(file_path: str) -> str:
    """Generate SHA-256 hash of file for change detection."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def load_processing_state(state_file: str) -> Dict[str, Any]:
    """Load processing state from JSON file."""
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            return json.load(f)
    return {"processed_files": {}}

def save_processing_state(state_file: str, state: Dict[str, Any]):
    """Save processing state to JSON file."""
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)

def clean_text(text: str) -> str:
    """Clean text by removing special characters and replacing with dots."""
    # Replace special characters with dots, but keep common punctuation
    cleaned = re.sub(r'[^\w\s\.\'\"\?\!]', '.', text)
    # Replace multiple consecutive dots with a single dot
    cleaned = re.sub(r'\.+', '.', cleaned)
    # Replace multiple spaces with single space
    cleaned = re.sub(r'\s+', ' ', cleaned)
    # Remove leading/trailing whitespace
    cleaned = cleaned.strip()
    return cleaned

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text.strip():  # Only add non-empty pages
                    text += page_text
            # Clean the extracted text
            cleaned_text = clean_text(text)
            return cleaned_text
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return ""

def process_pdf_file(pdf_path: str, embeddings, index, state: Dict[str, Any], state_file: str) -> bool:
    """Process a single PDF file and store in Pinecone."""
    file_hash = get_file_hash(pdf_path)
    file_info = state["processed_files"].get(pdf_path, {})
    
    # Check if file has been processed and hasn't changed
    if file_info.get("hash") == file_hash and file_info.get("processed"):
        print(f"✓ {os.path.basename(pdf_path)} - Already processed (no changes)")
        return True
    
    print(f"Processing: {os.path.basename(pdf_path)}")
    
    # Extract text
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print(f"⚠ Warning: No text extracted from {pdf_path}")
        return False
    
    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )
    docs = splitter.create_documents([text])
    
    if not docs:
        print(f"⚠ Warning: No chunks created from {pdf_path}")
        return False
    
    # Generate embeddings
    doc_texts = [doc.page_content for doc in docs]
    doc_vectors = embeddings.embed_documents(doc_texts)
    
    # Prepare vectors with metadata
    vectors = []
    for i, (text_chunk, vector) in enumerate(zip(doc_texts, doc_vectors)):
        vector_id = f"{os.path.basename(pdf_path)}-chunk-{i}-{file_hash[:8]}"
        metadata = {
            "text": text_chunk,
            "source_file": pdf_path,
            "file_name": os.path.basename(pdf_path),
            "chunk_index": i,
            "total_chunks": len(docs),
            "file_hash": file_hash,
            "processed_at": datetime.now().isoformat()
        }
        vectors.append((vector_id, vector, metadata))
    
    # Upsert vectors in batches
    try:
        for i in range(0, len(vectors), BATCH_SIZE):
            batch = vectors[i:i + BATCH_SIZE]
            index.upsert(vectors=batch)
        
        # Update processing state
        state["processed_files"][pdf_path] = {
            "hash": file_hash,
            "processed": True,
            "chunks": len(docs),
            "processed_at": datetime.now().isoformat()
        }
        save_processing_state(state_file, state)
        
        print(f"✓ {os.path.basename(pdf_path)} - Processed {len(docs)} chunks")
        return True
        
    except Exception as e:
        print(f"✗ Error processing {pdf_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Batch process PDFs and store in Pinecone")
    parser.add_argument("folder_path", help="Path to folder containing PDF files")
    parser.add_argument("--state-file", default="pdf_processing_state.json", 
                       help="File to store processing state (default: pdf_processing_state.json)")
    parser.add_argument("--force-reprocess", action="store_true", 
                       help="Force reprocessing of all files")
    parser.add_argument("--clear-index", action="store_true", 
                       help="Clear the entire index before processing")
    
    args = parser.parse_args()
    
    folder_path = Path(args.folder_path)
    if not folder_path.exists():
        print(f"Error: Folder {folder_path} does not exist")
        return
    
    # Initialize Pinecone
    create_index_if_not_exists()
    index = pinecone.Index(INDEX_NAME)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    # Clear index if requested
    if args.clear_index:
        print("Clearing index...")
        index.delete(delete_all=True)
        print("Index cleared!")
    
    # Load processing state
    state = load_processing_state(args.state_file)
    
    # Find all PDF files
    pdf_files = list(folder_path.glob("**/*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return
    
    print(f"Found {len(pdf_files)} PDF files")
    
    # Process files
    successful = 0
    failed = 0
    
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        if args.force_reprocess:
            # Remove from state to force reprocessing
            state["processed_files"].pop(str(pdf_file), None)
        
        if process_pdf_file(str(pdf_file), embeddings, index, state, args.state_file):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Total files: {len(pdf_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"State saved to: {args.state_file}")
    
    # Show index stats
    try:
        stats = index.describe_index_stats()
        total_vectors = stats.get('total_vector_count', 0)
        print(f"Total vectors in index: {total_vectors}")
    except Exception as e:
        print(f"Could not retrieve index stats: {e}")

if __name__ == "__main__":
    main() 