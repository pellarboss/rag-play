import os
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
import re

# --- CONFIG ---
INDEX_NAME = "prebuilt-demo"

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

# --- INIT ---
pinecone = Pinecone(api_key=PINECONE_API_KEY)
index = pinecone.Index(INDEX_NAME)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

def process_text_to_sentence(text):
    """Process text to extract a complete sentence"""
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Find sentence boundaries (period, exclamation, question mark followed by space or end)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # If we have complete sentences, return the first one
    if sentences and sentences[0].strip():
        first_sentence = sentences[0].strip()
        # Ensure it ends with proper punctuation
        if not first_sentence.endswith(('.', '!', '?')):
            first_sentence += '.'
        return first_sentence
    
    # If no clear sentence boundary, try to find a reasonable cutoff
    # Look for the first period, exclamation, or question mark
    match = re.search(r'[^.!?]*[.!?]', text)
    if match:
        return match.group(0).strip()
    
    # If still no good cutoff, return the first 200 characters with ellipsis
    if len(text) > 200:
        return text[:200].strip() + "..."
    
    return text

# --- Streamlit UI ---
st.set_page_config(page_title="Prebuilt Demo", layout="centered")
st.title("Pellar - Prebuilt Demo")

st.info("Connected to prebuilt-demo index. Ask questions about the embedded documents!")

# Query input
query = st.text_input("Ask a question about the documents", placeholder="e.g., What is the main topic?")

if query:
    with st.spinner("Searching for answer..."):
        query_vec = embeddings.embed_query(query)
        results = index.query(vector=query_vec, top_k=1, include_metadata=True)
        
        if results['matches']:
            best_match = results['matches'][0]
            score = best_match['score']
            raw_text = best_match['metadata']['text']
            
            # Process text to keep it as a sentence
            processed_text = process_text_to_sentence(raw_text)
            
            # Display the best match in a nice format
            st.markdown("### Best Match")
            st.markdown(f"**Relevance Score:** {score:.3f}")
            st.markdown("**Relevant Text:**")
            st.markdown(f"> {processed_text}")
            
            # Add some styling
            if score > 0.8:
                st.success("High confidence match found!")
            elif score > 0.6:
                st.warning("Moderate confidence match found.")
            else:
                st.error("Low confidence match - consider rephrasing your question.")
        else:
            st.error("No relevant matches found. Try rephrasing your question.")

st.markdown("---")
st.caption("Pellar 2025")
