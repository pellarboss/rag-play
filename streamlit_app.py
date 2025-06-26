import os
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

# --- CONFIG ---
INDEX_NAME = "rag-demo"

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

# --- INIT ---
pinecone = Pinecone(api_key=PINECONE_API_KEY)

if not pinecone.has_index(INDEX_NAME):
    pinecone.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pinecone.Index(INDEX_NAME)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# --- Streamlit UI ---
st.set_page_config(page_title="RAG Demo", layout="centered")
st.title("Pellar - RAG Demo")

uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf"])

# Initialize session state for tracking file changes
if 'processed_file_name' not in st.session_state:
    st.session_state.processed_file_name = None
if 'processed_file_size' not in st.session_state:
    st.session_state.processed_file_size = None

# Check if a file has been processed
file_processed = st.session_state.processed_file_name is not None

if uploaded_file:
    # Check if file has changed
    current_file_name = uploaded_file.name
    current_file_size = uploaded_file.size
    
    file_changed = (
        st.session_state.processed_file_name != current_file_name or
        st.session_state.processed_file_size != current_file_size
    )
    
    if file_changed:
        # Handle different file types
        if uploaded_file.type == "application/pdf":
            with st.spinner("Reading PDF..."):
                pdf_reader = PdfReader(uploaded_file)
                raw_text = ""
                for page in pdf_reader.pages:
                    raw_text += page.extract_text()
            st.success(f"PDF processed successfully! Extracted {len(raw_text)} characters.")
        else:
            raw_text = uploaded_file.read().decode("utf-8")
            st.success(f"Text file processed successfully! Extracted {len(raw_text)} characters.")
        
        with st.spinner("Processing document..."):
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            docs = splitter.create_documents([raw_text])
            doc_texts = [doc.page_content for doc in docs]
            doc_vectors = embeddings.embed_documents(doc_texts)

            # Upsert with metadata
            vectors = [
                (f"doc-{i}", doc_vectors[i], {"text": doc_texts[i]})
                for i in range(len(doc_vectors))
            ]
            index.upsert(vectors=vectors)
            st.success(f"Document processed and indexed! Created {len(docs)} chunks.")
        
        # Update session state to track the processed file
        st.session_state.processed_file_name = current_file_name
        st.session_state.processed_file_size = current_file_size
        file_processed = True
    else:
        st.info(f"Using previously processed file: {current_file_name}")
        file_processed = True

# Query input - only enabled if file is processed
if file_processed:
    query = st.text_input("Ask a question about the document", placeholder="e.g., What is the main topic?")
else:
    query = st.text_input("Ask a question about the document", placeholder="Please upload a document first", disabled=True)

if file_processed and query:
    with st.spinner("Searching for answer..."):
        query_vec = embeddings.embed_query(query)
        results = index.query(vector=query_vec, top_k=1, include_metadata=True)
        
        if results['matches']:
            best_match = results['matches'][0]
            score = best_match['score']
            text = best_match['metadata']['text']
            
            # Display the best match in a nice format
            st.markdown("### Best Match")
            st.markdown(f"**Relevance Score:** {score:.3f}")
            st.markdown("**Relevant Text:**")
            st.markdown(f"> {text}")
            
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
