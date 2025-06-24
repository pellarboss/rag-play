import os
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
query = st.text_input("Ask a question about the document")

if uploaded_file:
    raw_text = uploaded_file.read().decode("utf-8")
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

    if query:
        with st.spinner("Generating answer..."):
            query_vec = embeddings.embed_query(query)
            results = index.query(vector=query_vec, top_k=3, include_metadata=True)
            for match in results['matches']:
                st.markdown(f"**Score:** {match['score']:.3f}")
                st.markdown(f"**Text:** {match['metadata']['text']}")
                st.markdown("---")

st.markdown("---")
#st.markdown("Pellar 2025")
