# app.py
import streamlit as st
from pathlib import Path
from rag import ingest_url, ingest_pdf, answer_query, get_store
import os

st.set_page_config(page_title="Multi-source RAG", layout="wide")
st.title("Multi-source RAG — Ingest URL / PDF / Text → Chroma/Simple → Retrieve → (GROQ) LLM")

st.sidebar.header("Ingest")
url = st.sidebar.text_input("Ingest from URL (paste URL and press Add URL)")
if st.sidebar.button("Add URL"):
    if not url.strip():
        st.sidebar.warning("Enter a URL")
    else:
        with st.spinner("Fetching and ingesting URL..."):
            count = ingest_url(url)
            st.sidebar.success(f"Ingested {count} chunks from URL")

uploaded = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
if st.sidebar.button("Ingest PDF"):
    if not uploaded:
        st.sidebar.warning("Upload a PDF first")
    else:
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        target = data_dir / uploaded.name
        with open(target, "wb") as f:
            f.write(uploaded.getbuffer())
        with st.spinner("Ingesting PDF..."):
            count = ingest_pdf(str(target))
            st.sidebar.success(f"Ingested {count} chunks from PDF")

st.sidebar.markdown("---")
store = get_store()
st.sidebar.write("Indexed documents:", "empty" if store.is_empty() else "has data")

st.header("Query")
query = st.text_input("Enter your question")
k = st.slider("Top K", min_value=1, max_value=10, value=3)
if st.button("Search"):
    if not query.strip():
        st.warning("Enter a query")
    else:
        with st.spinner("Retrieving and generating..."):
            res = answer_query(query, top_k=k)
        st.subheader("Answer")
        st.write(res["answer"])
        st.subheader("Sources (top hits)")
        for score, meta in res["hits"]:
            st.markdown(f"- **score**: {score:.4f} — source: `{meta.get('source')}`")
            st.write(meta.get("text")[:1000])
