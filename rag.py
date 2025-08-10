# rag.py
import os
import json
from pathlib import Path
from dotenv import load_dotenv
import numpy as np

from utils import fetch_url_text, extract_text_from_pdf, chunk_text
from vector_store import get_vectorstore

load_dotenv()

CONFIG = {
    "EMBED_MODEL": os.getenv("SAMPLE_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    "EMBED_BACKEND": os.getenv("EMBEDDING_BACKEND", "hf"),   # 'hf' or 'openai'
    "VECTORSTORE": os.getenv("VECTORSTORE", "simple"),      # 'chroma' or 'simple'
    "CHROMA_PERSIST_DIR": os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
    "LLM_BACKEND": os.getenv("LLM_BACKEND", "none"),        # 'groq' / 'openai' / 'none'
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
    "GROQ_API_KEY": os.getenv("GROQ_API_KEY", ""),
    "GROQ_MODEL": os.getenv("GROQ_MODEL", "mixtral-8x7b-8192"),
}

# ---------- Embedding layer (lazy) ----------
def get_embedder():
    if CONFIG["EMBED_BACKEND"] == "openai":
        try:
            import openai
            openai.api_key = CONFIG["OPENAI_API_KEY"]
        except Exception as e:
            raise RuntimeError("OpenAI embedding backend selected but openai package not available") from e

        def embed_texts(texts):
            resp = openai.Embedding.create(model="text-embedding-3-small", input=texts)
            return [np.array(d["embedding"], dtype=np.float32) for d in resp["data"]]
        return embed_texts

    else:
        # local sentence-transformers
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(CONFIG["EMBED_MODEL"])
        def embed_texts(texts):
            arr = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return [np.asarray(x, dtype=np.float32) for x in arr]
        return embed_texts

# ---------- Vector store accessor ----------
def get_store():
    if CONFIG["VECTORSTORE"] == "chroma":
        return get_vectorstore("chroma", collection_name="rag_collection", persist_directory=CONFIG["CHROMA_PERSIST_DIR"])
    return get_vectorstore("simple", path="vectorstore")

# ---------- Simple retriever wrapper for LangChain compatibility ----------
class SimpleRetriever:
    """
    A minimal retriever wrapping our SimpleVectorStore so LangChain chains can use it.
    It implements get_relevant_documents(query: str) -> List[Document].
    """
    def __init__(self, store, embed_fn, top_k=5):
        self.store = store
        self.embed_fn = embed_fn
        self.top_k = top_k

    def get_relevant_documents(self, query: str):
        # Lazy import Document to avoid module-level heavy imports
        try:
            from langchain.schema import Document
        except Exception:
            # If langchain not available, create a tiny fallback Document-like object
            class Document:
                def __init__(self, page_content, metadata=None):
                    self.page_content = page_content
                    self.metadata = metadata or {}
            # assign to name so code below uses this
            globals()["Document"] = Document

        # embed the query
        q_emb = self.embed_fn([query])[0]
        hits = self.store.search(q_emb, top_k=self.top_k)
        docs = []
        for score, meta in hits:
            docs.append(Document(page_content=meta.get("text",""), metadata=meta))
        return docs

# ---------- Ingestion helpers ----------
def ingest_url(url: str, chunk_size=500, chunk_overlap=80):
    text = fetch_url_text(url)
    if not text.strip():
        return 0
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    metas = [{"text": c, "source": url} for c in chunks]
    embed = get_embedder()
    embeddings = embed_texts_safe(embed, chunks)
    store = get_store()
    store.add(embeddings, metas)
    store.save()
    return len(chunks)

def ingest_pdf(path: str, chunk_size=500, chunk_overlap=80):
    text = extract_text_from_pdf(path)
    if not text.strip():
        return 0
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    metas = [{"text": c, "source": str(path)} for c in chunks]
    embed = get_embedder()
    embeddings = embed_texts_safe(embed, chunks)
    store = get_store()
    store.add(embeddings, metas)
    store.save()
    return len(chunks)

# ---------- safe embed wrapper ----------
def embed_texts_safe(embed_fn, texts):
    embs = embed_fn(texts)
    return [np.asarray(e, dtype=np.float32) for e in embs]

# ---------- retrieval ----------
def retrieve(query: str, top_k: int = 3):
    embed = get_embedder()
    q_emb = embed_texts_safe(embed, [query])[0]
    store = get_store()
    results = store.search(q_emb, top_k=top_k)
    return results  # list of (score, metadata)

# ---------- GROQ LLM integration (lazy) ----------
def generate_answer_with_groq(query: str, contexts: list, top_k: int = 3):
    """
    contexts: list of text passages (strings)
    Returns: generated text (string)
    """
    # Lazy import ChatGroq and LangChain chain builder
    try:
        from langchain_groq import ChatGroq
    except Exception as e:
        raise RuntimeError("langchain_groq not installed or import failed. Install langchain_groq to use GROQ.") from e

    try:
        # LangChain chain path: create a retriever compatible with LangChain, then use RetrievalQAWithSourcesChain.
        from langchain.chains import RetrievalQAWithSourcesChain
    except Exception:
        RetrievalQAWithSourcesChain = None

    # Create ChatGroq LLM (only now)
    llm = ChatGroq(model=CONFIG["GROQ_MODEL"], api_key=CONFIG["GROQ_API_KEY"], temperature=0.0)

    # If we can build a LangChain RetrievalQA chain (and have a retriever), do so.
    store = get_store()
    embed_fn = get_embedder()
    retriever = SimpleRetriever(store, embed_fn, top_k=top_k)

    if RetrievalQAWithSourcesChain is not None:
        try:
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
            out = chain.invoke({"question": query})
            # out typically contains 'answer' and 'sources'
            return out.get("answer", "").strip()
        except Exception:
            # fallback to direct prompt
            pass

    # Fallback: build a prompt with contexts and call llm directly
    # Build a concise prompt
    combined_context = "\n\n".join(contexts)
    prompt = (
        "You are a helpful assistant. Use only the context below to answer the question, and cite the sources.\n\n"
        f"CONTEXT:\n{combined_context}\n\nQUESTION:\n{query}\n\nAnswer concisely and mention sources."
    )

    # Many ChatGroq wrappers accept __call__ or generate; try both safely
    try:
        # prefer the LangChain call method if available
        resp = llm.generate([{"role": "user", "content": prompt}])
        # resp structure depends on implementation; try to extract text
        # Attempt 1: resp.generations[0][0].text (LangChain-style)
        try:
            return resp.generations[0][0].text.strip()
        except Exception:
            pass
        # Attempt 2: resp[0].text
        try:
            return str(resp[0]).strip()
        except Exception:
            pass
    except Exception:
        pass

    # Last fallback: plain concatenated context
    return "\n\n".join(contexts)

# ---------- unified answering function ----------
def answer_query(query: str, top_k: int = 3):
    """
    Returns dict: {answer, sources (list), hits (list of (score, meta))}
    """
    hits = retrieve(query, top_k=top_k)
    contexts = [h[1].get("text","") for h in hits]
    sources = [h[1].get("source","") for h in hits]

    backend = CONFIG["LLM_BACKEND"].lower()
    if backend == "groq":
        # call GROQ-enabled generator
        try:
            answer = generate_answer_with_groq(query, contexts, top_k=top_k)
            return {"answer": answer, "sources": sources, "hits": hits}
        except Exception as e:
            # return fallback concatenation with error note
            fallback = "\n\n".join(contexts)
            return {"answer": fallback, "sources": sources, "hits": hits, "error": str(e)}
    elif backend == "openai":
        # We can keep previous openai path if user wants — for brevity do concatenation fallback
        answer = "\n\n".join(contexts)
        return {"answer": answer, "sources": sources, "hits": hits}
    else:
        # LLM disabled: return concatenated contexts
        return {"answer": "\n\n".join(contexts), "sources": sources, "hits": hits}
