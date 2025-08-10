# ingest.py
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
from vector_store import SimpleVectorStore
import argparse

# simple splitter
def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i: i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - chunk_overlap
    return chunks

def extract_text_from_pdf(pdf_path: Path):
    reader = PdfReader(str(pdf_path))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)

def main(pdf_path: str = "data/sample.pdf", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"{pdf_path} not found. Put your PDF at that path.")

    print("Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        raise ValueError("No extractable text found in PDF. It may be scanned images (requires OCR).")

    print("Splitting into chunks...")
    chunks = split_text(text, chunk_size=500, chunk_overlap=80)
    print(f"Got {len(chunks)} chunks.")

    print("Loading embedding model...")
    model = SentenceTransformer(model_name)
    print("Computing embeddings (this may take a moment)...")
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

    metadatas = [{"text": chunks[i], "source": str(pdf_path), "chunk_id": i} for i in range(len(chunks))]

    store = SimpleVectorStore(path="vectorstore")
    store.add(list(embeddings), metadatas)
    store.save()
    print("Saved vector store to ./vectorstore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default="data/sample.pdf", help="Path to PDF file")
    args = parser.parse_args()
    main(args.pdf)
