# utils.py
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from pypdf import PdfReader

def fetch_url_text(url: str, timeout=10):
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "rag-bot/1.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    # naive text extraction: body text
    for script in soup(["script","style","noscript"]):
        script.extract()
    text = soup.get_text(separator="\n")
    # collapse whitespace
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines)

def extract_text_from_pdf(path: str):
    p = Path(path)
    if not p.exists():
        return ""
    reader = PdfReader(str(p))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)

def chunk_text(text: str, chunk_size: int=500, chunk_overlap: int=80):
    words = text.split()
    if len(words) == 0:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - chunk_overlap
    return chunks
