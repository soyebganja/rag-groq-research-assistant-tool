# vector_store.py
from pathlib import Path
import numpy as np
import json
from typing import List, Dict

class SimpleVectorStore:
    def __init__(self, path: str = "vectorstore"):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.vec_file = self.path / "vectors.npz"
        self.meta_file = self.path / "metadatas.json"
        self._embeddings = []  # list of numpy arrays
        self._metadatas = []   # list of dicts
        if self.vec_file.exists() and self.meta_file.exists():
            self._load()

    def _load(self):
        data = np.load(self.vec_file, allow_pickle=True)
        self._embeddings = [data[f"arr_{i}"] for i in range(len(data.files))]
        with open(self.meta_file, "r", encoding="utf-8") as f:
            self._metadatas = json.load(f)

    def save(self):
        np_dict = {f"arr_{i}": emb for i, emb in enumerate(self._embeddings)}
        np.savez(self.vec_file, **np_dict)
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(self._metadatas, f, ensure_ascii=False, indent=2)

    def add(self, embeddings: List[np.ndarray], metadatas: List[Dict]):
        assert len(embeddings) == len(metadatas)
        for e, m in zip(embeddings, metadatas):
            arr = np.asarray(e, dtype=np.float32).reshape(-1)
            self._embeddings.append(arr)
            self._metadatas.append(m)

    def is_empty(self):
        return len(self._embeddings) == 0

    def _stack(self):
        if not self._embeddings:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack(self._embeddings)

    def search(self, query_emb: np.ndarray, top_k: int = 5):
        if len(self._embeddings) == 0:
            return []
        Q = np.asarray(query_emb, dtype=np.float32).reshape(1, -1)
        M = self._stack()
        dot = (M @ Q.T).reshape(-1)
        q_norm = np.linalg.norm(Q)
        m_norm = np.linalg.norm(M, axis=1)
        denom = (m_norm * q_norm) + 1e-10
        scores = dot / denom
        top_idx = np.argsort(-scores)[:top_k]
        return [(float(scores[i]), self._metadatas[i]) for i in top_idx]

# helper to choose store
def get_vectorstore(kind: str = "simple", **kwargs):
    if kind == "chroma":
        try:
            import chromadb
            from chromadb.utils import embedding_functions
            from chromadb.config import Settings
            from chromadb import Client
            persist_dir = kwargs.get("persist_directory", "./chroma_db")
            client = Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir))
            return ChromaWrapper(client=client, collection_name=kwargs.get("collection_name", "default"), persist_directory=persist_dir)
        except Exception as e:
            print("Chroma not available, falling back to SimpleVectorStore:", e)
            return SimpleVectorStore(path=kwargs.get("path", "vectorstore"))
    else:
        return SimpleVectorStore(path=kwargs.get("path", "vectorstore"))

# Minimal Chroma wrapper (if user installs chromadb)
class ChromaWrapper:
    def __init__(self, client, collection_name="default", persist_directory="./chroma_db"):
        self.client = client
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        # get or create collection
        try:
            self.collection = client.get_collection(name=collection_name)
        except Exception:
            self.collection = client.create_collection(name=collection_name)
    def add(self, embeddings, metadatas):
        ids = [f"doc_{i}" for i in range(len(embeddings))]
        self.collection.add(documents=[m.get("text","") for m in metadatas],
                            metadatas=metadatas,
                            ids=ids,
                            embeddings=[e.tolist() for e in embeddings])
    def is_empty(self):
        return self.collection.count() == 0
    def save(self):
        # chroma persists automatically with the client settings
        pass
    def search(self, query_emb, top_k=5):
        res = self.collection.query(query_embeddings=query_emb.tolist(), n_results=top_k, include=["metadatas","distances"])
        results = []
        for i in range(len(res["metadatas"][0])):
            md = res["metadatas"][0][i]
            dist = res["distances"][0][i]
            # convert distance to similarity-ish score
            score = 1.0 / (1.0 + dist)
            results.append((float(score), md))
        return results
