import os
import logging
import pickle
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
TOP_K = int(os.getenv("TOP_K", "5"))


def get_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"임베딩 모델 로드: {EMBEDDING_MODEL}")
        return model
    except ImportError:
        raise RuntimeError("pip install sentence-transformers 필요")


def embed_texts(model, texts: List[str]) -> np.ndarray:
    """passage: 접두어 추가 후 임베딩 (e5 비대칭 인코딩)"""
    prefixed = [f"passage: {t}" for t in texts]
    vecs = model.encode(prefixed, show_progress_bar=True, batch_size=32)
    return np.array(vecs, dtype="float32")


def embed_query(model, query: str) -> np.ndarray:
    """query: 접두어 추가 후 임베딩"""
    vec = model.encode([f"query: {query}"])
    return np.array(vec, dtype="float32")


class FAISSVectorStore:
    def __init__(self, chunks: List[dict], model=None):
        self.chunks = chunks
        self.model = model or get_embedder()
        self.index: Optional[faiss.IndexFlatIP] = None
        self._build()

    def _build(self):
        if not self.chunks:
            return
        texts = [c["text"] for c in self.chunks]
        logger.info(f"인덱스 구축 중... ({len(texts)}청크)")
        vecs = embed_texts(self.model, texts)
        faiss.normalize_L2(vecs)
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)
        logger.info("FAISS 인덱스 완료")

    def search(self, query: str, top_k: int = TOP_K) -> List[dict]:
        if not self.index or self.index.ntotal == 0:
            return []
        q = embed_query(self.model, query)
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)
        return results

    def format_context(self, results: List[dict]) -> str:
        if not results:
            return "관련 문서를 찾을 수 없습니다."
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(
                f"[출처 {i}: {r['source']} p.{r['page']} (유사도: {r['score']:.3f})]\n{r['text']}"
            )
        return "\n\n---\n\n".join(parts)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls, path: str, model=None) -> "FAISSVectorStore":
        index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.pkl", "rb") as f:
            chunks = pickle.load(f)
        store = cls.__new__(cls)
        store.chunks = chunks
        store.model = model or get_embedder()
        store.index = index
        return store
