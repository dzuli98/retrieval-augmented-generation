import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import faiss
import numpy as np
from openai import OpenAI

from .config import Config
from .models import Document


@dataclass
class RetrievalResult:
    document: Document
    score: float = 0.0


class FAISSIndex:

    def __init__(self, config: Config):
        self.config = config
        self.embeddings: Optional[np.ndarray] = None
        self.index = None
        self._client = None

    def _get_openai_client(self):
        if self._client is None:
            self._client = OpenAI(api_key=self.config.openai_api_key)
        return self._client

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        client = self._get_openai_client()

        all_embeddings = []
        batch_size = 100

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = client.embeddings.create(
                model=self.config.embedding_model, input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings, dtype=np.float32)

    def build(self, documents: List[str]) -> None:
        self.embeddings = self._embed_texts(documents)
        faiss.normalize_L2(self.embeddings)
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        query_embedding = self._embed_texts([query])
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((int(idx), float(score)))
        return results

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.config.index_path), exist_ok=True)
        faiss.write_index(self.index, self.config.index_path)
        np.save(self.config.embeddings_path, self.embeddings)

    def load(self) -> None:
        self.index = faiss.read_index(self.config.index_path)
        self.embeddings = np.load(self.config.embeddings_path)
