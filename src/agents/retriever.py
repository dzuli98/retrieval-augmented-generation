import os
import pickle
from typing import List

from ..config import Config
from ..indexing import FAISSIndex, RetrievalResult
from ..logger import get_logger
from ..models import Document


class RetrieverAgent:
    def __init__(self, config: Config):
        self.config = config
        self.index = FAISSIndex(config)
        self.documents: List[Document] = []
        self.logger = get_logger()
        self._indexed = False

    def build_index(self, documents: List[Document]) -> None:
        self.documents = documents
        if (
            os.path.exists(self.config.index_path)
            and os.path.exists(self.config.embeddings_path)
            and os.path.exists(self.config.documents_path)
        ):
            self.load_index()
        else:
            texts = [doc.content for doc in documents]
            self.logger.info(f"Building FAISS index for {len(documents)} documents...")
            self.index.build(texts)
            self._indexed = True
            self._save_index()
            self.logger.info("FAISS index ready!")

    def retrieve(self, query: str, top_k: int = None) -> List[Document]:
        if not self._indexed:
            raise RuntimeError("Index not built. Call build_index() first.")

        top_k = top_k or self.config.top_k

        self.logger.info(f"Retrieving top-{top_k} documents for: '{query[:50]}...'")
        results = self.index.search(query, top_k)
        retrieval_results = []
        for idx, score in results:
            doc = self.documents[idx]
            retrieval_results.append(
                RetrievalResult(
                    document=Document(
                        doc_id=doc.doc_id,
                        content=doc.content,
                        label=doc.label,
                        relevance_score=score,
                    ),
                    score=score,
                )
            )
        self.logger.info(f"  Retrieved {len(retrieval_results)} documents:")
        for r in retrieval_results:
            self.logger.info(f"    - {r.document.doc_id}: score={r.score:.3f}")

        return [r.document for r in retrieval_results]

    def _save_index(self) -> None:
        os.makedirs(os.path.dirname(self.config.index_path), exist_ok=True)
        self.index.save()
        with open(self.config.documents_path, "wb") as f:
            pickle.dump(self.documents, f)

    def load_index(self) -> None:
        if (
            os.path.exists(self.config.index_path)
            and os.path.exists(self.config.embeddings_path)
            and os.path.exists(self.config.documents_path)
        ):
            self.index.load()
            with open(self.config.documents_path, "rb") as f:
                self.documents = pickle.load(f)
            self._indexed = True
            self.logger.info("Index loaded from disk.")
        else:
            self.logger.warning("Index files not found. Build index first.")
