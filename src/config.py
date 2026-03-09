import os
from dataclasses import dataclass


@dataclass
class Config:
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    top_k: int = 5

    index_path: str = "data/index.faiss"
    embeddings_path: str = "data/embeddings.npy"
    documents_path: str = "data/documents.pkl"

    verbose: bool = True

    @classmethod
    def from_env(cls, api_key: str = None) -> "Config":
        return cls(openai_api_key=api_key or os.getenv("OPENAI_API_KEY", ""))
