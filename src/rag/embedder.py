from typing import List, Optional
import ollama
import logging

class BaseEmbedder:
    def embed_documents(self, texts: List[str]) -> List[Optional[List[float]]]:
        raise NotImplementedError
    def embed_query(self, query: str) -> List[float]:
        raise NotImplementedError
    def get_dimension(self) -> int:
        raise NotImplementedError

class OllamaEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "nomic-embed-text", dimension: int = 768):
        self.model_name = model_name
        self.dimension = dimension
        self.logger = logging.getLogger(__name__)

    def embed_documents(self, texts: List[str]) -> List[Optional[List[float]]]:
        embeddings = []
        for text in texts:
            try:
                res = ollama.embeddings(model=self.model_name, prompt=text)
                emb = res["embedding"]
                if len(emb) != self.dimension:
                    self.logger.warning(f"Dimension mismatch: expected {self.dimension}, got {len(emb)}")
                    embeddings.append(None)
                else:
                    embeddings.append(emb)
            except Exception as e:
                self.logger.error(f"Embedding failed: {e}")
                embeddings.append(None) 
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        res = ollama.embeddings(model=self.model_name, prompt=query)
        return res["embedding"]

    def get_dimension(self) -> int:
        return self.dimension