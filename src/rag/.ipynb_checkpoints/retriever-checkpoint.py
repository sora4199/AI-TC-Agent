import json
import faiss
from typing import Dict, Any
from pathlib import Path
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, index_path: str = None, contexts_path: str = None, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        ROOT_DIR = Path(__file__).resolve().parents[3]
        self.index_path = index_path if index_path else str(ROOT_DIR / "virtual_database" / "index.faiss")
        self.contexts_path = contexts_path if contexts_path else str(ROOT_DIR / "virtual_database" / "contexts.json")
        
        self.index = faiss.read_index(self.index_path)
        with open(self.contexts_path, "r", encoding="utf-8") as f:
            self.contexts = json.load(f)
            
        self.model = SentenceTransformer(model_name)

    def retrieve(self, query: str, top_k: int = 20) -> Dict[str, Any]:
        if not query or not isinstance(query, str):
            return {"documents": [], "metadata": {"status": "ERROR"}}
            
        k = min(top_k, len(self.contexts))
        query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(query_embedding, k)
        
        documents = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0: continue
            documents.append({
                "id": f"doc_{idx:03d}",
                "content": self.contexts[idx],
                "metadata": {"retrieval_score": float(score)}
            })
            
        return {
            "documents": documents,
            "metadata": {"status": "SUCCESS", "num_results": len(documents)}
        }