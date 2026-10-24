from typing import List, Dict, Any
import os
import json
import numpy as np
import faiss
import shutil
from .loader import Document

class BaseVectorStore:
    def add(self, documents: List[Document], embeddings: List[List[float]]):
        raise NotImplementedError
    def search(self, query_embedding: List[float], top_k: int, filters: Dict = None) -> List[Dict[str, Any]]:
        raise NotImplementedError
    def is_hash_processed(self, content_hash: str) -> bool:
        raise NotImplementedError

class FaissVectorStore(BaseVectorStore):
    def __init__(self, dimension: int, persist_dir: str = "./data/faiss_storage"):
        self.dimension = dimension
        self.persist_dir = persist_dir
        self.index_path = os.path.join(persist_dir, "faiss.index")
        self.store_path = os.path.join(persist_dir, "doc_store.json")
        self.doc_store: Dict[int, Dict] = {}
        self.processed_hashes: set = set()
        self.current_id = 0
        
        os.makedirs(persist_dir, exist_ok=True)
        self._load_if_exists()

    def _load_if_exists(self):
        if os.path.exists(self.index_path) and os.path.exists(self.store_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.store_path, 'r', encoding='utf-8') as f:
                store_data = json.load(f)
                self.doc_store = {int(k): v for k, v in store_data.get("docs", {}).items()}
                self.processed_hashes = set(store_data.get("hashes", []))
                self.current_id = max(self.doc_store.keys(), default=-1) + 1
        else:
            self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))

    def is_hash_processed(self, content_hash: str) -> bool:
        return content_hash in self.processed_hashes

    def save(self):
        tmp_index = f"{self.index_path}.tmp"
        tmp_store = f"{self.store_path}.tmp"
        
        faiss.write_index(self.index, tmp_index)
        with open(tmp_store, 'w', encoding='utf-8') as f:
            json.dump({
                "docs": self.doc_store,
                "hashes": list(self.processed_hashes)
            }, f, ensure_ascii=False)
            
        shutil.move(tmp_index, self.index_path)
        shutil.move(tmp_store, self.store_path)

    def add(self, documents: List[Document], embeddings: List[List[float]]):
        if not documents or len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings length mismatch.")
            
        vectors = np.array(embeddings).astype('float32')
        ids = np.arange(self.current_id, self.current_id + len(documents))
        
        self.index.add_with_ids(vectors, ids)
        for i, doc in enumerate(documents):
            self.doc_store[int(ids[i])] = {"content": doc.content, "metadata": doc.metadata}
            hash_val = doc.metadata.get("content_hash")
            if hash_val:
                self.processed_hashes.add(hash_val)
                
        self.current_id += len(documents)
        self.save()

    def search(self, query_embedding: List[float], top_k: int, filters: Dict = None) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []
            
        fetch_k = top_k * 5 if filters else top_k
        fetch_k = min(fetch_k, self.index.ntotal)
        
        query_vector = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_vector, fetch_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue
            
            doc_dict = self.doc_store[int(idx)]
            # Sao chép metadata để tránh can thiệp vào kho dữ liệu gốc
            meta = doc_dict["metadata"].copy()
            
            if filters:
                match = all(meta.get(k) == v for k, v in filters.items())
                if not match: continue
                
            # Đẩy các tham số kỹ thuật vào thẳng metadata
            meta["retrieval_distance"] = float(distances[0][i])
            meta["retrieval_score_metric"] = "L2_distance"
                
            results.append({
                "id": meta.get("chunk_id", str(idx)),
                "content": doc_dict["content"],
                "metadata": meta
            })
            
            if len(results) == top_k:
                break
                
        return results