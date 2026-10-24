from typing import Dict, Any, Optional
import logging
from .loader import BaseLoader
from .chunker import BaseChunker
from .embedder import BaseEmbedder
from .vector_store import BaseVectorStore
from .retriever import Retriever
from .generator import BaseGenerator

class RAGPipeline:
    def __init__(self, loader: BaseLoader, chunker: BaseChunker, embedder: BaseEmbedder, 
                 vector_store: BaseVectorStore, retriever: Retriever, generator: BaseGenerator):
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.retriever = retriever
        self.generator = generator
        self.logger = logging.getLogger(__name__)

    def ingest(self, file_path: str, extra_metadata: Dict = None) -> Dict[str, Any]:
        documents = self.loader.load(file_path, extra_metadata)
        if not documents:
            return {"status": "SKIPPED", "msg": "Empty file"}
            
        doc_hash = documents[0].metadata.get("content_hash")
        if self.vector_store.is_hash_processed(doc_hash):
            return {"status": "SKIPPED", "msg": "File already ingested (Idempotency matched)"}

        chunked_docs = self.chunker.chunk(documents)
        texts = [doc.content for doc in chunked_docs]
        raw_embeddings = self.embedder.embed_documents(texts)
        
        valid_docs, valid_embeddings = [], []
        for doc, emb in zip(chunked_docs, raw_embeddings):
            if emb is not None:
                valid_docs.append(doc)
                valid_embeddings.append(emb)
                
        if not valid_docs:
            return {"status": "FAILED", "msg": "All embeddings failed"}
            
        assert len(valid_docs) == len(valid_embeddings), "Fatal mapping error."
        
        self.vector_store.add(valid_docs, valid_embeddings)
        return {"status": "SUCCESS", "chunks_indexed": len(valid_docs)}

    def query(self, query: str, session_id: str = "", user_id: Optional[str] = None, metadata: Dict = None) -> Dict[str, Any]:
        internal_top_k = 5 
        retrieval_res = self.retriever.retrieve(query, top_k=internal_top_k)
        
        status = retrieval_res.get("metadata", {}).get("status")
        if status != "SUCCESS":
            return {
                "answer": "Không tìm thấy dữ liệu liên quan." if status == "NO_RESULTS" else "Hệ thống truy xuất đang gặp sự cố.",
                "sources": [],
                "metadata": {"status": status, "error": retrieval_res.get("metadata", {}).get("error")}
            }
            
        gen_result = self.generator.generate(query, retrieval_res["documents"])
        
        if gen_result["status"] != "SUCCESS":
            return {
                "answer": "Hệ thống đang gặp sự cố khi tổng hợp câu trả lời.",
                "sources": [],
                "metadata": {"status": gen_result["status"], "error": gen_result.get("error")}
            }
        
        normalized_sources = []
        for citation in gen_result.get("citations", []):
            normalized_sources.append({
                "document_id": citation.get("chunk_id") or citation.get("source", "unknown"),
                "content": citation.get("content", ""),
                "metadata": {"source": citation.get("source", "")}
            })
        
        return {
            "answer": gen_result["answer"],
            "sources": normalized_sources,
            "metadata": {"status": "SUCCESS"}
        }