from typing import List
import uuid
from .loader import Document

class BaseChunker:
    def chunk(self, documents: List[Document]) -> List[Document]:
        raise NotImplementedError

class ParagraphChunker(BaseChunker):
    def __init__(self, max_chunk_size: int = 800, fallback_char_overlap: int = 100):
        if max_chunk_size <= 0 or fallback_char_overlap < 0 or fallback_char_overlap >= max_chunk_size:
            raise ValueError("Invalid chunk_size or overlap configuration.")
        self.max_chunk_size = max_chunk_size
        self.fallback_char_overlap = fallback_char_overlap

    def chunk(self, documents: List[Document]) -> List[Document]:
        chunked_docs = []
        for doc in documents:
            paragraphs = [p for p in doc.content.split('\n\n') if p.strip()]
            current_chunk = ""
            
            for para in paragraphs:
                if len(current_chunk) + len(para) + 2 <= self.max_chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk:
                        self._append_chunk(chunked_docs, doc, current_chunk.strip())
                    
                    if len(para) > self.max_chunk_size:
                        self._fallback_char_split(chunked_docs, doc, para)
                        current_chunk = ""
                    else:
                        current_chunk = para + "\n\n"
            
            if current_chunk:
                self._append_chunk(chunked_docs, doc, current_chunk.strip())
                
        return chunked_docs

    def _append_chunk(self, chunked_docs: List[Document], original_doc: Document, text: str):
        meta = original_doc.metadata.copy()
        meta["chunk_id"] = f"{meta.get('document_id')}_chunk_{str(uuid.uuid4())[:8]}"
        chunked_docs.append(Document(content=text, metadata=meta))

    def _fallback_char_split(self, chunked_docs: List[Document], original_doc: Document, text: str):
        start = 0
        while start < len(text):
            end = start + self.max_chunk_size
            self._append_chunk(chunked_docs, original_doc, text[start:end])
            start += (self.max_chunk_size - self.fallback_char_overlap)