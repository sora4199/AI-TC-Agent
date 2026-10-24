import os
import uuid
import hashlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Document:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if "document_id" not in self.metadata:
            self.metadata["document_id"] = str(uuid.uuid4())
        if "content_hash" not in self.metadata:
            self.metadata["content_hash"] = hashlib.sha256(self.content.encode('utf-8')).hexdigest()

class BaseLoader:
    def load(self, file_path: str, extra_metadata: Dict[str, Any] = None) -> List[Document]:
        raise NotImplementedError

class TextLoader(BaseLoader):
    def load(self, file_path: str, extra_metadata: Dict[str, Any] = None) -> List[Document]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        if not content:
            return []
            
        metadata = {
            "source": os.path.basename(file_path),
            "extension": ".txt",
            "ingested_at": datetime.now().isoformat()
        }
        if extra_metadata:
            metadata.update(extra_metadata)
            
        return [Document(content=content, metadata=metadata)]

class PDFLoader(BaseLoader):
    def load(self, file_path: str, extra_metadata: Dict[str, Any] = None) -> List[Document]:
        try:
            import PyPDF2
        except ImportError:
            raise ImportError("Vui lòng cài đặt thư viện: pip install PyPDF2")
            
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        text_content = ""
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_content += text + "\n"
                    
        if not text_content.strip():
            return []
            
        metadata = {
            "source": os.path.basename(file_path),
            "extension": ".pdf",
            "ingested_at": datetime.now().isoformat()
        }
        if extra_metadata:
            metadata.update(extra_metadata)
            
        return [Document(content=text_content.strip(), metadata=metadata)]

class UniversalLoader(BaseLoader):
    def load(self, file_path: str, extra_metadata: Dict[str, Any] = None) -> List[Document]:
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.txt':
            return TextLoader().load(file_path, extra_metadata)
        elif ext == '.pdf':
            return PDFLoader().load(file_path, extra_metadata)
        else:
            raise ValueError(f"Chưa hỗ trợ định dạng file: {ext}")