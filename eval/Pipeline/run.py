"""
Tích hợp Evaluation Pipeline với RAG project THẬT (retriever.py + generator.py
bạn đã gửi, nằm trong thư mục project/).

Chạy: python3 run_project_eval.py

Yêu cầu trước khi chạy:
    - pip install faiss-cpu sentence-transformers ollama --break-system-packages
    - Ollama đang chạy (`ollama serve`) và đã pull model dùng trong generator.py
      (mặc định "qwen2.5:3b" -> `ollama pull qwen2.5:3b`)
    - Sửa INDEX_PATH / CONTEXTS_PATH bên dưới trỏ đúng tới file
      virtual_database/index.faiss và contexts.json THẬT của bạn (retriever.py
      gốc tự suy ra đường dẫn từ vị trí file, nhưng ở đây ta truyền thẳng để
      tránh phụ thuộc cấu trúc thư mục gốc của bạn).

LƯU Ý QUAN TRỌNG:
    Script này gọi thẳng Retriever + OllamaGenerator, KHÔNG dùng class
    RAGPipeline đầy đủ trong pipeline.py, vì RAGPipeline.__init__ còn cần
    loader/chunker/embedder/vector_store (chỉ phục vụ việc ingest, không
    cần thiết để evaluate query -> answer). Logic predict_fn_rag() bên dưới
    tái hiện đúng phần retriever.retrieve() -> generator.generate() ->
    normalize sources giống hệt RAGPipeline.query() trong pipeline.py.

    Nếu sau này bạn muốn evaluate luôn cả bước ingest, cần thêm
    loader.py/chunker.py/embedder.py/vector_store.py vào project/ và import
    trực tiếp class RAGPipeline thay vì ghép thủ công như dưới đây.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from project.retriever import Retriever
from project.generator import OllamaGenerator

from eval.dataset_loader import load_dataset
from eval.evaluators import EVALUATOR_REGISTRY
from eval.pipeline import run_pipeline, print_summary

# ------------------- CẤU HÌNH - CHỈNH LẠI CHO ĐÚNG PROJECT CỦA BẠN -------------------
INDEX_PATH = "virtual_database/index.faiss"          # <-- sửa đường dẫn thật
CONTEXTS_PATH = "virtual_database/contexts.json"      # <-- sửa đường dẫn thật
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "qwen2.5:3b"
# ---------------------------------------------------------------------------------------

retriever = Retriever(index_path=INDEX_PATH, contexts_path=CONTEXTS_PATH, model_name=EMBED_MODEL)
generator = OllamaGenerator(model_name=GEN_MODEL)


def predict_fn_retriever(input_data):
    """
    input_data (theo data/retriever_sample.json): {"query": str, "top_k": int}
    retriever.retrieve() trả về đúng sẵn schema RetrievalResult
    ({"documents": [{"id","content","metadata"}], "metadata": {...}})
    nên không cần map lại gì thêm.
    """
    query = input_data["query"]
    top_k = input_data.get("top_k", 5)
    return retriever.retrieve(query, top_k=top_k)


def predict_fn_rag(input_data):
    """
    input_data (theo data/rag_sample.json): {"query", "session_id", "user_id", "metadata"}
    Ghép retriever + generator lại, tương đương RAGPipeline.query() nhưng
    không cần khởi tạo loader/chunker/embedder/vector_store.
    """
    query = input_data["query"]
    retrieval_res = retriever.retrieve(query, top_k=5)

    if retrieval_res.get("metadata", {}).get("status") != "SUCCESS":
        return {"answer": "", "sources": [], "metadata": {"status": retrieval_res.get("metadata", {}).get("status")}}

    gen_result = generator.generate(query, retrieval_res["documents"])

    if gen_result["status"] != "SUCCESS":
        return {"answer": "", "sources": [], "metadata": {"status": gen_result["status"], "error": gen_result.get("error")}}

    sources = [
        {
            "document_id": c.get("chunk_id") or c.get("source", "unknown"),
            "content": c.get("content", ""),
            "metadata": {"source": c.get("source", "")},
        }
        for c in gen_result.get("citations", [])
    ]

    return {"answer": gen_result["answer"], "sources": sources, "metadata": {"status": "SUCCESS"}}


def main():
    jobs = [
        ("retriever", "data/retriever_sample.json", predict_fn_retriever, "reports/retriever_project_report.json"),
        ("rag", "data/rag_sample.json", predict_fn_rag, "reports/rag_project_report.json"),
    ]

    for module_name, data_path, predict_fn, save_path in jobs:
        dataset = load_dataset(data_path)
        evaluator = EVALUATOR_REGISTRY[module_name]()
        report = run_pipeline(
            dataset=dataset,
            predict_fn=predict_fn,
            evaluator=evaluator,
            module_name=module_name,
            save_path=save_path,
        )
        print_summary(report)
        print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        print("Kiểm tra: đường dẫn INDEX_PATH/CONTEXTS_PATH đã đúng chưa? Ollama đã chạy và pull model chưa?")
