"""
Ví dụ THẬT: dùng Ollama làm module Router + RAG, chạy qua eval pipeline.

Yêu cầu trước khi chạy:
    1. Ollama đang chạy: `ollama serve` (hoặc app Ollama đã mở sẵn)
    2. Đã pull model: `ollama pull llama3.1` (hoặc đổi MODEL bên dưới theo model bạn có)
    3. pip install requests --break-system-packages

Chạy: python3 run_with_ollama.py
"""

from eval.dataset_loader import load_dataset
from eval.evaluators import EVALUATOR_REGISTRY
from eval.pipeline import run_pipeline, print_summary
from eval.adapters.ollama_adapter import call_ollama_json, call_ollama_text

MODEL = "llama3.1"  # đổi thành model bạn đã `ollama pull`


# ---------- predict_fn cho Router: cần output JSON có cấu trúc ----------

def predict_fn_router(input_data):
    query = input_data["query"]

    prompt = f"""Bạn là bộ định tuyến (router) cho 1 hệ thống AI agent.
Phân loại câu hỏi sau vào đúng 1 trong 3 route: "rag", "agent", "ood".
- "rag": câu hỏi cần tra cứu thông tin/tài liệu.
- "agent": câu hỏi cần thực hiện hành động, tính toán, gọi tool.
- "ood": câu hỏi ngoài phạm vi hệ thống (out-of-domain).

Câu hỏi: "{query}"

Trả lời CHỈ bằng JSON đúng format:
{{"route": "rag" | "agent" | "ood", "confidence": <số 0-1>, "reason": "<lý do ngắn gọn>"}}
"""

    result = call_ollama_json(prompt, model=MODEL)
    # đảm bảo đủ field theo schema RouteDecision dù model trả thiếu
    return {
        "route": result.get("route", "ood"),
        "confidence": result.get("confidence", 0.0),
        "reason": result.get("reason"),
    }


# ---------- predict_fn cho RAG/Agent: trả lời dạng text tự do ----------

def predict_fn_rag(input_data):
    query = input_data["query"]

    prompt = f"Trả lời ngắn gọn, chính xác câu hỏi sau bằng tiếng Việt:\n{query}"
    answer = call_ollama_text(prompt, model=MODEL)

    return {
        "answer": answer.strip(),
        "sources": [],   # nếu module RAG thật của bạn có retrieval, thêm source_ids vào đây
        "metadata": {"model": MODEL},
    }


def main():
    # --- Router ---
    dataset = load_dataset("data/router_sample.json")
    evaluator = EVALUATOR_REGISTRY["router"]()
    report = run_pipeline(dataset, predict_fn_router, evaluator, module_name="router_ollama",
                           save_path="reports/router_ollama_report.json")
    print_summary(report)
    print()

    # --- RAG ---
    dataset = load_dataset("data/rag_sample.json")
    evaluator = EVALUATOR_REGISTRY["rag"]()
    report = run_pipeline(dataset, predict_fn_rag, evaluator, module_name="rag_ollama",
                           save_path="reports/rag_ollama_report.json")
    print_summary(report)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Lỗi khi gọi Ollama: {e}")
        print("Kiểm tra: Ollama đã chạy chưa (`ollama serve`)? Model đã pull chưa (`ollama pull llama3.1`)?")
