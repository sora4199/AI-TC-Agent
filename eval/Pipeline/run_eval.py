"""
Demo chạy evaluation pipeline end-to-end.

QUAN TRỌNG: các hàm predict_fn_* bên dưới là HÀNG GIẢ (mock) chỉ để chứng minh
pipeline chạy được. Khi có module thật, bạn thay predict_fn bằng hàm gọi
module thật của bạn (Router.predict, Retriever.predict, Agent.predict, ...).
predict_fn chỉ cần tuân đúng: nhận `case["input"]` -> trả về output đúng
schema trong MODULE_SCHEMA.md.

Chạy: python run_eval.py
"""

from eval.dataset_loader import load_dataset
from eval.evaluators import EVALUATOR_REGISTRY
from eval.pipeline import run_pipeline, print_summary


# ---------- MOCK predict_fn - THAY BẰNG MODULE THẬT KHI CÓ ----------

def predict_fn_router(input_data):
    query = input_data["query"].lower()
    if any(ch.isdigit() for ch in query) or "tính" in query:
        route = "agent"
    elif "ăn gì" in query:
        route = "ood"
    else:
        route = "rag"
    return {"route": route, "confidence": 0.8, "reason": None}


def predict_fn_retriever(input_data):
    # giả lập: trả về 1 vài doc ngẫu nhiên, không thực sự retrieve
    return {
        "documents": [
            {"id": "doc_001", "content": "...", "metadata": {}},
            {"id": "doc_999", "content": "...", "metadata": {}},
        ],
        "metadata": {"num_results": 2},
    }


def predict_fn_rag(input_data):
    # giả lập: trả lời sai lệch nhẹ so với ground truth để thấy điểm số < 1.0
    return {
        "answer": "Trump",
        "sources": [{"document_id": "doc_022", "content": "...", "metadata": {}}],
        "metadata": {"route": "rag"},
    }


# ---------------------------------------------------------------------


def main():
    jobs = [
        ("router", "data/router_sample.json", predict_fn_router, "reports/router_report.json"),
        ("retriever", "data/retriever_sample.json", predict_fn_retriever, "reports/retriever_report.json"),
        ("rag", "data/rag_sample.json", predict_fn_rag, "reports/rag_report.json"),
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
    main()
