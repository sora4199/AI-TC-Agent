"""
Evaluation pipeline dùng chung cho mọi module.

Cách dùng:

    from eval.pipeline import run_pipeline
    from eval.dataset_loader import load_dataset
    from eval.evaluators import EVALUATOR_REGISTRY

    dataset = load_dataset("data/retriever_sample.json")
    evaluator = EVALUATOR_REGISTRY["retriever"]()

    report = run_pipeline(
        dataset=dataset,
        predict_fn=my_retriever_module.predict,   # hàm bạn tự viết, gọi module thật
        evaluator=evaluator,
        module_name="retriever",
    )

`predict_fn` PHẢI nhận vào `case["input"]` (dict đúng theo Request schema trong
MODULE_SCHEMA.md) và trả về output đúng Response schema tương ứng.
Đây là chỗ bạn cắm module thật (Router/Retriever/Reranker/Memory/Agent/RAG) vào.
"""

import json
import time
from pathlib import Path


def run_pipeline(dataset, predict_fn, evaluator, module_name, save_path=None):
    per_case_results = []
    errors = []

    for case in dataset:
        try:
            start = time.time()
            prediction = predict_fn(case["input"])
            latency = time.time() - start

            scores = evaluator.evaluate_case(prediction, case["ground_truth"])

            per_case_results.append({
                "id": case["id"],
                "case_type": case.get("case_type", "normal"),
                "scores": scores,
                "latency_sec": latency,
            })
        except Exception as e:
            errors.append({"id": case.get("id"), "error": str(e)})

    # aggregate() của evaluator cần list các dict `scores` (không phải cả record)
    flat_scores = [r["scores"] for r in per_case_results]
    aggregate_scores = evaluator.aggregate(flat_scores)

    # breakdown theo case_type (normal / edge / adversarial) - dùng base aggregate mặc định
    breakdown = {}
    for case_type in {r["case_type"] for r in per_case_results}:
        subset = [r["scores"] for r in per_case_results if r["case_type"] == case_type]
        breakdown[case_type] = evaluator.aggregate(subset)

    report = {
        "module": module_name,
        "num_cases": len(dataset),
        "num_success": len(per_case_results),
        "num_errors": len(errors),
        "aggregate": aggregate_scores,
        "breakdown_by_case_type": breakdown,
        "errors": errors,
        "per_case": per_case_results,
    }

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def print_summary(report):
    print(f"=== EVALUATION REPORT: {report['module']} ===")
    print(f"Tổng số case : {report['num_cases']}")
    print(f"Thành công   : {report['num_success']}")
    print(f"Lỗi          : {report['num_errors']}")
    print("--- Điểm trung bình (toàn bộ) ---")
    for k, v in report["aggregate"].items():
        if isinstance(v, (int, float)):
            print(f"  {k:20s}: {v:.4f}")
        else:
            print(f"  {k:20s}: {v}")
    if report["breakdown_by_case_type"]:
        print("--- Breakdown theo case_type ---")
        for case_type, scores in report["breakdown_by_case_type"].items():
            print(f"  [{case_type}]")
            for k, v in scores.items():
                if isinstance(v, (int, float)):
                    print(f"    {k:18s}: {v:.4f}")
    if report["errors"]:
        print("--- Case bị lỗi ---")
        for err in report["errors"]:
            print(f"  {err['id']}: {err['error']}")
