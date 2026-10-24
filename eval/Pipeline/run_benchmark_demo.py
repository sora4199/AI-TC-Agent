"""
Demo đầy đủ các tính năng "code được" trong yêu cầu:
  - Validate dataset + coverage check (chuẩn hóa format)
  - Chạy benchmark so sánh nhiều config/baseline
  - Tổng hợp bảng + vẽ plot (phục vụ báo cáo/paper)
  - Error analysis (tìm failure case)

Chạy: python3 run_benchmark_demo.py
"""

from eval.dataset_loader import load_dataset
from eval.dataset_manager import validate_dataset, coverage_report
from eval.evaluators import EVALUATOR_REGISTRY
from eval.benchmark import run_benchmark, compare_configs
from eval.report import comparison_table_markdown, comparison_table_csv, plot_comparison_bar_chart
from eval.error_analysis import find_low_score_cases, save_error_report


# ---- MOCK predict_fn cho 2 config khác nhau của Retriever - THAY BẰNG MODULE THẬT ----

def retriever_v1_bm25(input_data):
    """Giả lập baseline BM25: kết quả kém hơn."""
    return {"documents": [{"id": "doc_999", "content": "...", "metadata": {}}], "metadata": {}}


def retriever_v2_embedding(input_data):
    """Giả lập config mới (embedding search): kết quả tốt hơn."""
    return {
        "documents": [
            {"id": "doc_001", "content": "...", "metadata": {}},
            {"id": "doc_004", "content": "...", "metadata": {}},
        ],
        "metadata": {},
    }


def main():
    module_name = "retriever"
    data_path = "data/retriever_sample.json"

    # ---- BƯỚC 1: chuẩn hóa format - validate + coverage ----
    dataset = load_dataset(data_path)
    errors = validate_dataset(dataset, module_name)
    coverage = coverage_report(dataset)

    print("=== 1. VALIDATE DATASET ===")
    print(f"Coverage: {coverage['by_case_type']}")
    if coverage["missing_case_types"]:
        print(f"⚠️  Dataset mẫu này đang thiếu case_type: {coverage['missing_case_types']} "
              f"(dataset demo, bạn cần bổ sung khi có dataset thật)")
    if errors:
        print(f"❌ Lỗi format: {errors}")
    else:
        print("✅ Format hợp lệ.")
    print()

    # ---- BƯỚC 2: chạy benchmark so sánh 2 config ----
    print("=== 2. CHẠY BENCHMARK: BM25 vs Embedding ===")
    evaluator = EVALUATOR_REGISTRY[module_name]()
    configs = {
        "bm25_baseline": retriever_v1_bm25,
        "embedding_v2": retriever_v2_embedding,
    }
    benchmark_results = run_benchmark(
        dataset=dataset,
        configs=configs,
        evaluator=evaluator,
        module_name=module_name,
        save_dir="reports",
    )

    comparison = compare_configs(benchmark_results)

    # ---- BƯỚC 3: tổng hợp bảng + plot ----
    print("\n=== 3. BẢNG SO SÁNH (markdown) ===")
    print(comparison_table_markdown(comparison))

    comparison_table_csv(comparison, "reports/retriever_comparison.csv")
    plot_comparison_bar_chart(
        comparison,
        save_path="reports/retriever_comparison.png",
        metrics_to_plot=["recall@5", "precision@5", "ndcg@5", "mrr"],
        title="Retriever: BM25 vs Embedding",
    )
    print("\n✅ Đã lưu: reports/retriever_comparison.csv, reports/retriever_comparison.png")

    # ---- BƯỚC 4: error analysis trên config yếu hơn ----
    print("\n=== 4. ERROR ANALYSIS (config bm25_baseline, metric recall@5 < 0.5) ===")
    bad_report = benchmark_results["bm25_baseline"]
    failures = find_low_score_cases(bad_report, metric_key="recall@5", threshold=0.5)
    print(f"Số case fail: {len(failures)}/{bad_report['num_success']}")
    for f in failures:
        print(f"  - {f['id']} ({f['case_type']}): recall@5 = {f['scores']['recall@5']:.2f}")

    save_error_report(
        bad_report,
        metric_key="recall@5",
        threshold=0.5,
        save_path="reports/retriever_bm25_errors.json",
    )
    print("✅ Đã lưu chi tiết: reports/retriever_bm25_errors.json")


if __name__ == "__main__":
    main()
