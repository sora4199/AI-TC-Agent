import json
import time

# 1. Import đủ 5 Baselines
from no_router import NoRouter
from rule_router import RuleBasedRouter
from similarity_router import SimilarityRouter
from llm_router import LLMRouter
from classifier_router import ClassifierRouter

# 2. Đọc file dataset TRƯỚC để có biến `dataset`
with open("dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# 3. Hàm đánh giá chung
def evaluate(router, name):
    total = len(dataset)
    correct = 0
    false_accept = 0
    false_reject = 0
    latencies = []
    failures = []

    for item in dataset:
        q = item["query"]
        expected = item["label"]
        is_ood_expected = item["is_ood"]

        start = time.perf_counter()
        pred = router.route(q)
        lat = (time.perf_counter() - start) * 1000
        latencies.append(lat)

        pred_route = pred.get("route", "")
        pred_ood = pred.get("is_ood", False)

        if pred_route == expected:
            correct += 1
        else:
            failures.append((q, pred_route, expected))

        if is_ood_expected and not pred_ood:
            false_accept += 1
        if not is_ood_expected and pred_ood:
            false_reject += 1

    acc = (correct / total) * 100
    avg_lat = sum(latencies) / total

    print(f"\n[{name}]")
    print(f"  - Accuracy     : {acc:5.1f}% ({correct}/{total})")
    print(f"  - Avg Latency  : {avg_lat:7.2f} ms")
    print(f"  - False Accept : {false_accept} (Lọt OOD)")
    print(f"  - False Reject : {false_reject} (Từ chối nhầm)")
    
    if failures:
        print("  - Dự đoán sai:")
        for q, p, e in failures:
            print(f"    * '{q}' -> Đoán: [{p}] | Đúng: [{e}]")

# 4. Khởi tạo và chạy đánh giá cả 5 Baselines
print("=" * 80)
evaluate(NoRouter(), "Baseline 1: No Router")
evaluate(RuleBasedRouter(), "Baseline 2: Rule-based")
evaluate(SimilarityRouter(), "Baseline 3: Similarity (TF-IDF)")
evaluate(LLMRouter(), "Baseline 4: LLM-based (Qwen2.5:7b)")

# Khởi tạo ClassifierRouter (sau khi dataset đã load xong)
classifier_router = ClassifierRouter(training_data=dataset, ood_threshold=0.35)
evaluate(classifier_router, "Baseline 5: Classifier (TF-IDF + LogReg)")
print("=" * 80)