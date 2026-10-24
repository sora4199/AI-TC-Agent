import json
import time
from sklearn.metrics import classification_report
from no_router import NoRouter
from rule_router import RuleBasedRouter
from similarity_router import SimilarityRouter
from llm_router import LLMRouter
from classifier_router import ClassifierRouter

with open("dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

labels_order = ["knowledge_qa", "product_search", "multi_step", "ambiguous", "ood"]

def evaluate(router, name):
    y_true = []
    y_pred = []
    latencies = []
    false_accept = 0
    false_reject = 0

    for item in dataset:
        q = item["query"]
        expected = item["label"]
        is_ood_expected = item["is_ood"]

        start = time.perf_counter()
        pred = router.route(q)
        lat = (time.perf_counter() - start) * 1000
        latencies.append(lat)

        p_route = pred.get("route", "ood").lower()
        p_ood = pred.get("is_ood", (p_route == "ood"))

        y_true.append(expected)
        y_pred.append(p_route)

        if is_ood_expected and not p_ood:
            false_accept += 1
        if not is_ood_expected and p_ood:
            false_reject += 1

    avg_lat = sum(latencies) / len(dataset)
    print(f"\n{'='*75}\n[{name}] - Latency: {avg_lat:.2f} ms | Lọt OOD: {false_accept} | Chặn nhầm: {false_reject}")
    print(classification_report(y_true, y_pred, labels=labels_order, zero_division=0, digits=2))

evaluate(NoRouter(), "Baseline 1: No Router")
evaluate(RuleBasedRouter(), "Baseline 2: Rule-based")
evaluate(SimilarityRouter(), "Baseline 3: Similarity (TF-IDF)")
evaluate(LLMRouter(), "Baseline 4: LLM-based (Qwen2.5:7b)")

clf = ClassifierRouter(training_data=dataset, ood_threshold=0.35)
evaluate(clf, "Baseline 5: Classifier (TF-IDF + LogReg)")
import json
import time
from sklearn.metrics import classification_report
from no_router import NoRouter
from rule_router import RuleBasedRouter
from similarity_router import SimilarityRouter
from llm_router import LLMRouter
from classifier_router import ClassifierRouter

with open("dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

labels_order = ["knowledge_qa", "product_search", "multi_step", "ambiguous", "ood"]

def evaluate(router, name):
    y_true = []
    y_pred = []
    latencies = []
    false_accept = 0
    false_reject = 0

    for item in dataset:
        q = item["query"]
        expected = item["label"]
        is_ood_expected = item["is_ood"]

        start = time.perf_counter()
        pred = router.route(q)
        lat = (time.perf_counter() - start) * 1000
        latencies.append(lat)

        p_route = pred.get("route", "ood").lower()
        p_ood = pred.get("is_ood", (p_route == "ood"))

        y_true.append(expected)
        y_pred.append(p_route)

        if is_ood_expected and not p_ood:
            false_accept += 1
        if not is_ood_expected and p_ood:
            false_reject += 1

    avg_lat = sum(latencies) / len(dataset)
    print(f"\n{'='*75}\n[{name}] - Latency: {avg_lat:.2f} ms | Lọt OOD: {false_accept} | Chặn nhầm: {false_reject}")
    print(classification_report(y_true, y_pred, labels=labels_order, zero_division=0, digits=2))

evaluate(NoRouter(), "Baseline 1: No Router")
evaluate(RuleBasedRouter(), "Baseline 2: Rule-based")
evaluate(SimilarityRouter(), "Baseline 3: Similarity (TF-IDF)")
evaluate(LLMRouter(), "Baseline 4: LLM-based (Qwen2.5:7b)")

clf = ClassifierRouter(training_data=dataset, ood_threshold=0.35)
evaluate(clf, "Baseline 5: Classifier (TF-IDF + LogReg)")
