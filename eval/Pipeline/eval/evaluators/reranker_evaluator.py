from ..metrics.retrieval_metrics import evaluate_retrieval
from .base import BaseEvaluator


class RerankerEvaluator(BaseEvaluator):
    """
    prediction: dict theo schema `RerankResult` ->
        {"documents": [{"document": {"id": ...}, "score": ..., "rank": ...}, ...]}
        (danh sách ĐÃ được sắp theo rank tăng dần)
    ground_truth: dict -> {"relevant_doc_ids": [...]}
    """
    module_name = "reranker"

    def __init__(self, k_list=(1, 3, 5, 10)):
        self.k_list = k_list

    def evaluate_case(self, prediction, ground_truth):
        ranked = sorted(prediction.get("documents", []), key=lambda x: x.get("rank", 0))
        retrieved_ids = [item["document"]["id"] for item in ranked]
        relevant_ids = ground_truth.get("relevant_doc_ids", [])
        return evaluate_retrieval(retrieved_ids, relevant_ids, k_list=self.k_list)
