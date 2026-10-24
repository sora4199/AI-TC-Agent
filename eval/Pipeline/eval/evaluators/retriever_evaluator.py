from ..metrics.retrieval_metrics import evaluate_retrieval
from .base import BaseEvaluator


class RetrieverEvaluator(BaseEvaluator):
    """
    prediction: dict theo schema `RetrievalResult` -> {"documents": [{"id": ..., ...}, ...]}
    ground_truth: dict -> {"relevant_doc_ids": [...]}
    """
    module_name = "retriever"

    def __init__(self, k_list=(1, 3, 5, 10)):
        self.k_list = k_list

    def evaluate_case(self, prediction, ground_truth):
        retrieved_ids = [doc["id"] for doc in prediction.get("documents", [])]
        relevant_ids = ground_truth.get("relevant_doc_ids", [])
        return evaluate_retrieval(retrieved_ids, relevant_ids, k_list=self.k_list)
