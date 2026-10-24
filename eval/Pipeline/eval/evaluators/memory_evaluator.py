from ..metrics.retrieval_metrics import evaluate_retrieval
from .base import BaseEvaluator


class MemoryEvaluator(BaseEvaluator):
    """
    prediction: dict theo schema `MemoryResult` -> {"memories": [{"id": ..., ...}, ...]}
    ground_truth: dict -> {"relevant_memory_ids": [...]}
    """
    module_name = "memory"

    def __init__(self, k_list=(1, 3, 5)):
        self.k_list = k_list

    def evaluate_case(self, prediction, ground_truth):
        retrieved_ids = [m["id"] for m in prediction.get("memories", [])]
        relevant_ids = ground_truth.get("relevant_memory_ids", [])
        return evaluate_retrieval(retrieved_ids, relevant_ids, k_list=self.k_list)
