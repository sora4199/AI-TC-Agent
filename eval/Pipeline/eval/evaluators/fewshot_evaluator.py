from ..metrics.retrieval_metrics import evaluate_retrieval
from .base import BaseEvaluator


class FewShotEvaluator(BaseEvaluator):
    """
    Dynamic Few-shot: module chọn ra tập example phù hợp nhất để đưa vào prompt.
    Bản chất giống bài toán retrieval -> tái dùng retrieval metrics.

    prediction: {"selected_example_ids": [str, ...]}   (đã sắp theo thứ tự ưu tiên)
    ground_truth: {"relevant_example_ids": [str, ...]}
    """
    module_name = "fewshot"

    def __init__(self, k_list=(1, 3, 5)):
        self.k_list = k_list

    def evaluate_case(self, prediction, ground_truth):
        selected_ids = prediction.get("selected_example_ids", [])
        relevant_ids = ground_truth.get("relevant_example_ids", [])
        return evaluate_retrieval(selected_ids, relevant_ids, k_list=self.k_list)
