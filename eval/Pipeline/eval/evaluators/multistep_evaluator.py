from ..metrics.multistep_metrics import evaluate_multistep
from .base import BaseEvaluator


class MultiStepEvaluator(BaseEvaluator):
    """
    prediction: {"steps": [{"action": str}, ...], "answer": str}
    ground_truth: {"steps": [{"action": str}, ...], "answer": str}
    """
    module_name = "multistep"

    def evaluate_case(self, prediction, ground_truth):
        return evaluate_multistep(prediction, ground_truth)
