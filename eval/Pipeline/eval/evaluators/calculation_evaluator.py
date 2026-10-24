from ..metrics.calculation_metrics import evaluate_calculation
from .base import BaseEvaluator


class CalculationEvaluator(BaseEvaluator):
    """
    prediction: {"result": number}
    ground_truth: {"result": number, "tolerance": float (optional, default 1e-6)}
    """
    module_name = "calculation"

    def evaluate_case(self, prediction, ground_truth):
        tolerance = ground_truth.get("tolerance", 1e-6)
        return evaluate_calculation(
            prediction.get("result"),
            ground_truth.get("result"),
            tolerance=tolerance,
        )
