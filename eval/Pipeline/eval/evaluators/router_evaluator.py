from ..metrics.classification_metrics import evaluate_router
from .base import BaseEvaluator


class RouterEvaluator(BaseEvaluator):
    """
    prediction: dict theo schema `RouteDecision` -> {"route": "rag"|"agent"|"ood", ...}
    ground_truth: dict -> {"route": "rag"|"agent"|"ood"}
    """
    module_name = "router"

    def evaluate_case(self, prediction, ground_truth):
        pred_route = prediction.get("route")
        gt_route = ground_truth.get("route")
        return {
            "correct": float(pred_route == gt_route),
            "predicted_route": pred_route,
            "ground_truth_route": gt_route,
        }

    def aggregate(self, per_case_scores):
        # accuracy/confusion-matrix cần tính trên TOÀN BỘ tập, không phải trung bình từng case
        predictions = [s["predicted_route"] for s in per_case_scores]
        ground_truths = [s["ground_truth_route"] for s in per_case_scores]
        return evaluate_router(predictions, ground_truths)
