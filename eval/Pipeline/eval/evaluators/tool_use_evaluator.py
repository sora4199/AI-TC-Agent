from ..metrics.tool_use_metrics import evaluate_tool_use
from .base import BaseEvaluator


class ToolUseEvaluator(BaseEvaluator):
    """
    prediction: {"tool_calls": [{"tool_name": str, "arguments": dict}, ...]}
    ground_truth: {"tool_calls": [{"tool_name": str, "arguments": dict}, ...]}
    """
    module_name = "tool_use"

    def evaluate_case(self, prediction, ground_truth):
        pred_calls = prediction.get("tool_calls", [])
        gt_calls = ground_truth.get("tool_calls", [])
        return evaluate_tool_use(pred_calls, gt_calls)
