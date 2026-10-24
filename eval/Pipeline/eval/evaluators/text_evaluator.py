from ..metrics.text_metrics import evaluate_text
from .base import BaseEvaluator


class TextEvaluator(BaseEvaluator):
    """
    Dùng cho module Agent / RAG.
    prediction: dict theo schema `Response` -> {"answer": str, "sources": [...], "metadata": {...}}
    ground_truth: dict -> {"answer": str, "source_ids": [...] (tuỳ chọn)}
    """
    module_name = "text"

    def evaluate_case(self, prediction, ground_truth):
        pred_answer = prediction.get("answer", "")
        gt_answer = ground_truth.get("answer", "")

        scores = evaluate_text(pred_answer, gt_answer)

        # Nếu ground truth có kèm source_ids đúng, đo thêm việc trích dẫn nguồn có khớp không
        gt_source_ids = ground_truth.get("source_ids")
        if gt_source_ids is not None:
            pred_source_ids = {
                s.get("document_id") for s in prediction.get("sources", [])
            }
            gt_source_ids = set(gt_source_ids)
            if gt_source_ids:
                scores["source_recall"] = len(pred_source_ids & gt_source_ids) / len(gt_source_ids)

        return scores
