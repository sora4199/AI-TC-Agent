"""
Metrics cho module Router (Query -> RouteDecision.route).
Đây là bài toán classification, ground truth là 1 label đúng (VD: "rag").
"""

from collections import defaultdict


def accuracy(predictions, ground_truths):
    """predictions, ground_truths: List[str] cùng độ dài."""
    if not predictions:
        return 0.0
    correct = sum(p == g for p, g in zip(predictions, ground_truths))
    return correct / len(predictions)


def confusion_matrix(predictions, ground_truths):
    """Trả về dict[gt_label][pred_label] = count."""
    matrix = defaultdict(lambda: defaultdict(int))
    for p, g in zip(predictions, ground_truths):
        matrix[g][p] += 1
    return {gt: dict(preds) for gt, preds in matrix.items()}


def precision_recall_f1_per_class(predictions, ground_truths):
    """Precision/Recall/F1 cho từng class (route)."""
    labels = set(predictions) | set(ground_truths)
    result = {}

    for label in labels:
        tp = sum(1 for p, g in zip(predictions, ground_truths) if p == label and g == label)
        fp = sum(1 for p, g in zip(predictions, ground_truths) if p == label and g != label)
        fn = sum(1 for p, g in zip(predictions, ground_truths) if p != label and g == label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        result[label] = {"precision": precision, "recall": recall, "f1": f1}

    return result


def evaluate_router(predictions, ground_truths):
    """
    predictions, ground_truths: List[str] (route labels, VD "rag"/"agent"/"ood")
    Gộp accuracy tổng + breakdown theo từng class.
    """
    return {
        "accuracy": accuracy(predictions, ground_truths),
        "per_class": precision_recall_f1_per_class(predictions, ground_truths),
        "confusion_matrix": confusion_matrix(predictions, ground_truths),
    }
