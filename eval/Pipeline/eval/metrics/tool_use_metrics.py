"""
Metrics cho module Tool-use: agent có gọi đúng tool + đúng argument không.

prediction / ground_truth format (list các lệnh gọi tool theo thứ tự):
    tool_calls: [{"tool_name": str, "arguments": dict}, ...]
"""


def tool_selection_accuracy(pred_calls, gt_calls):
    """Tỷ lệ các bước gọi ĐÚNG tên tool, so theo vị trí (thứ tự)."""
    if not gt_calls:
        return 0.0
    n = min(len(pred_calls), len(gt_calls))
    correct = sum(
        1 for i in range(n) if pred_calls[i].get("tool_name") == gt_calls[i].get("tool_name")
    )
    return correct / len(gt_calls)


def argument_match_score(pred_calls, gt_calls):
    """
    Với các bước gọi ĐÚNG tool (theo vị trí), tính % key/value argument khớp.
    Trả về điểm trung bình trên các bước có tool đúng (0 nếu không có bước nào đúng tool).
    """
    n = min(len(pred_calls), len(gt_calls))
    scores = []

    for i in range(n):
        pred, gt = pred_calls[i], gt_calls[i]
        if pred.get("tool_name") != gt.get("tool_name"):
            continue
        gt_args = gt.get("arguments", {})
        pred_args = pred.get("arguments", {})
        if not gt_args:
            scores.append(1.0)
            continue
        matched = sum(1 for k, v in gt_args.items() if pred_args.get(k) == v)
        scores.append(matched / len(gt_args))

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def call_count_match(pred_calls, gt_calls):
    """1.0 nếu số lượng tool call bằng đúng ground truth, ngược lại 0.0."""
    return float(len(pred_calls) == len(gt_calls))


def evaluate_tool_use(pred_calls, gt_calls):
    return {
        "tool_selection_accuracy": tool_selection_accuracy(pred_calls, gt_calls),
        "argument_match": argument_match_score(pred_calls, gt_calls),
        "call_count_match": call_count_match(pred_calls, gt_calls),
    }
