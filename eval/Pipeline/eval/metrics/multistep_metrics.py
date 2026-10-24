"""
Metrics cho module Agent / Multi-step Task: agent có thực hiện đúng
CHUỖI bước (plan) và ra đáp án cuối đúng hay không.

prediction / ground_truth format:
    steps: [{"action": str}, ...]   (danh sách tên action theo thứ tự thực hiện)
    answer: str                     (đáp án cuối cùng, optional)
"""

from .text_metrics import evaluate_text


def step_sequence_score(pred_steps, gt_steps):
    """
    LCS-based similarity giữa 2 chuỗi action (giống ROUGE-L nhưng cho step thay vì token).
    1.0 = thực hiện đúng thứ tự các bước cần thiết, dù có thể thêm/thiếu bước khác.
    """
    pred_actions = [s.get("action") for s in pred_steps]
    gt_actions = [s.get("action") for s in gt_steps]

    if not pred_actions or not gt_actions:
        return float(pred_actions == gt_actions)

    dp = [[0] * (len(gt_actions) + 1) for _ in range(len(pred_actions) + 1)]
    for i in range(1, len(pred_actions) + 1):
        for j in range(1, len(gt_actions) + 1):
            if pred_actions[i - 1] == gt_actions[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]

    precision = lcs / len(pred_actions)
    recall = lcs / len(gt_actions)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def step_count_diff(pred_steps, gt_steps):
    """Chênh lệch tuyệt đối số bước thực hiện so với kế hoạch chuẩn."""
    return abs(len(pred_steps) - len(gt_steps))


def evaluate_multistep(prediction, ground_truth):
    """
    prediction / ground_truth: dict {"steps": [...], "answer": str}
    """
    pred_steps = prediction.get("steps", [])
    gt_steps = ground_truth.get("steps", [])

    scores = {
        "step_sequence_score": step_sequence_score(pred_steps, gt_steps),
        "step_count_diff": step_count_diff(pred_steps, gt_steps),
    }

    if "answer" in ground_truth:
        text_scores = evaluate_text(prediction.get("answer", ""), ground_truth.get("answer", ""))
        scores["task_success_em"] = text_scores["exact_match"]
        scores["task_success_f1"] = text_scores["token_f1"]

    return scores
