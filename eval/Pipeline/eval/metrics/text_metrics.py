"""
Metrics cho module trả về TEXT tự do (Agent, RAG -> field `answer`).
Dùng để so sánh prediction (string) với ground truth (string).
"""

import re
import string

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)

    def white_space_fix(text):
        return ' '.join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(
        remove_articles(
            remove_punc(
                lower(s)
            )
        )
    )


def exact_match_score(prediction, ground_truth):
    """1.0 nếu prediction và ground truth trùng nhau sau khi normalize, ngược lại 0.0."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def token_f1_score(prediction, ground_truth):
    """F1 theo overlap token (giống SQuAD F1), độc lập với thứ tự token."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return float(pred_tokens == gt_tokens)

    common = {}
    for tok in pred_tokens:
        common[tok] = min(pred_tokens.count(tok), gt_tokens.count(tok))
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)

    return 2 * precision * recall / (precision + recall)


def rouge_l_score(prediction, ground_truth):
    """
    Compute ROUGE-L based on Longest Common Subsequence (LCS).
    Returns F1 score.
    """
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return float(pred_tokens == gt_tokens)

    dp = [[0] * (len(gt_tokens) + 1)
          for _ in range(len(pred_tokens) + 1)]

    for i in range(1, len(pred_tokens) + 1):
        for j in range(1, len(gt_tokens) + 1):
            if pred_tokens[i - 1] == gt_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(
                    dp[i - 1][j],
                    dp[i][j - 1]
                )

    lcs = dp[-1][-1]

    precision = lcs / len(pred_tokens)
    recall = lcs / len(gt_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def compute_bleu_scores(prediction, ground_truth):
    """
    Compute BLEU-1, BLEU-2, BLEU-3 and BLEU-4.
    """
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        score = float(pred_tokens == gt_tokens)
        return {
            "bleu1": score,
            "bleu2": score,
            "bleu3": score,
            "bleu4": score,
        }

    reference = [gt_tokens]
    hypothesis = pred_tokens

    smoothing = SmoothingFunction().method1

    return {
        "bleu1": sentence_bleu(reference, hypothesis, weights=(1, 0, 0, 0), smoothing_function=smoothing),
        "bleu2": sentence_bleu(reference, hypothesis, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing),
        "bleu3": sentence_bleu(reference, hypothesis, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smoothing),
        "bleu4": sentence_bleu(reference, hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing),
    }


def evaluate_text(prediction, ground_truth):
    """Gộp toàn bộ text metrics cho 1 cặp (prediction, ground_truth)."""
    bleu = compute_bleu_scores(prediction, ground_truth)
    return {
        "exact_match": exact_match_score(prediction, ground_truth),
        "token_f1": token_f1_score(prediction, ground_truth),
        "rouge_l": rouge_l_score(prediction, ground_truth),
        **bleu,
    }
