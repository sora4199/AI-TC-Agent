
from eval.metrics import (
    rouge_l_score,
    recall_score,
    compute_bleu_scores,
)


def evaluate(gt, pred):
    """
    Args:
        gt: str (ground truth)
        pred: str (prediction)

    Returns:
        rouge_l: float
        recall: float
        bleu1: float
        bleu2: float
        bleu3: float
        bleu4: float
    """

    rouge_l = rouge_l_score(pred, gt)
    recall = recall_score(pred, gt)

    bleu_scores = compute_bleu_scores(pred, gt)

    return (
        rouge_l,
        recall,
        bleu_scores["bleu1"],
        bleu_scores["bleu2"],
        bleu_scores["bleu3"],
        bleu_scores["bleu4"],
    )


if __name__ == "__main__":

    gt = "donal trump"
    pred = "trump"

    (
        rouge_l,
        recall,
        bleu1,
        bleu2,
        bleu3,
        bleu4,
    ) = evaluate(
        gt=gt,
        pred=pred,
    )

    print("=== EVALUATION RESULT ===")
    print(f"Ground truth : {gt}")
    print(f"Prediction   : {pred}")
    print(f"ROUGE-L      : {rouge_l:.4f}")
    print(f"Recall       : {recall:.4f}")
    print(f"BLEU-1       : {bleu1:.4f}")
    print(f"BLEU-2       : {bleu2:.4f}")
    print(f"BLEU-3       : {bleu3:.4f}")
    print(f"BLEU-4       : {bleu4:.4f}")
