"""
Metrics cho module trả về DANH SÁCH item có id (Retriever -> documents,
Memory -> memories). Ground truth là tập id được coi là "đúng"/liên quan.

Input chuẩn cho mọi hàm bên dưới:
    retrieved_ids: List[str]   -> id lấy theo đúng thứ tự trả về (rank 1 trước)
    relevant_ids:  Set[str]    -> tập id đúng (ground truth)
    k: int                     -> cắt top-k trước khi tính (None = lấy hết)
"""

import math


def _cut(retrieved_ids, k):
    if k is None:
        return retrieved_ids
    return retrieved_ids[:k]


def recall_at_k(retrieved_ids, relevant_ids, k=None):
    if not relevant_ids:
        return 0.0
    top = _cut(retrieved_ids, k)
    hit = len(set(top) & set(relevant_ids))
    return hit / len(relevant_ids)


def precision_at_k(retrieved_ids, relevant_ids, k=None):
    top = _cut(retrieved_ids, k)
    if not top:
        return 0.0
    hit = len(set(top) & set(relevant_ids))
    return hit / len(top)


def mrr(retrieved_ids, relevant_ids, k=None):
    """Mean Reciprocal Rank: 1/rank của item liên quan đầu tiên xuất hiện."""
    top = _cut(retrieved_ids, k)
    for idx, doc_id in enumerate(top, start=1):
        if doc_id in relevant_ids:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(retrieved_ids, relevant_ids, k=None):
    """
    nDCG với relevance nhị phân (1 nếu id nằm trong relevant_ids, 0 nếu không).
    Nếu bạn có relevance dạng điểm số (graded), có thể mở rộng sau.
    """
    top = _cut(retrieved_ids, k)
    if not top or not relevant_ids:
        return 0.0

    dcg = 0.0
    for idx, doc_id in enumerate(top, start=1):
        rel = 1.0 if doc_id in relevant_ids else 0.0
        dcg += rel / math.log2(idx + 1)

    ideal_hits = min(len(relevant_ids), len(top))
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_retrieval(retrieved_ids, relevant_ids, k_list=(1, 3, 5, 10)):
    """Gộp toàn bộ metric cho 1 query, tính ở nhiều mức k."""
    relevant_ids = set(relevant_ids)
    result = {"mrr": mrr(retrieved_ids, relevant_ids)}
    for k in k_list:
        result[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
        result[f"precision@{k}"] = precision_at_k(retrieved_ids, relevant_ids, k)
        result[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant_ids, k)
    return result
