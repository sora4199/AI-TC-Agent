"""
Base class cho evaluator của từng module.

Mỗi evaluator con cần định nghĩa 2 hàm:
    - evaluate_case(pred, ground_truth) -> dict các điểm số cho 1 test case
    - (tùy chọn) override aggregate() nếu cần logic tổng hợp riêng
        (mặc định: lấy trung bình cộng của các field số)
"""

from collections import defaultdict


class BaseEvaluator:
    module_name = "base"

    def evaluate_case(self, prediction, ground_truth):
        raise NotImplementedError

    def aggregate(self, per_case_scores):
        """
        per_case_scores: List[dict] (mỗi dict là output của evaluate_case)
        Trả về dict điểm trung bình cho từng metric.
        Field không phải số (VD: confusion_matrix) sẽ bị bỏ qua khi tính trung bình,
        nhưng vẫn có thể merge riêng nếu cần (xem RouterEvaluator).
        """
        if not per_case_scores:
            return {}

        sums = defaultdict(float)
        counts = defaultdict(int)

        for scores in per_case_scores:
            for key, value in scores.items():
                if isinstance(value, (int, float)):
                    sums[key] += value
                    counts[key] += 1

        return {key: sums[key] / counts[key] for key in sums}
