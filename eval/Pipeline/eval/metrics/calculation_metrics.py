"""
Metrics cho module Calculation: so sánh kết quả số học.
"""


def numeric_exact_match(pred_value, gt_value, tolerance=1e-6):
    """So sánh 2 số, cho phép sai số nhỏ (tolerance) do float rounding."""
    try:
        return float(abs(float(pred_value) - float(gt_value)) <= tolerance)
    except (TypeError, ValueError):
        return 0.0


def relative_error(pred_value, gt_value):
    """Sai số tương đối |pred-gt|/|gt|. Trả về None nếu gt=0 hoặc không parse được số."""
    try:
        pred_value, gt_value = float(pred_value), float(gt_value)
    except (TypeError, ValueError):
        return None
    if gt_value == 0:
        return None if pred_value != 0 else 0.0
    return abs(pred_value - gt_value) / abs(gt_value)


def evaluate_calculation(pred_value, gt_value, tolerance=1e-6):
    rel_err = relative_error(pred_value, gt_value)
    return {
        "exact_match": numeric_exact_match(pred_value, gt_value, tolerance),
        "relative_error": rel_err if rel_err is not None else 1.0,
    }
