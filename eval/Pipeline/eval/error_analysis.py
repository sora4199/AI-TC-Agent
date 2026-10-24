"""
Error analysis: từ report của pipeline, tự động tìm ra các test case
có điểm thấp (failure case) để phân tích nguyên nhân.
"""

import json
from pathlib import Path


def find_low_score_cases(report, metric_key, threshold=None, top_n=10, lower_is_better=False):
    """
    Lọc case theo 1 metric cụ thể.

    - Nếu threshold được cung cấp: lấy toàn bộ case có score TỆ HƠN threshold.
    - Nếu không: sắp xếp và lấy top_n case TỆ NHẤT.

    lower_is_better=True dùng cho các metric càng thấp càng tốt (VD: step_count_diff,
    relative_error). Mặc định False (càng cao càng tốt, như accuracy, recall...).
    """
    scored_cases = [
        r for r in report["per_case"]
        if metric_key in r["scores"] and isinstance(r["scores"][metric_key], (int, float))
    ]

    if threshold is not None:
        if lower_is_better:
            filtered = [r for r in scored_cases if r["scores"][metric_key] > threshold]
        else:
            filtered = [r for r in scored_cases if r["scores"][metric_key] < threshold]
        filtered.sort(key=lambda r: r["scores"][metric_key], reverse=lower_is_better)
        return filtered

    scored_cases.sort(key=lambda r: r["scores"][metric_key], reverse=lower_is_better)
    return scored_cases[:top_n] if not lower_is_better else scored_cases[-top_n:]


def error_breakdown_by_case_type(report, metric_key, threshold, lower_is_better=False):
    """Đếm số lượng failure case theo từng case_type (normal/edge/adversarial)."""
    failures = find_low_score_cases(report, metric_key, threshold=threshold, lower_is_better=lower_is_better)
    breakdown = {}
    for r in failures:
        breakdown[r["case_type"]] = breakdown.get(r["case_type"], 0) + 1
    return breakdown


def save_error_report(report, metric_key, threshold, save_path, lower_is_better=False):
    """Lưu danh sách failure case + breakdown ra file JSON để review."""
    failures = find_low_score_cases(report, metric_key, threshold=threshold, lower_is_better=lower_is_better)
    breakdown = error_breakdown_by_case_type(report, metric_key, threshold, lower_is_better)

    output = {
        "module": report["module"],
        "metric": metric_key,
        "threshold": threshold,
        "num_failures": len(failures),
        "num_total": report["num_success"],
        "failure_rate": len(failures) / report["num_success"] if report["num_success"] else 0.0,
        "breakdown_by_case_type": breakdown,
        "failures": failures,
    }

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return output
