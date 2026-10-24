"""
Quản lý benchmark dataset: validate format chuẩn hóa + kiểm tra coverage
(đảm bảo mỗi module có đủ normal / edge / adversarial case).
"""

import json
from pathlib import Path

VALID_CASE_TYPES = {"normal", "edge", "adversarial"}

# field bắt buộc trong `ground_truth` của từng module - dùng để validate format
REQUIRED_GT_FIELDS = {
    "router": ["route"],
    "retriever": ["relevant_doc_ids"],
    "reranker": ["relevant_doc_ids"],
    "memory": ["relevant_memory_ids"],
    "agent": ["answer"],
    "rag": ["answer"],
    "tool_use": ["tool_calls"],
    "multistep": ["steps"],
    "calculation": ["result"],
    "fewshot": ["relevant_example_ids"],
}


def validate_dataset(dataset, module_name):
    """
    Kiểm tra format của toàn bộ dataset cho 1 module.
    Trả về list các lỗi (rỗng = dataset hợp lệ).
    """
    errors = []
    required_fields = REQUIRED_GT_FIELDS.get(module_name)

    if required_fields is None:
        return [f"Module '{module_name}' chưa được định nghĩa trong REQUIRED_GT_FIELDS."]

    for i, case in enumerate(dataset):
        case_id = case.get("id", f"index_{i}")

        if "input" not in case:
            errors.append(f"[{case_id}] thiếu field 'input'")
        if "ground_truth" not in case:
            errors.append(f"[{case_id}] thiếu field 'ground_truth'")
            continue

        gt = case["ground_truth"]
        for field in required_fields:
            if field not in gt:
                errors.append(f"[{case_id}] ground_truth thiếu field bắt buộc '{field}'")

        case_type = case.get("case_type", "normal")
        if case_type not in VALID_CASE_TYPES:
            errors.append(
                f"[{case_id}] case_type='{case_type}' không hợp lệ "
                f"(phải là một trong {sorted(VALID_CASE_TYPES)})"
            )

    return errors


def coverage_report(dataset):
    """Đếm số lượng case theo case_type, cảnh báo nếu thiếu loại nào."""
    counts = {t: 0 for t in VALID_CASE_TYPES}
    for case in dataset:
        case_type = case.get("case_type", "normal")
        counts[case_type] = counts.get(case_type, 0) + 1

    missing = [t for t in VALID_CASE_TYPES if counts.get(t, 0) == 0]

    return {
        "total": len(dataset),
        "by_case_type": counts,
        "missing_case_types": missing,
        "is_balanced": len(missing) == 0,
    }


def validate_and_report(path, module_name):
    """Tiện ích: load file, validate, in coverage - dùng khi thu thập dataset mới từ member."""
    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    errors = validate_dataset(dataset, module_name)
    coverage = coverage_report(dataset)

    print(f"=== VALIDATE DATASET: {path} (module={module_name}) ===")
    print(f"Tổng số case: {coverage['total']}")
    print(f"Theo case_type: {coverage['by_case_type']}")
    if coverage["missing_case_types"]:
        print(f"⚠️  THIẾU loại case: {coverage['missing_case_types']}")
    if errors:
        print(f"❌ {len(errors)} lỗi format:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("✅ Format hợp lệ.")

    return {"errors": errors, "coverage": coverage}


def save_versioned_dataset(dataset, module_name, version, out_dir="data/versions"):
    """Lưu 1 bản dataset kèm version, để track thay đổi qua thời gian."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{module_name}_v{version}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    return str(path)
