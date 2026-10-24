"""
Đọc dataset test case từ file JSON.

Format file JSON kỳ vọng (list các test case):
[
    {
        "id": "case_001",
        "case_type": "normal" | "edge" | "adversarial",
        "input": { ... theo đúng Request schema của module (xem MODULE_SCHEMA.md) },
        "ground_truth": { ... }
    },
    ...
]
"""

import json


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Dataset {path} phải là 1 list các test case.")

    for i, case in enumerate(data):
        if "input" not in case or "ground_truth" not in case:
            raise ValueError(
                f"Test case thứ {i} trong {path} thiếu field 'input' hoặc 'ground_truth'."
            )
        case.setdefault("id", f"case_{i:04d}")
        case.setdefault("case_type", "normal")

    return data
