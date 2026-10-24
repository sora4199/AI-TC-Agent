"""
Converter TỔNG QUÁT: chuyển bất kỳ dataset nào (HuggingFace hoặc file JSON/CSV local)
sang đúng format pipeline cần, miễn là bạn khai báo đúng TÊN FIELD của dataset gốc.

Hỗ trợ 3 nguồn:
  - convert_from_huggingface()  : dataset trên HuggingFace Hub
  - convert_from_json()          : file .json local (list of dict)
  - convert_from_csv()           : file .csv local

Áp dụng cho module dạng TEXT (Agent/RAG - question/answer). Nếu bạn cần\
convert cho module khác (Router, Retriever...), xem phần "Convert cho module
khác" ở cuối file.
"""

import csv
import json
import random


def _sample_indices(total, limit, seed):
    indices = list(range(total))
    if limit is not None and limit < total:
        random.Random(seed).shuffle(indices)
        indices = indices[:limit]
        indices.sort()
    return indices


def _save(converted, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu {len(converted)} case -> {save_path}")
    return save_path


# ---------- Nguồn 1: HuggingFace dataset ----------

def convert_from_huggingface(
    dataset_name,
    question_field,
    answer_field,
    save_path,
    config_name=None,
    split="test",
    id_field=None,
    limit=None,
    seed=42,
):
    """
    VD: convert_from_huggingface(
            dataset_name="rag-datasets/rag-mini-bioasq",
            config_name="question-answer-passages",
            question_field="question",
            answer_field="answer",
            save_path="data/rag_bioasq.json",
            limit=100,
        )
    """
    from datasets import load_dataset as load_hf_dataset

    hf_dataset = load_hf_dataset(dataset_name, config_name) if config_name else load_hf_dataset(dataset_name)
    data_split = hf_dataset[split] if split in hf_dataset else list(hf_dataset.values())[0]

    total = len(data_split)
    indices = _sample_indices(total, limit, seed)

    converted = []
    for idx in indices:
        item = data_split[idx]
        converted.append({
            "id": str(item[id_field]) if id_field else str(idx),
            "case_type": "normal",
            "input": {"query": item[question_field]},
            "ground_truth": {"answer": item[answer_field]},
        })

    print(f"Tổng số case trong dataset gốc: {total}")
    return _save(converted, save_path)


# ---------- Nguồn 2: file JSON local ----------

def convert_from_json(
    input_path,
    question_field,
    answer_field,
    save_path,
    id_field=None,
    limit=None,
    seed=42,
):
    """
    VD: convert_from_json(
            input_path="test.json",
            question_field="question",
            answer_field="answer",
            save_path="data/rag_from_test_json.json",
        )
    """
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    total = len(raw_data)
    indices = _sample_indices(total, limit, seed)

    converted = []
    for idx in indices:
        item = raw_data[idx]
        converted.append({
            "id": str(item[id_field]) if id_field else str(idx),
            "case_type": "normal",
            "input": {"query": item[question_field]},
            "ground_truth": {"answer": item[answer_field]},
        })

    print(f"Tổng số case trong dataset gốc: {total}")
    return _save(converted, save_path)


# ---------- Nguồn 3: file CSV local ----------

def convert_from_csv(
    input_path,
    question_field,
    answer_field,
    save_path,
    id_field=None,
    limit=None,
    seed=42,
):
    """
    VD: convert_from_csv(
            input_path="my_qa.csv",
            question_field="query_text",
            answer_field="correct_answer",
            save_path="data/rag_from_csv.json",
        )
    """
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = list(csv.DictReader(f))

    total = len(raw_data)
    indices = _sample_indices(total, limit, seed)

    converted = []
    for idx in indices:
        item = raw_data[idx]
        converted.append({
            "id": str(item[id_field]) if id_field else str(idx),
            "case_type": "normal",
            "input": {"query": item[question_field]},
            "ground_truth": {"answer": item[answer_field]},
        })

    print(f"Tổng số case trong dataset gốc: {total}")
    return _save(converted, save_path)


if __name__ == "__main__":
    convert_from_json(
        input_path="../../dataset/test.json",      
        question_field="question",
        answer_field="answer",
        save_path="data/converterdata.json",
        limit=100,
    )


"""
---------- Convert cho MODULE KHÁC (không phải text QA) ----------

Router - dataset gốc có field {query, route}:
    converted.append({
        "id": ..., "case_type": "normal",
        "input": {"query": item["query"], "session_id": "s1", "user_id": None, "metadata": {}},
        "ground_truth": {"route": item["route"]},
    })

Retriever - dataset gốc có field {query, relevant_ids} (list):
    converted.append({
        "id": ..., "case_type": "normal",
        "input": {"query": item["query"], "top_k": 5},
        "ground_truth": {"relevant_doc_ids": item["relevant_ids"]},
    })

Cấu trúc luôn giống nhau: đọc field gốc -> nhét vào đúng "input"/"ground_truth"
theo bảng schema trong README.md (mục "Bảng module <-> evaluator").
"""
