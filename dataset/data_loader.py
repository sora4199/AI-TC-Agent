import json
from pathlib import Path


# ============================================================
# DATA PATH
# ============================================================

# Lấy thư mục hiện tại của file test.py.
# Vì vậy chương trình có thể được chạy từ bất kỳ thư mục nào.
BASE_DIR = Path(__file__).resolve().parent

# File chứa các câu hỏi và đáp án.
TEST_FILE = BASE_DIR / "test.json"


# ============================================================
# LOAD DATA
# ============================================================

def load_test_data():
    """
    Load test dataset từ test.json.

    Returns
    -------
    list[dict]
        Danh sách các sample.

        Mỗi sample có dạng:
        {
            "id": "...",
            "question": "...",
            "answer": "..."
        }
    """

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


# ============================================================
# EXAMPLE
# ============================================================
#
# File khác có thể sử dụng:
#
#     from test import load_test_data
#
#     data = load_test_data()
#
# Sau đó:
#
#     sample = data[0]
#
#     question = sample["question"]
#     answer = sample["answer"]
#
# ============================================================
if __name__ == "__main__":
    data = load_test_data()

    print(f"Loaded {len(data)} samples")

    print(data[0])
"""
(myconda) sotatek@sotatek-Z790-D-DDR4:~/Projects/Tools/Notebook/Luong.Nguyen/NEU-AI/ai-espirit/dataset_test$ python data_loader.py
Loaded 1003 samples
{'id': '7407e25ef6180902d432faa084eb27c4cf0b25ba22c0df20bc5fbd6e2968ba4a', 'question': 'Đội bóng Boca Juniors mà Diego Maradona đã chơi có trụ sở tại La Boca, gần Buenos Aires phải không?', 'answer': 'đúng'}
"""