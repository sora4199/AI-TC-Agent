
import json
import hashlib
import ollama
import random
from pathlib import Path

try:
    DATA_FILE = Path(__file__).resolve().parent / 'Data.txt'

    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

        # Chia văn bản theo "---" trước
        raw_chunks = [
            chunk.strip()
            for chunk in content.split('---')
            if chunk.strip()
        ]

        # Tiếp tục cắt các chunk lớn thành các chunk nhỏ
        knowledge_chunks = []

        MAX_CHUNK_SIZE = 500

        for chunk in raw_chunks:
            if len(chunk) <= MAX_CHUNK_SIZE:
                knowledge_chunks.append(chunk)
            else:
                lines = chunk.splitlines()
                current_chunk = ""

                for line in lines:
                    line = line.strip()

                    if not line:
                        continue

                    if current_chunk and len(current_chunk) + len(line) + 1 > MAX_CHUNK_SIZE:
                        knowledge_chunks.append(current_chunk.strip())
                        current_chunk = line
                    else:
                        if current_chunk:
                            current_chunk += "\n" + line
                        else:
                            current_chunk = line

                if current_chunk:
                    knowledge_chunks.append(current_chunk.strip())

except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'Data.txt'")
    exit()

print(f"Đã tìm thấy {len(knowledge_chunks)} đoạn văn bản đầu vào. Bắt đầu xử lý...")

dataset = []
items_per_chunk = max(1, 200 // len(knowledge_chunks)) if knowledge_chunks else 15
num_multiset_samples = 30

for idx, text_context in enumerate(knowledge_chunks):
    prompt = f"""
    Bạn là một trợ lý AI tạo dữ liệu. Hãy dựa vào văn bản gốc dưới đây để tạo ra đúng {items_per_chunk} cặp Câu hỏi và Câu trả lời khác nhau.
    
    VĂN BẢN GỐC: \"\"\" {text_context} \"\"\"
    
    YÊU CẦU NGHIÊM NGẶT:
    1. Tạo ra {items_per_chunk} câu hỏi tự nhiên.
    2. Trả về duy nhất 1 mảng JSON chứa các object có 2 key chính xác là "question" và "answer".
    
    ĐỊNH DẠNG BẮT BUỘC:
    [
      {{"question": "Câu hỏi 1...", "answer": "Câu trả lời 1..."}},
      {{"question": "Câu hỏi 2...", "answer": "Câu trả lời 2..."}}
    ]
    """
    
    try:
        response = ollama.chat(
            model='qwen2.5:7b',
            messages=[{'role': 'user', 'content': prompt}],
            format='json'
        )
        
        raw_output = response['message']['content'].strip()
        parsed_data = json.loads(raw_output)
        
        # Tự động lấy danh sách nếu AI bọc JSON trong một object cha (ví dụ: {"data": [...]})
        ai_json_list = []
        if isinstance(parsed_data, list):
            ai_json_list = parsed_data
        elif isinstance(parsed_data, dict):
            for val in parsed_data.values():
                if isinstance(val, list):
                    ai_json_list = val
                    break
            if not ai_json_list:
                ai_json_list = [parsed_data]

        # Bóc tách dữ liệu linh hoạt
        for item in ai_json_list:
            if isinstance(item, dict):
                q = item.get("question") or item.get("Question") or item.get("cau_hoi")
                a = item.get("answer") or item.get("Answer") or item.get("tra_loi")
                
                if q and a:
                    hash_input = str(q).encode('utf-8')
                    generated_id = hashlib.sha256(
                        hash_input
                    ).hexdigest()
                    
                    dataset.append({
                        "id": generated_id,
                        "question": q,
                        "answer": a,
                        "gold_chunk": text_context
                    })
        
        print(f"✔ Done item {idx + 1}/{len(knowledge_chunks)} - Tổng số dataset hiện tại: {len(dataset)}")

    except Exception as e:
        print(f" Error compiling item {idx + 1}: {e}")


TEST_FILE = Path(__file__).resolve().parent / 'test.json'

with open(TEST_FILE, 'w', encoding='utf-8') as outfile:
    json.dump(dataset, outfile, ensure_ascii=False, indent=4)

print(f"Success! Đã tạo thành công {len(dataset)} dataset vào file 'test.json'.")
